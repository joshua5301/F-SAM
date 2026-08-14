"""PG-SAM: SAM restricted to phantom gates.

A phantom gate is a multiplicative parameter pinned at 1 (hooked onto a module
output, never updated), so the network is unchanged and dL/dg_u = <dL/da_u, a_u>
is the ablation saliency of unit u -- gauge invariant and dimensionless, hence
comparable across units.
"""

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

GRANULARITIES = ('channel', 'branch', 'block', 'stage', 'logit', 'stream')


def _residual(m):
    """(branch, shortcut) for known residual blocks, else (None, None)."""
    if hasattr(m, 'residual_function') and hasattr(m, 'shortcut'):
        return m.residual_function, m.shortcut
    if hasattr(m, 'block') and hasattr(m, 'downsample'):
        return m.block, m.downsample
    if type(m).__name__ == 'BasicUnit' and hasattr(m, 'block'):
        return m.block, None
    return None, None


class GateBank(nn.Module):
    def __init__(self, model, spec):
        super().__init__()
        self.gates = nn.ParameterDict()
        self.gran = {}
        dev = next(model.parameters()).device
        for g in [s.strip() for s in str(spec).split(',') if s.strip()]:
            assert g in GRANULARITIES, 'unknown granularity %r' % g
            getattr(self, '_' + g)(model, dev)
        assert len(self.gates), 'no gate attached for %r' % spec

    def _add(self, module, key, size, gran, dev):
        key = key.replace('.', '_')
        self.gates[key] = nn.Parameter(torch.ones(size, device=dev))
        self.gran[key] = gran
        module.register_forward_hook(self._hook(key))

    def _hook(self, key):
        def hook(module, inputs, out):
            if not torch.is_tensor(out):
                return None
            g = self.gates[key]
            if g.numel() == 1:
                return out * g
            return out * g.view([-1 if d == 1 else 1 for d in range(out.dim())])
        return hook

    def _channel(self, model, dev):
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self._add(m, 'ch.' + n, m.num_features, 'channel', dev)

    def _branch(self, model, dev):
        for n, m in model.named_modules():
            res, skip = _residual(m)
            if res is None:
                continue
            self._add(res, 'br.' + n + '.res', 1, 'branch', dev)
            if skip is not None:
                self._add(skip, 'br.' + n + '.skip', 1, 'branch', dev)

    def _block(self, model, dev):
        for n, m in model.named_modules():
            if _residual(m)[0] is not None:
                self._add(m, 'blk.' + n, 1, 'block', dev)

    def _stage(self, model, dev):
        root = model.f if isinstance(getattr(model, 'f', None), nn.Sequential) else model
        for n, c in root.named_children():
            if not isinstance(c, nn.Linear) and any(True for _ in c.parameters()):
                self._add(c, 'st.' + n, 1, 'stage', dev)

    def _logit(self, model, dev):
        self._add(model, 'logit', 1, 'logit', dev)

    def _stream(self, model, dev):
        # per-channel gate on the block output (post-addition, post-ReLU)
        for n, m in model.named_modules():
            res, _ = _residual(m)
            if res is None:
                continue
            c = None
            for mm in res.modules():
                if isinstance(mm, nn.BatchNorm2d):
                    c = mm.num_features
                elif isinstance(mm, nn.Conv2d):
                    c = mm.out_channels
            self._add(m, 'sm.' + n, c, 'stream', dev)

    def by_gran(self):
        out = {}
        for k in self.gates:
            out.setdefault(self.gran[k], []).append(self.gates[k])
        return out


class PGSAM(torch.optim.Optimizer):
    """Param groups carry: perturb, rho, scope (id of the shared l2 ball),
    is_gate, adaptive."""

    def __init__(self, param_groups, base_optimizer, **kwargs):
        super(PGSAM, self).__init__(param_groups, dict(
            rho=0.0, perturb=False, scope='w', is_gate=False,
            adaptive=False, **kwargs))
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        norms = self._scope_norms()
        for group in self._active():
            norm = norms.get(group['scope'])          # scope None -> unnormalised
            scale = torch.tensor(group['rho']) if norm is None else group['rho'] / (norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                if not group['is_gate']:
                    self.state[p]['old_p'] = p.data.clone()
                p.add_((torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            if group['is_gate']:
                for p in group['params']:
                    p.data.fill_(1.0)                       # pinned at 1, no drift
            elif group['perturb'] and group['rho']:
                for p in group['params']:
                    old = self.state[p].pop('old_p', None)
                    if old is not None:
                        p.data = old
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, 'PG-SAM requires a closure'
        self.first_step(zero_grad=True)
        torch.enable_grad()(closure)()
        self.second_step()

    def _active(self):
        return [g for g in self.param_groups if g['perturb'] and g['rho']]

    @torch.no_grad()
    def _scope_norms(self):
        sq = {}
        for group in self._active():
            if group['scope'] is None:
                continue
            for p in group['params']:
                if p.grad is None:
                    continue
                v = (torch.abs(p) * p.grad) if group['adaptive'] else p.grad
                sq[group['scope']] = sq.get(group['scope'], 0.0) + v.pow(2).sum()
        return {k: v.sqrt() for k, v in sq.items()}

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def build_pgsam(model, args, base_optimizer=torch.optim.SGD, verbose=True):
    """--gate-rho is a per-coordinate RMS: rho_g = gate_rho * sqrt(N_g), so 0.05
    means every gate is wiggled by ~5% whatever the granularity. It accepts a
    float or a per-granularity spec, e.g. "channel:0.05,branch:0.1".
    """
    bank = GateBank(model, args.gates) if args.gates else None

    bn_w = {id(m.weight) for m in model.modules()
            if isinstance(m, _BatchNorm) and m.weight is not None}
    bn_b = {id(m.bias) for m in model.modules()
            if isinstance(m, _BatchNorm) and m.bias is not None}
    bn_ids = bn_w | bn_b
    pick = lambda f: [p for p in model.parameters() if f(p)]
    named = [('bn_scale', pick(lambda p: id(p) in bn_w)),
             ('bn_bias', pick(lambda p: id(p) in bn_b)),
             ('weight', pick(lambda p: id(p) not in bn_ids and p.dim() >= 2)),
             ('bias', pick(lambda p: id(p) not in bn_ids and p.dim() < 2))]

    # which weight-space coordinates the adversary may use
    arm = {'none': (), 'all': ('bn_scale', 'bn_bias', 'weight', 'bias'),
           'bn': ('bn_scale', 'bn_bias'),
           'bn_scale': ('bn_scale',), 'bn_bias': ('bn_bias',)}[args.perturb]

    groups = [dict(params=ps, name=n, rho=args.rho if n in arm else 0.0,
                   perturb=n in arm and args.rho > 0, scope='w',
                   is_gate=False, adaptive=bool(args.adaptive),
                   lr=args.lr, weight_decay=args.weight_decay)
              for n, ps in named if ps]

    if bank is not None:
        spec = str(args.gate_rho)
        eps = ({k.strip(): float(v) for k, v in (i.split(':') for i in spec.split(','))}
               if ':' in spec else float(spec))
        for gran, ps in bank.by_gran().items():
            n = sum(p.numel() for p in ps)
            rho = eps.get(gran, 0.0) if isinstance(eps, dict) else eps
            if args.gate_norm != 'none':
                rho *= n ** 0.5           # rho_g = gate_rho*sqrt(N_g): per-coordinate RMS
            groups.append(dict(params=ps, name='gate:' + gran, rho=rho, perturb=rho > 0,
                               is_gate=True, adaptive=False, lr=0.0, weight_decay=0.0,
                               scope=None if args.gate_norm == 'none' else
                                     ('gates' if args.gate_norm == 'global' else 'gate:' + gran)))

    opt = PGSAM(groups, base_optimizer, lr=args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay, nesterov=False)
    if verbose:
        for g in opt.param_groups:
            print('  %-14s n=%-9d perturb=%-5s rho=%-8.4g scope=%s'
                  % (g['name'], sum(p.numel() for p in g['params']), g['perturb'],
                     g['rho'], g['scope']))
    return opt
