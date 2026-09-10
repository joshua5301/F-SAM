"""PG-SAM: SAM restricted to phantom gates.

A phantom gate is a multiplicative parameter pinned at 1 (hooked onto a module
output, never updated), so the network is unchanged and dL/dg_u = <dL/da_u, a_u>
is the ablation saliency of unit u -- gauge invariant and dimensionless, hence
comparable across units.
"""

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

GRANULARITIES = ('channel', 'channel_pre', 'channel_pre_write', 'channel_pre_mid',
                 'channel_pre_front', 'channel_pre_back', 'channel_shift',
                 'channel_mat', 'channel_mix', 'shuffle',
                 'branch', 'block', 'stage', 'logit', 'stream', 'stream_dev',
                 # transformer (timm VisionTransformer): channel-last tensors [B, N, C]
                 'ln_pre', 'ln_dev', 'head', 'head_temp', 'mlp', 'mlp_dev')
_LAST = ('ln_pre', 'ln_dev', 'mlp', 'mlp_dev')          # gates on the last dim


def _bns(model):
    """all BN modules in definition (= depth) order."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]


def _mid_bns(model):
    """ids of BNs internal to a residual branch (every BN but the branch's last)."""
    mids = set()
    for _, m in model.named_modules():
        res, _ = _residual(m)
        if res is None:
            continue
        bns = [b for b in res.modules() if isinstance(b, (nn.BatchNorm2d, nn.BatchNorm1d))]
        mids |= {id(b) for b in bns[:-1]}
    return mids


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
        self.n = {}              # size used for rho_g = gate_rho*sqrt(N_g) (channels, not entries)
        self._heads = {}         # head gates: num_heads per key
        self._perm = {}          # shuffle: batch permutation, held fixed while perturbed
        self.perturbed = False   # set by PGSAM between first_step and second_step
        dev = next(model.parameters()).device
        for g in [s.strip() for s in str(spec).split(',') if s.strip()]:
            assert g in GRANULARITIES, 'unknown granularity %r' % g
            getattr(self, '_' + g)(model, dev)
        assert len(self.gates), 'no gate attached for %r' % spec

    def _add(self, module, key, size, gran, dev, n=None, pre=False):
        key = key.replace('.', '_')
        self.gates[key] = nn.Parameter(torch.ones(size, device=dev))
        self.gran[key] = gran
        self.n[key] = self.gates[key].numel() if n is None else n
        if pre:
            module.register_forward_pre_hook(self._pre_hook(key))
        else:
            module.register_forward_hook(self._hook(key))
        return key

    def _pre_hook(self, key):
        # per-head scalar gate on the input of attn.proj: [B, N, C] laid out head-major
        def hook(module, inputs):
            x = inputs[0]
            B, N, C = x.shape
            H = self._heads[key]
            g = self.gates[key].view(1, 1, H, 1)
            return (x.view(B, N, H, C // H) * g).view(B, N, C),
        return hook

    def _hook(self, key):
        def hook(module, inputs, out):
            if not torch.is_tensor(out):
                return None
            g = self.gates[key]
            gran = self.gran[key]
            if gran in _LAST:
                g = g.view([1] * (out.dim() - 1) + [-1])
                if gran == 'ln_pre':       # gamma-scaling of LN's x-hat: beta + g(y - beta)
                    ref = module.bias.detach() if module.bias is not None else 0.0
                    return out + (g - 1) * (out - ref)
                if gran in ('ln_dev', 'mlp_dev'):   # shrink toward the (batch x token) channel mean
                    mu = out.mean(tuple(range(out.dim() - 1)), keepdim=True).detach()
                    return out + (g - 1) * (out - mu)
                return out * g             # mlp
            if gran == 'head_temp':        # scale q per head inside qkv's output: attention temperature
                H = self._heads[key]
                B, N, C3 = out.shape
                C = C3 // 3
                q = out[..., :C].reshape(B, N, H, C // H) * g.view(1, 1, H, 1)
                return torch.cat([q.reshape(B, N, C), out[..., C:]], dim=-1)
            if gran in ('channel_mat', 'channel_mix'):
                # matrix gate on the standardized deviation: y + A(y - beta), A = P - 1 (P pinned at 1)
                A = g - 1
                if gran == 'channel_mix':                     # off-diagonal only
                    A = A - torch.diag(A.diagonal())
                ref = module.bias.detach() if module.bias is not None else 0.0
                d = out - (ref.view([-1 if i == 1 else 1 for i in range(out.dim())])
                           if torch.is_tensor(ref) else ref)
                mixed = torch.einsum('ab,nbhw->nahw', A, d) if out.dim() == 4 else d @ A.t()
                return out + mixed
            if g.numel() > 1:
                g = g.view([-1 if d == 1 else 1 for d in range(out.dim())])
            if gran == 'channel_shift':    # additive gate on x-hat: threshold shift in sigmas
                w = module.weight.detach().view(g.shape) if module.weight is not None else 1.0
                return out + (g - 1) * w
            if gran.startswith('channel_pre'):  # shrink toward the channel mean (= BN bias)
                ref = module.bias.detach().view(g.shape) if module.bias is not None else 0.0
                return out + (g - 1) * (out - ref)
            if gran == 'shuffle':          # shrink toward a batch-shuffled self
                p = self._perm.get(key)
                if not self.perturbed or p is None or p.numel() != out.shape[0]:
                    p = torch.randperm(out.shape[0], device=out.device)
                    self._perm[key] = p
                return out + (g - 1) * (out - out[p]).detach()
            if gran == 'stream_dev':       # shrink toward the batch mean, no norm layer needed
                dims = [d for d in range(out.dim()) if d != 1]
                mu = out.mean(dims, keepdim=True).detach()
                return out + (g - 1) * (out - mu)
            return out * g
        return hook

    def _channel(self, model, dev):
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self._add(m, 'ch.' + n, m.num_features, 'channel', dev)

    def _channel_pre(self, model, dev):
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self._add(m, 'chp.' + n, m.num_features, 'channel_pre', dev)

    def _channel_pre_write(self, model, dev):
        # write ports only: branch-terminal BNs, projection-skip BNs, stem
        mids = _mid_bns(model)
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) and id(m) not in mids:
                self._add(m, 'chpw.' + n, m.num_features, 'channel_pre_write', dev)

    def _channel_pre_mid(self, model, dev):
        # internal BNs only: inside a branch, feeding the branch's own next conv
        mids = _mid_bns(model)
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) and id(m) in mids:
                self._add(m, 'chpm.' + n, m.num_features, 'channel_pre_mid', dev)

    def _channel_shift(self, model, dev):
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self._add(m, 'chs.' + n, m.num_features, 'channel_shift', dev)

    def _channel_mat(self, model, dev):
        # C x C gate per BN; rho sized by channel count so gate_rho matches channel_pre's scale
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                c = m.num_features
                self._add(m, 'chM.' + n, (c, c), 'channel_mat', dev, n=c)

    def _channel_mix(self, model, dev):
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                c = m.num_features
                self._add(m, 'chX.' + n, (c, c), 'channel_mix', dev, n=c)

    def _channel_pre_front(self, model, dev):
        # first half of the BNs in depth order (resnet18: stem + conv2_x + conv3_x)
        bns = _bns(model)
        for n, m in bns[:len(bns) // 2]:
            self._add(m, 'chpf.' + n, m.num_features, 'channel_pre_front', dev)

    def _channel_pre_back(self, model, dev):
        # second half of the BNs in depth order (resnet18: conv4_x + conv5_x)
        bns = _bns(model)
        for n, m in bns[len(bns) // 2:]:
            self._add(m, 'chpb.' + n, m.num_features, 'channel_pre_back', dev)

    def _shuffle(self, model, dev):
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self._add(m, 'shf.' + n, m.num_features, 'shuffle', dev)

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

    def _stream_dev(self, model, dev):
        # same position as stream, but mean-referenced: out -> mu + g*(out - mu)
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
            self._add(m, 'smd.' + n, c, 'stream_dev', dev)

    # ---- transformer granularities (duck-typed on timm's Attention / Mlp) ----
    def _ln_pre(self, model, dev):
        for n, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                self._add(m, 'lnp.' + n, m.normalized_shape[-1], 'ln_pre', dev)

    def _ln_dev(self, model, dev):
        for n, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                self._add(m, 'lnd.' + n, m.normalized_shape[-1], 'ln_dev', dev)

    def _head(self, model, dev):
        for n, m in model.named_modules():
            if all(hasattr(m, a) for a in ('qkv', 'proj', 'num_heads')):
                k = self._add(m.proj, 'hd.' + n, m.num_heads, 'head', dev, pre=True)
                self._heads[k] = m.num_heads

    def _head_temp(self, model, dev):
        for n, m in model.named_modules():
            if all(hasattr(m, a) for a in ('qkv', 'proj', 'num_heads')):
                k = self._add(m.qkv, 'ht.' + n, m.num_heads, 'head_temp', dev)
                self._heads[k] = m.num_heads

    def _mlp(self, model, dev):
        for n, m in model.named_modules():
            if all(hasattr(m, a) for a in ('fc1', 'act', 'fc2')):
                self._add(m.act, 'mlp.' + n, m.fc1.out_features, 'mlp', dev)

    def _mlp_dev(self, model, dev):
        for n, m in model.named_modules():
            if all(hasattr(m, a) for a in ('fc1', 'act', 'fc2')):
                self._add(m.act, 'mlpd.' + n, m.fc1.out_features, 'mlp_dev', dev)

    def by_gran(self):
        out = {}
        for k in self.gates:
            out.setdefault(self.gran[k], []).append(k)
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
        bank = getattr(self, 'bank', None)
        if bank is not None:
            bank.perturbed = True
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
        bank = getattr(self, 'bank', None)
        if bank is not None:
            bank.perturbed = False
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

    norm = (_BatchNorm, nn.LayerNorm)        # 'bn' arms = normalisation affine, BN or LN
    bn_w = {id(m.weight) for m in model.modules()
            if isinstance(m, norm) and m.weight is not None}
    bn_b = {id(m.bias) for m in model.modules()
            if isinstance(m, norm) and m.bias is not None}
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
        for gran, ks in bank.by_gran().items():
            ps = [bank.gates[k] for k in ks]
            n = sum(bank.n[k] for k in ks)
            rho = eps.get(gran, 0.0) if isinstance(eps, dict) else eps
            if args.gate_norm != 'none':
                rho *= n ** 0.5           # rho_g = gate_rho*sqrt(N_g): per-coordinate RMS
            groups.append(dict(params=ps, name='gate:' + gran, rho=rho, perturb=rho > 0,
                               is_gate=True, adaptive=False, lr=0.0, weight_decay=0.0,
                               scope=None if args.gate_norm == 'none' else
                                     ('gates' if args.gate_norm == 'global' else 'gate:' + gran)))

    opt = PGSAM(groups, base_optimizer, lr=args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay, nesterov=False)
    opt.bank = bank
    if verbose:
        for g in opt.param_groups:
            print('  %-14s n=%-9d perturb=%-5s rho=%-8.4g scope=%s'
                  % (g['name'], sum(p.numel() for p in g['params']), g['perturb'],
                     g['rho'], g['scope']))
    return opt
