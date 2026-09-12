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
                 'channel_mat', 'channel_mix', 'conv_mix', 'conv_mat', 'conv_diag', 'all_mix', 'shuffle',
                 'branch', 'block', 'stage', 'logit', 'stream', 'stream_dev',
                 # transformer (timm VisionTransformer): channel-last tensors [B, N, C]
                 'ln', 'ln_pre', 'ln_dev', 'head', 'head_dev', 'head_temp', 'mlp', 'mlp_dev')
_LAST = ('ln', 'ln_pre', 'ln_dev', 'mlp', 'mlp_dev')          # gates on the last dim


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
            xh = x.view(B, N, H, C // H)
            if self.gran[key] == 'head_dev':   # shrink toward the head's mean output vector
                mu = xh.mean((0, 1), keepdim=True).detach()
                return (xh + (g - 1) * (xh - mu)).view(B, N, C),
            return (xh * g).view(B, N, C),
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
            if gran in ('channel_mat', 'channel_mix', 'conv_mix', 'conv_mat', 'all_mix'):
                # matrix gate on the centred activation: y + A(y - ref), A = P - 1 (P pinned at 1)
                A = g - 1
                if gran not in ('channel_mat', 'conv_mat'):   # off-diagonal only
                    A = A - torch.diag(A.diagonal())
                shape = [-1 if i == 1 else 1 for i in range(out.dim())]
                if gran in ('channel_mat', 'channel_mix') and module.bias is not None:
                    ref = module.bias.detach().view(shape)   # BN affine bias = the channel mean
                else:                                        # no affine to read: use the batch mean
                    ref = out.mean([i for i in range(out.dim()) if i != 1], keepdim=True).detach()
                d = out - ref
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
            if gran in ('stream_dev', 'conv_diag'):   # shrink toward the batch mean, no norm layer needed
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

    def _conv_mix(self, model, dev):
        # mixing gate on conv outputs, i.e. BEFORE the normalisation that follows
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                self._add(m, 'cvX.' + n, (m.out_channels,) * 2, 'conv_mix', dev, n=m.out_channels)

    def _conv_mat(self, model, dev):
        # full C x C gate on conv outputs; differs from conv_mix only where no norm follows the conv
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                self._add(m, 'cvM.' + n, (m.out_channels,) * 2, 'conv_mat', dev, n=m.out_channels)

    def _conv_diag(self, model, dev):
        # per-channel mean-referenced gate on conv outputs: the diagonal of conv_mat on its own
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                self._add(m, 'cvD.' + n, m.out_channels, 'conv_diag', dev)

    def _all_mix(self, model, dev):
        # every named feature map: conv outputs, norm outputs, residual block outputs
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                c = m.out_channels
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                c = m.num_features
            elif _residual(m)[0] is not None:
                c = None
                for mm in _residual(m)[0].modules():
                    if isinstance(mm, nn.BatchNorm2d):
                        c = mm.num_features
                    elif isinstance(mm, nn.Conv2d):
                        c = mm.out_channels
            else:
                continue
            if c is not None:
                self._add(m, 'aX.' + n, (c, c), 'all_mix', dev, n=c)

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
    def _ln(self, model, dev):
        # zero-referenced LN gate: g*y, the direct analogue of ResNet's `channel`
        for n, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                self._add(m, 'ln.' + n, m.normalized_shape[-1], 'ln', dev)

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

    def _head_dev(self, model, dev):
        for n, m in model.named_modules():
            if all(hasattr(m, a) for a in ('qkv', 'proj', 'num_heads')):
                k = self._add(m.proj, 'hdd.' + n, m.num_heads, 'head_dev', dev, pre=True)
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
            adaptive=False, proj='none', envelope=False, ascent=True, **kwargs))
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def _direction(self, p, group):
        """(ascent direction, squared size in the group's metric) for one parameter."""
        g = p.grad
        proj = group.get('proj', 'none')
        if proj == 'none' or p.dim() < 2:
            v = (torch.abs(p) * g) if group['adaptive'] else g
            d = (torch.pow(p, 2) * g) if group['adaptive'] else g
            return d, v.pow(2).sum(), None
        C = p.shape[0]
        W = p.detach().reshape(C, -1)
        M = g.reshape(C, -1) @ W.t()                          # G W^T, C x C
        if proj == 'gl':           # free matrix: dW = A W, A unconstrained (= conv_mat in weight space)
            return (M @ W).reshape(p.shape), M.pow(2).sum(), M
        if proj == 'orbit':        # dW = A W with ||A||_F: the weight-space form of conv_mix
            M = M - torch.diag(M.diagonal())
            return (M @ W).reshape(p.shape), M.pow(2).sum(), M
        if proj == 'rot':          # pure rotation: A antisymmetric, W' = Cayley(A) W  (exact SO(C))
            K = 0.5 * (M - M.t())
            return (K @ W).reshape(p.shape), K.pow(2).sum(), K
        # 'tangent': Euclidean size of dW restricted to the orbit tangent space {A W}
        Winv = torch.linalg.solve(W @ W.t() + 1e-6 * torch.eye(C, device=p.device, dtype=p.dtype), W)
        P = M @ Winv                                          # G W^T (W W^T)^-1 W
        return P.reshape(p.shape), P.pow(2).sum(), None

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        dirs, mats, sq = {}, {}, {}
        for group in self._active():
            for p in group['params']:
                if p.grad is None:
                    continue
                d, n2, M = self._direction(p, group)
                dirs[p], mats[p] = d, M
                if group['scope'] is not None:
                    sq[group['scope']] = sq.get(group['scope'], 0.0) + n2
        for group in self._active():
            norm = sq.get(group['scope'])                      # scope None -> unnormalised
            scale = torch.tensor(group['rho']) if norm is None else group['rho'] / (norm.sqrt() + 1e-12)
            for p in group['params']:
                if p not in dirs:
                    continue
                if not group['is_gate']:
                    self.state[p]['old_p'] = p.data.clone()
                if group.get('proj', 'none') == 'rot':
                    A = mats[p] * scale.to(p)                        # antisymmetric, ||A||_F = rho share
                    I = torch.eye(A.shape[0], device=p.device, dtype=p.dtype)
                    Q = torch.linalg.solve(I - 0.5 * A, I + 0.5 * A)  # Cayley: Q in SO(C)
                    if group.get('envelope', False):
                        self.state[p]['Q'] = Q
                    if group.get('ascent', True):
                        C = p.shape[0]
                        p.data = (Q @ p.data.reshape(C, -1)).reshape(p.shape)
                    continue_rot = True
                else:
                    continue_rot = False
                if continue_rot:
                    continue
                if group.get('envelope', False):
                    if mats[p] is not None:
                        self.state[p]['A'] = mats[p] * scale.to(p)   # A* = rho M/||M||, for the descent step
                    elif group['adaptive'] and not group['is_gate']:
                        # ASAM: w' = w(1+eps) with e_w = w^2 grad scale  ->  w'/w = 1 + w grad scale
                        self.state[p]['f'] = 1 + p * p.grad * scale.to(p)
                if group.get('ascent', True):
                    p.add_(dirs[p] * scale.to(p))
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
                    f = self.state[p].pop('f', None)
                    if f is not None and p.grad is not None:
                        p.grad = p.grad * f                     # dL/dw = (w'/w) * dL/dw'
                    Q = self.state[p].pop('Q', None)
                    if Q is not None and p.grad is not None:
                        C = p.shape[0]
                        p.grad = (Q.t() @ p.grad.reshape(C, -1)).reshape(p.shape)   # dL/dW = Q^T dL/dW'
                    A = self.state[p].pop('A', None)
                    if A is not None and p.grad is not None:
                        # envelope term: dL/dW = (I+A)^T dL/dW' for W' = (I+A)W  (what the gate form does)
                        C = p.shape[0]
                        G = p.grad.reshape(C, -1)
                        p.grad = ((torch.eye(C, device=p.device, dtype=p.dtype) + A).t() @ G).reshape(p.shape)
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
    lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
    head_ids = {id(q) for q in lins[-1].parameters()} if lins else set()   # classifier: never rotated/mixed
    named = [('bn_scale', pick(lambda p: id(p) in bn_w)),
             ('bn_bias', pick(lambda p: id(p) in bn_b)),
             ('conv', pick(lambda p: id(p) not in bn_ids and p.dim() == 4)),
             ('linear', pick(lambda p: id(p) not in bn_ids and p.dim() == 2 and id(p) not in head_ids)),
             ('weight', pick(lambda p: id(p) not in bn_ids and (p.dim() == 3 or (p.dim() == 2 and id(p) in head_ids)))),
             ('bias', pick(lambda p: id(p) not in bn_ids and p.dim() < 2))]

    # which weight-space coordinates the adversary may use, and in which metric
    #   conv    : SAM on conv weights only (Euclidean)
    #   tangent : SAM on conv weights, projected onto the orbit tangent space {A W} (Euclidean)
    #   orbit   : dW = A W with ||A||_F = rho -- conv_mix in weight space (no gates needed)
    mats = ('conv', 'linear')            # every matrix-shaped weight except the classifier
    arm = {'none': (), 'all': ('bn_scale', 'bn_bias', 'conv', 'linear', 'weight', 'bias'),
           'bn': ('bn_scale', 'bn_bias'),
           'bn_scale': ('bn_scale',), 'bn_bias': ('bn_bias',),
           'conv': ('conv',), 'tangent': mats, 'orbit': mats, 'rot': mats, 'gl': mats}[args.perturb]
    proj = args.perturb if args.perturb in ('tangent', 'orbit', 'rot', 'gl') else 'none'

    groups = [dict(params=ps, name=n, rho=args.rho if n in arm else 0.0,
                   perturb=n in arm and args.rho > 0, scope='w', proj=proj,
                   envelope=bool(getattr(args, 'envelope', False)),
                   ascent=not getattr(args, 'no_ascent', False),
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

    kw = dict(lr=args.lr, weight_decay=args.weight_decay)
    if base_optimizer is torch.optim.SGD:
        kw.update(momentum=args.momentum, nesterov=False)
    opt = PGSAM(groups, base_optimizer, **kw)
    opt.bank = bank
    if verbose:
        for g in opt.param_groups:
            print('  %-14s n=%-9d perturb=%-5s rho=%-8.4g scope=%-12s proj=%s'
                  % (g['name'], sum(p.numel() for p in g['params']), g['perturb'],
                     g['rho'], g['scope'], g['proj']))
    return opt
