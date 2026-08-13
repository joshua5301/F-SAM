# PG-SAM — Phantom-Gate SAM

Gates `g = 1` are hooked onto module outputs at a chosen granularity; the SAM
adversary moves only the gates, the base optimizer never touches them.

```
y = g * f(x),   g == 1 outside the ascent step
dL/dg_u = <dL/da_u, a_u>        # ablation saliency of unit u
```

The network is unchanged: gates stay out of the checkpoint, out of weight decay,
and cost nothing at inference (with `--gate-rho 0` the trajectory is
bit-identical to not having them). What they buy is a coordinate system —
`dL/dg` is dimensionless and invariant to the rescaling gauge `a_u -> alpha a_u`,
so an l2 ball over gates allocates its budget by saliency instead of by
parametrisation artefacts (init variance, fan-in, layer type), which is what a
weight-space ball does.

Everything lives in [`pgsam.py`](pgsam.py) (`GateBank`, `PGSAM`, `build_pgsam`);
[`train.py`](train.py) gains five flags and an `--optimizer PGSAM` branch.

## Flags

```
--gates channel,branch    granularities: channel branch block stage logit
--gate-rho 0.05           per-coordinate RMS gate perturbation (rho_g = gate_rho*sqrt(N_g)),
                          or a per-granularity spec "channel:0.05,branch:0.1"
--gate-norm global|group  global (default): one shared normaliser, the gradient decides
                          how the budget splits across granularities; group: a fixed
                          radius per granularity
--perturb none|all|bn     weight-space arm: none = pure PG-SAM, all = SAM, bn = SAM-ON
--rho 0.2                 weight-space radius (used when --perturb != none)
--adaptive                ASAM, i.e. a gate on every weight element. Weight groups
                          only -- it is an exact no-op on gates, which are already 1
```

`gate_rho` is an RMS so that "every gate is wiggled by 5%" means the same thing
for 4800 channel gates and 16 branch gates — the sqrt(N) correction that puts
granularities in one currency.

With a single granularity the two `--gate-norm` modes are identical. They differ
only when several granularities (or the weights) are perturbed at once:

    global:  e_g = rho_g * grad_g / ||grad_all||     rho_g = gate_rho*sqrt(N_g)
    group:   e_g = rho_g * grad_g / ||grad_g||

So `global` lets a granularity claim more budget when it is more sensitive,
while still discounting coarse gates by sqrt(N_g) — otherwise 16 branch gates
swamp 4800 channel gates (at init they carry ~3x the squared gradient mass).
`group` pins each granularity to a fixed radius regardless of the gradient.

`global` is shared by the *gates* only; weight groups always keep their own ball.
Putting both in one is the incommensurable case the gate coordinates exist to
avoid, and it degenerates in practice: 11.2M weight gradients drown out 4800 gate
gradients, cutting the gate perturbation from 5% to 0.1%.

## Arms (CIFAR-100 / ResNet-18, 200 ep, cutout, cosine, wd 1e-3)

| arm | flags | decides |
|---|---|---|
| SGD | `--perturb none --rho 0` | floor |
| SAM | `--perturb all --rho 0.2` | reference |
| SAM-ON | `--perturb bn --rho 0.5` | published gate-like method |
| ASAM | `--perturb all --rho 1.0 --adaptive` | the per-weight gate: finest granularity |
| **PG-SAM(ch)** | `--gates channel --gate-rho 0.05` | pure usage direction, no BN affine confound |
| **PG-SAM(br)** | `--gates branch --gate-rho 0.05` | module-level over-reliance |
| **PG-SAM(ch+br)** | `--gates channel,branch` | does granularity compose? (auto-allocated) |
| fixed per-granularity | `+ --gate-norm group` | is auto-allocation better than a fixed split? |
| SAM + PG-SAM | `--perturb all --rho 0.2 --gates channel --gate-rho 0.05` | do the two add up? |

Sweep `--gate-rho` over `{0.02, 0.05, 0.1, 0.2}` first — it is the only knob that
matters and its scale is interpretable.

## Two facts to keep in mind

Both were verified numerically on this repo's ResNet; both follow from BN making
the loss invariant to the scale of a pre-BN activation.

1. **Weight space cannot express a usage perturbation.** Scaling the weights of
   a unit that feeds a BN changes nothing, so that direction is flat and its
   gradient component is exactly zero (measured: 0.0000 against a gradient norm
   of 4.10). Weight-space SAM therefore spends its whole budget on *what units
   compute*, never on *how much they are used* — the two methods perturb nearly
   orthogonal subspaces, and the gate is the only handle on the second one.

2. **Whole-module scale is gauge, so only relative mix carries signal.** Branch
   and shortcut saliencies come out equal and opposite, and `stage`/`block`
   gates are near-zero except at the ends of the network, because the next BN
   re-normalises whatever a gate scales. The adversary handles this for free — a
   flat direction gets zero gradient, so the ascent step already lives in the
   quotient — but it means `stage`/`block` granularity buys little in a BN net,
   and that signed depth profiles like `s_F/(s_F+s_s)` are unusable; compare
   magnitudes instead. The Euler-identity / saliency-conservation argument does
   not transfer to normalised networks.

Also note that gates overlap by construction: since a residual branch ends in a
BN, the branch gate is exactly the all-ones direction inside that BN's channel
gates. `--gates channel,branch` is therefore not an orthogonal decomposition but
a separate price tag on the coherent direction — which is the point of
`--gate-rho "channel:0.05,branch:0.1"`.
