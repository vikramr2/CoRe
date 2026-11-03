# Study `hybrid-columns-4`: Map-STDP columns in the TD-STDP backbone on CartPole at n = 40, both bases

**Pre-registered 2026-10-29 12:03 EDT, before any run of this study** (approved by the user; code at the commit recorded in `manifest.json`, the first commit after this stamp, run from worktree `../map-stdp-hc4`, branch `hybrid-columns-4-study`). The design and tests below are fixed.

Nothing in this directory is written by hand except this README. `manifest.json`, `code.patch`, `runs/`, `summary.*` and `pairs.csv` come only from `python -m experiments.report`.

## What the study is

- **Question (user, 2026-10-29):** on CartPole, do the three regimes of action columns keep performance against the backbone, with enough seeds to decide it?
- **Why it is needed:** `hybrid-columns-3` (016 §17) left two of CartPole's three headline contrasts inconclusive (`cols_on` −12.8 [−28.5, +2.4], `cols_sealed` −8.5 [−25.9, +6.7] AUC at n = 20) and `cols_lam0` non-inferior by 6 points. The CI half-width at n = 20 is about 15 AUC points, so at the margin Δ = 25 any shift below about −10 reads inconclusive whether or not it is real. At n = 40 the half-width is about 10, which decides a shift of −12 either way. CartPole runs take 3–5 minutes, so the power is cheap.
- **Companion:** `hybrid-columns-3-ck` (the CK-faithful base on three tasks) runs at the same time; this study carries that base on CartPole too, so the published-backbone contrast is also decided at n = 40.
- **Design record:** `docs/iterations/016-columns-more-tasks.md` §19.

## Design

- **Common flags, every arm:** `--engine nlif --rate-target 40 --block 20 --fast --rate-match-rec --ent-match-rec`; CK's preset η (CartPole's preset η are SGD learning rates; `--eta-cal` silences the actor, 016 §15). Exactly `hybrid-columns-3`'s CartPole settings.
- **Arms** (identical except for):

| Arm | Flags | What it is |
| --- | --- | --- |
| `base` | `--norm-logits` | no recurrence, normalised scores: `hybrid-columns-3`'s base, the study's reference |
| `cols_lam0` | `--norm-logits --chi-rec 0.3 --rec cols --lam 0` | implicit: recurrent action columns, task term on R only |
| `cols_on` | `--norm-logits --chi-rec 0.3 --rec cols --lam 1` | learned: columns plus the map term (r = 1) |
| `cols_sealed` | `--norm-logits --chi-rec 0.3 --rec cols --rec-seal --lam 0` | enforced: the same draws of R without its cross-column synapses |
| `base_ck` | – | no recurrence, CK's softmax on the unnormalised scores: the published backbone at the matched start (`hybrid-columns-3-ck`'s arm) |

- **Frozen settings:** χ_rec 0.3 and r = 1, as in the earlier studies; the matching rules as in `hybrid-columns-3`.
- **Episodes:** 400, as CartPole's preset.
- **Groups:** `CartPole_<arm>`: 5 groups × 40 seeds = 200 runs.
- **Seeds:** 6000–6039, fresh. No earlier run used seeds at or above 6000.

## Pre-registered tests

**Primary metric:** AUC, the mean return over the 400 episodes.

**Contrasts:** two-sided Mann–Whitney on the primary metric, Holm-adjusted within each family. Effect size: Hodges–Lehmann shift with a 95% CI.

| Family | Contrast | Asks |
| --- | --- | --- |
| **headline** (`primary`, 3 tests) | `cols_lam0` vs `base` | does implicit modularity keep or improve performance? |
| | `cols_on` vs `base` | learned modularity (Map-STDP) |
| | `cols_sealed` vs `base` | enforced modularity |
| published backbone (`ck`, 4 tests) | `cols_lam0`, `cols_on`, `cols_sealed` vs `base_ck` | each regime against the backbone as published |
| | `base` vs `base_ck` | the normalised readout's own cost on CartPole |
| secondary (`secondary`, 3 tests) | `cols_on` vs `cols_lam0`; `cols_on` vs `cols_sealed`; `cols_lam0` vs `cols_sealed` | between the regimes |

For each headline and `ck` contrast:
- **Improves:** Holm-adjusted p < 0.05 and a positive shift.
- **Keeps performance (non-inferiority):** the lower end of the 95% CI of the shift is above −Δ, with **Δ = 25** AUC, as in the earlier studies.
- **Worse:** a significant negative shift.

**Pre-registered expectations** (from `hybrid-columns-3` and the §18 pilot, n = 20 and n = 10; expectations, not results): all three regimes non-inferior to both bases; `base` non-inferior to `base_ck`; the regimes do not differ. The §18 pilot's hint that sealed columns do better under the race readout is about the race, not this readout, and is not a prediction here.

**Secondary (reported, not tested for the claims):** solved count, T_s, last-100 return; the matching record per run (`low_rate`); `rec.RK` and `rec.RKc` per snapshot (the fixed monitor, 016 §15: causal counter per executed action); stability monitors (max Re λ, τ_loop, χG), rates, policy entropy; ms per frame. The actor-rate decline after solving (016 §17: column arms fall to 3–4 Hz with about 20 silent neurons) is recorded per arm from the logs.

**Not done:** no settings change after launch, and no seeds are added or dropped. A crashed run is relaunched from its checkpoint. If a group still has fewer than 40 runs, it is reported with its n.

## Run conditions

- **Machine:** the local workstation (72 cores), with `OMP`, `MKL`, `OPENBLAS` and `NUMBA` threads set to 1, 50 workers (launched before `hybrid-columns-3-ck`'s LunarLander runs settle in, so the two studies share the machine without oversubscription).
- **Code state:** git worktree `../map-stdp-hc4` (branch `hybrid-columns-4-study`) at the commit recorded in `manifest.json`.
- **Projected time:** about 30 min (3–5 min per run; column arms 5 min).
- **One code state:** `manifest.json` records the git HEAD and the sha256 of the uncommitted code; the runner refuses to relaunch if it differs.

## Commands

```bash
python -m experiments.report run hybrid-columns-4 --workers 50   # launch or resume every declared run; then summarize
python -m experiments.report summarize hybrid-columns-4          # summary.csv, pairs.csv and summary.md from runs/
```

For example:

```bash
python -m experiments.tdstdp --task CartPole --seed 6000 --engine nlif --rate-target 40 --block 20 --fast --rate-match-rec --ent-match-rec --norm-logits --chi-rec 0.3 --rec cols --lam 1 --out reports/hybrid-columns-4/runs/CartPole_cols_on/s6000.json
```
