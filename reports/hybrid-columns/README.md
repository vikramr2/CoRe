# Study `hybrid-columns`: Map-STDP columns in the TD-STDP backbone, without Adam

**Pre-registered 2026-10-27 20:02 EDT, before any run of this study.** Code at git `0d2d851`, branch `hybrid-columns-study` (worktree `../map-stdp-hc`). Design and tests below are fixed.

Nothing in this directory is written by hand except this README: `manifest.json`, `code.patch`, `runs/`, `summary.*` and `pairs.csv` come only from `python -m experiments.report`.

## What the study is

- **Backbone:** the TD-STDP actor-critic of Chung & Kozma 2020 (iteration 013, `reports/tdstdp-baseline`), without Adam.
- **The question (user, 2026-10-27):** does modular organisation of the actor keep or improve performance without Adam, and which kind of modularity? Each action's actor E population is one module (a "column"), with recurrent E→E synapses R between all actor E posts. The input→actor W is CK's, learned by TD alone. Three regimes of modularity are compared on R:
  - **implicit:** the task term alone shapes R;
  - **learned:** the task term plus the map term (Map-STDP);
  - **enforced:** R built without cross-column synapses, task term alone.
- **Engine:** noisy LIF (`--engine nlif`, dt 1 ms, τ_m 20 ms, τ_ref 2 ms) in the low-rate regime (`--rate-target 40`: initial post rate 40 Hz, α_a and α_c rescaled by k = r_CK / 40; on CartPole η rescaled by the update-RMS ratio). All arms share it. BindsNET's LIF baseline is a descriptive reference only.
- **Design record:** `docs/iterations/015-hybrid-columns.md`. Development results (seeds 0–5, `experiments/results/pilot015/`) set the frozen settings below; they are not part of the study.

## Design

- **Common flags:** `--engine nlif --rate-target 40 --block 20 --fast`; LunarLander adds `--eta-cal`. This is the baseline's step rule: preset η on CartPole, the calibrated local step on LunarLander (iteration 013).
- **Arms** (identical except for):

| Arm | Flags | What it is |
| --- | --- | --- |
| `base` | – | no recurrence (R = 0): the matched no-Adam baseline |
| `cols_lam0` | `--chi-rec 0.3 --rec cols --lam 0` | implicit: recurrent action columns, task term on R only |
| `cols_on` | `--chi-rec 0.3 --rec cols --lam 1` | learned: columns plus the map term (r = 1) |
| `cols_sealed` | `--chi-rec 0.3 --rec cols --rec-seal --lam 0` | enforced: the same draws of R without its cross-column synapses; the one-hop task credit on R is exactly 0 when every target is in the source's column, so R changes only by homeostasis |
| `lat_on` | `--chi-rec 0.3 --rec lat --lam 1` | columns, 2 latent modules (input→latent frozen), map term (r = 1) |

- **Frozen settings, chosen on stability and structure only, never on return:**
  - χ_rec = 0.3: largest module eigenvalue ≤ 0.27, loop time ≤ 28 ms, χG ≤ 0.25 (pilots B and C);
  - every arm of a seed shares s_in, the low-rate scaling and the pre-set rates η_rec and λ_rec (measured on the unsealed draws);
  - r = 1: the plan's pre-set default; λ = r × RMS(task step on R) / RMS(structural step), a pre-set calibration (`calibrate_rec`);
  - `lat_on` passed the latent go criteria set before pilot C (iteration 015 §6).
- **Episodes:** as the baseline. CartPole 400. LunarLander pooled over 16 copies, at least 3000 and at most 10000, stopping once solved (CK's criterion) after 3000.
- **Groups:** `<task>_<arm>`: 2 tasks × 5 arms = 10 groups, 20 seeds each, 200 runs.
- **Seeds:** 3000–3019, fresh: no earlier run used seeds at or above 3000.

## Pre-registered tests

**Primary metric:** LunarLander last-100 return; CartPole AUC (mean return over the 400 episodes; the baseline solves every run, so the solved count cannot separate arms).

**Contrasts** (per task): two-sided Mann–Whitney on the primary metric, Holm-adjusted within each family. Effect size: Hodges–Lehmann shift with a 95% CI.

| Family | Contrast | Asks |
| --- | --- | --- |
| **headline** (`primary`, 6 tests) | `cols_lam0` vs `base` | does implicit modularity keep or improve performance? |
| | `cols_on` vs `base` | does learned modularity (Map-STDP) keep or improve performance? |
| | `cols_sealed` vs `base` | does enforced modularity keep or improve performance? |
| secondary (`secondary`, 6 tests) | `cols_on` vs `cols_lam0` | what does the map term add over implicit structure? |
| | `cols_on` vs `cols_sealed` | learned vs enforced structure |
| | `lat_on` vs `cols_on` | what do latent modules add? |

For each headline contrast:
- **Improves:** Holm-adjusted p < 0.05 and a positive shift.
- **Keeps performance (non-inferiority):** the lower end of the 95% CI of the shift is above −Δ, with **Δ = 25**: 25 points of last-100 return on LunarLander and 25 points of AUC on CartPole (user, 2026-10-27).
- **Worse:** a significant negative shift.

Every contrast is reported, pass or fail.

**What the pilot suggests** (3 seeds, development, so stated before the study as expectations, not results; iteration 015 §6):
- On LunarLander, `cols_lam0` beat `base` on all three seeds (last-100 means 52 vs 26). Without the map term, the task term spreads the columns apart and builds a hub into the main-engine column.
- The map term seals the columns and removes that pathway; `cols_on` was level with `base` (24). `cols_sealed` was not piloted; the map-term arms were fully sealed by episode 200, so it is expected near `cols_on`.
- On CartPole, all arms were level (AUC 336–367); the task term seals the columns there on its own.

**Secondary (reported, not tested for the claims):** solved count, T_f, T_s; LunarLander AUC over the first 3000 episodes; the module weight table of R per snapshot (`rec.RK`) and the within-module share; stability monitors (max Re λ, τ_loop, χG), rates, action repeat vs chance; L_obs (uninformative at the pilot's 0.1–0.2% causal share, reported anyway); ms per frame. Descriptive: the BindsNET baselines in `reports/tdstdp-baseline`.

**Not done:** no settings change after launch, and no seeds are added or dropped. A crashed run is relaunched from its checkpoint. If a group still has fewer than 20 runs, it is reported with its n.

## Run conditions

- **Machine:** the local workstation (72 cores), with `OMP`, `MKL`, `OPENBLAS` and `NUMBA` threads set to 1 (`report.run`), 64 workers.
- **Code state:** a git worktree of a commit made for the study, so the main checkout can keep changing.
- **Projected time:** about 12–16 h. Pilot cost per 3000 LunarLander episodes was 0.4 h (base) and 0.6–0.9 h (recurrent arms); unsolved runs go to 10000 episodes, and later episodes are longer. CartPole runs take about 5–10 min.
- **Checkpoints:** every run checkpoints and resumes (`runs/<group>/s<seed>.ckpt.npz`, gitignored). A resumed run is not bit-identical; the count is in `resumes`.
- **One code state:** `manifest.json` records the git HEAD and the sha256 of the uncommitted code; the runner refuses to relaunch if it differs.

## Commands

```bash
python -m experiments.report run hybrid-columns --workers 64   # launch or resume every declared run; then summarize
python -m experiments.report summarize hybrid-columns          # summary.csv, pairs.csv and summary.md from runs/
```

For example (exact per-run commands are in `manifest.json`):

```bash
python -m experiments.tdstdp --task LunarLander --seed 3000 --engine nlif --rate-target 40 --block 20 --fast --eta-cal --chi-rec 0.3 --rec lat --lam 1 --out reports/hybrid-columns/runs/LunarLander_lat_on/s3000.json
python -m experiments.tdstdp --task CartPole --seed 3019 --engine nlif --rate-target 40 --block 20 --fast --out reports/hybrid-columns/runs/CartPole_base/s3019.json
```
