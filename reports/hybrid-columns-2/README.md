# Study `hybrid-columns-2`: Map-STDP columns in the TD-STDP backbone on Acrobot, without Adam

**Pre-registered 2026-10-28 15:25 EDT, before any run of this study** (approved by the user; code at commit `8d0a9e4`). The design and tests below are fixed.

Nothing in this directory is written by hand except this README. `manifest.json`, `code.patch`, `runs/`, `summary.*` and `pairs.csv` come only from `python -m experiments.report`.

## What the study is

- **Question (user, 2026-10-27):** across tasks, when does each regime of modularity help? This study adds a task whose actions are exclusive but must be **sequenced**: Acrobot's three torques, pumped in alternation.
- **What came before** (`reports/hybrid-columns`, iteration 015 §8):
  - On CartPole and LunarLander no regime improved the primary metric.
  - Sealed columns were non-inferior on both tasks. Task-formed and map-term columns were non-inferior on LunarLander and inconclusive on CartPole.
  - Every column arm learned LunarLander faster over its first 3000 episodes (AUC +9 to +11). Iteration 016 §10–11 later showed that most of this came from the column arms starting at a higher rate and a sharper policy (dev pilot, n = 6: +11/+13 unmatched, +3/+6 with both matchings below, not significant).
- **Backbone, engine and arms:** as `hybrid-columns`.
  - TD-STDP actor-critic of Chung & Kozma 2020 without Adam.
  - Noisy LIF (`--engine nlif`, dt 1 ms, τ_m 20 ms, τ_ref 2 ms) at an initial post rate of 40 Hz.
  - Each action's actor E population is one column, with recurrent E→E synapses R.
- **Taxi was planned and dropped.** It failed the pilot's floor gate: no deliveries in 6000 base episodes (iteration 016 §9).
- **Design record:** `docs/iterations/016-columns-more-tasks.md`. The pilots (dev seeds 0–15, `experiments/results/pilot016/`) set the frozen settings below. They are not part of the study.

## Design

- **Task settings:** Acrobot-v1, by the pre-set rules in `experiments/tdstdp.py` `PRESETS["Acrobot"]` (iteration 016 §4). There are no authors' configs for this task, and nothing was tuned on return.
  - **Populations:** CartPole's (`n_ce` 40, `n_ae` 20 per action, one copy). α_a 50, τ_d 1000 ms, freeze 20 ms.
  - **Encoder:** LunarLander's (per-dimension Fourier order 1, ON/OFF split, constant unit, `in_rep` 16): 208 inputs.
  - **Step:** `--eta-cal`, with LunarLander's Adam steps as the step to match.
  - **Low-rate scaling k:** from CartPole's BindsNET rate (140 Hz).
  - **Reward:** `r_scale` 0.02 (random-policy reward RMS matched to CartPole's).
  - **Critic floor:** β_c −1.2 (1.2 × the random-policy value).
  - **Exploration:** entropy bonus c_e 1e-4 (LunarLander's).
- **Common flags:** `--engine nlif --rate-target 40 --block 20 --fast --eta-cal --rate-match-rec --ent-match-rec`.
- **Matched starts (pre-set rules, iteration 016 §10–11; no-ops for base):**
  - `--rate-match-rec`: s_in is calibrated with the arm's R at init, so every arm starts at the same post rate (40 Hz).
  - `--ent-match-rec`: α_a is scaled per arm (f ≈ 0.7–0.9) so the arm's initial policy entropy equals the base run's. The within-column loop amplifies rate differences between columns, which otherwise acts as a larger softmax gain.
  - Without them, 3 of 30 column runs collapsed to a fixed torque in the pilots, and the column arms differed from base in starting gain, not only in structure.
- **Arms** (identical except for):

| Arm | Flags | What it is |
| --- | --- | --- |
| `base` | – | no recurrence (R = 0): the matched no-Adam baseline |
| `cols_lam0` | `--chi-rec 0.3 --rec cols --lam 0` | implicit: recurrent action columns, task term on R only |
| `cols_on` | `--chi-rec 0.3 --rec cols --lam 1` | learned: columns plus the map term (r = 1) |
| `cols_sealed` | `--chi-rec 0.3 --rec cols --rec-seal --lam 0` | enforced: the same draws of R without its cross-column synapses |

- **Frozen settings:** χ_rec 0.3 and r = 1, as in `hybrid-columns`. Every arm of a seed shares the low-rate scaling k and η_rec / λ_rec (measured on the unsealed draws); s_in and α_a are per arm, by the matching rules above.
- **Episodes:** 500, fixed, with no early stop.
- **Groups:** `Acrobot_<arm>`: 4 groups × 20 seeds = 80 runs.
- **Seeds:** 4000–4019, fresh. No earlier run used seeds at or above 4000.

## Pre-registered tests

**Primary metric:** AUC, the mean return over the 500 episodes.

**Contrasts:** two-sided Mann–Whitney on the primary metric, Holm-adjusted within each family. Effect size: Hodges–Lehmann shift with a 95% CI.

| Family | Contrast | Asks |
| --- | --- | --- |
| **headline** (`primary`, 3 tests) | `cols_lam0` vs `base` | does implicit modularity keep or improve performance? |
| | `cols_on` vs `base` | does learned modularity (Map-STDP) keep or improve performance? |
| | `cols_sealed` vs `base` | does enforced modularity keep or improve performance? |
| secondary (`secondary`, 3 tests) | `cols_lam0` vs `cols_sealed` | **P-seq:** do cross-column links help when actions must be sequenced? |
| | `cols_on` vs `cols_lam0` | what does the map term add over implicit structure? |
| | `cols_on` vs `cols_sealed` | learned vs enforced structure |

For each headline contrast:
- **Improves:** Holm-adjusted p < 0.05 and a positive shift.
- **Keeps performance (non-inferiority):** the lower end of the 95% CI of the shift is above −Δ, with **Δ = 25 points of AUC** (as in `hybrid-columns`; confirmed by the user).
- **Worse:** a significant negative shift.

Every contrast is reported, pass or fail.

**P-seq, the question this task was chosen for:**
- If `cols_lam0` beats `cols_sealed`, sequencing alone makes cross-column links useful.
- If they are level, cross-column links neither help nor hurt sequencing, and any cost of columns comes from the within-column loop itself.

**What the pilot suggests** (both matchings, dev seeds 0–9, development, so these are expectations, not results):
- Base: AUC median −163 (range −271 to −134), no collapse.
- Column arms: no collapse (0/30). AUC median −200 (`cols_lam0`), −216 (`cols_on`), −217 (`cols_sealed`); HL shifts vs base −32, −47, −51 (MW p 0.02–0.05 at n = 10).
- The headline contrasts may therefore come out "worse" rather than non-inferior. That would be a reportable result: a cost of columns on a sequencing task.

**Secondary (reported, not tested for the claims):**
- solved count (trailing-100 mean ≥ −100), T_s, last-100 return;
- collapsed runs (last-100 = −500);
- the matching record per run (`low_rate`: `r_got_hz`, `ent_scale`, `h_base`, `h_got`);
- the module weight table of R per snapshot (`rec.RK`) and the within-column share;
- stability monitors (max Re λ, τ_loop, χG), rates, action repeat vs chance;
- ms per frame.

**Not done:** no settings change after launch, and no seeds are added or dropped. A crashed run is relaunched from its checkpoint. If a group still has fewer than 20 runs, it is reported with its n.

## Run conditions

- **Machine:** the local workstation (72 cores), with `OMP`, `MKL`, `OPENBLAS` and `NUMBA` threads set to 1, 64 workers.
- **Code state:** git worktree `../map-stdp-hc2` (branch `hybrid-columns-2-study`) at commit `8d0a9e4`.
- **Projected time:** about 30 min. Pilot runs took 1.3 min (base) and 3–6 min (column arms, including about 1 min of entropy matching).
- **One code state:** `manifest.json` records the git HEAD and the sha256 of the uncommitted code; the runner refuses to relaunch if it differs.

## Commands

```bash
python -m experiments.report run hybrid-columns-2 --workers 64   # launch or resume every declared run; then summarize
python -m experiments.report summarize hybrid-columns-2          # summary.csv, pairs.csv and summary.md from runs/
```

For example:

```bash
python -m experiments.tdstdp --task Acrobot --seed 4000 --engine nlif --rate-target 40 --block 20 --fast --eta-cal --rate-match-rec --ent-match-rec --chi-rec 0.3 --rec cols --lam 1 --out reports/hybrid-columns-2/runs/Acrobot_cols_on/s4000.json
```
