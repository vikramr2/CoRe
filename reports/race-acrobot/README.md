# Study `race-acrobot`: the race readout with the decision hold on Acrobot against the published backbone

**Pre-registered 2026-10-29 19:35 EDT, before any run of this study** (approved by the user; code at the first commit after this stamp, recorded in `manifest.json`; worktree `../map-stdp-racr`, branch `race-acrobot-study`). The design and tests below are fixed.

Nothing in this directory is written by hand except this README. `manifest.json`, `code.patch`, `runs/`, `summary.*` and `pairs.csv` come only from `python -m experiments.report`.

## What the study is

- **Question (user, 2026-10-29):** on Acrobot, does the TD-STDP backbone with its actor read out by a race on action-column spike counts, each decision held for two environment steps, match or beat the backbone as published (Chung & Kozma's softmax), and do recurrent action columns help it?
- **Why these arms and settings:** they are the best Acrobot settings found in the development pilots of iteration 016 (§20 critic diagnosis and grids, §21 the decision hold; dev seeds 20–29, `experiments/results/pilot016/`). Stated plainly: the race's settings were **selected per task on dev seeds**, as the backbone's own authors set theirs per task; this study runs them on fresh seeds. Two settings apply to every arm, the reference included: the critic bias β_c = −1.5 (the 1.5 V_rand rule: a margin covering the TD(0) bias of the critic's own spikes, which otherwise makes the floor value unrepresentable and kills the critic at the floor, 016 §20) and `--eta-cal`. The dev-pilot numbers are development results and are not reported.
- **Design record:** `docs/iterations/016-columns-more-tasks.md` §20–21; the procedure in `docs/columns-recipe.md`.

## Design

- **Common flags, every arm:** `--engine nlif --rate-target 40 --block 20 --fast --rate-match-rec --eta-cal --beta-c -1.5`.
- **Arms:**

| Arm | Flags | What it is |
| --- | --- | --- |
| `base_ck` | `--ent-match-rec` | CK's softmax on the unnormalised scores at the matched start and β_c −1.5: the published backbone with the critic-bias rule (the reference) |
| `base_race_h2` | `--readout race --rate-cost 5e-4 --rate-floor 0.5 --race-theta 16 --freeze 40` | the race, no recurrence: columns' E spike counts race to θ = 16 (the 20-ms rule, n_ae r₀ · 20 ms) and the winner is held for 2 env steps (the 40-ms window; n_hold = the smallest hold at which a uniform random policy terminates ≥ 20 % of episodes); credit = the population surrogate softmax(β log r), β by the contrast probe at init, no re-probe; the band: CK's target-rate term on the copy mean (c_t 5e-4, ρ̂ = r₀) plus the one-sided per-column floor at r₀/2 |
| `cols_lam0_race_h2` | `… --chi-rec 0.3 --rec cols --lam 0` | the race on implicit columns (recurrent E→E within and across columns, task term only) |
| `cols_on_race_h2` | `… --chi-rec 0.3 --rec cols --lam 1` | the race on learned columns (plus the map term, r = 1) |
| `cols_sealed_race_h2` | `… --chi-rec 0.3 --rec cols --rec-seal --lam 0` | the race on sealed columns (no cross-column synapses) |

- **Settings selected on dev seeds (016 §20–21):** the hold (n_hold 2, over 1 / 3 / 4 / 8 and over longer windows at the θ rule, more neurons, less noise, stronger loops, all of which collapsed or degraded); the band (the rate term's coefficient from the §18 sizing, the floor at half the target); θ 16 from the 20-ms rule; β_c −1.5 from the closed-form bias margin; no β re-probe (it hurts under the hold). The matched start and the loop gain are the pre-set rules of 016 §10 and 015.
- **Episodes:** 500 (Acrobot's preset). **Seeds:** 7100–7119, fresh (no earlier run used them). 5 groups × 20 seeds = 100 runs.

## Pre-registered tests

**Primary metric:** AUC (the mean return over the 500 episodes). Also reported: last-100 return and the solved count (trailing-100 mean ≥ −100).

**Contrasts:** two-sided Mann–Whitney on AUC, Holm within each family; Hodges–Lehmann shift with a 95 % CI.

| Family | Contrast | Asks |
| --- | --- | --- |
| **headline** (`primary`, 4 tests) | `base_race_h2` vs `base_ck` | does the race with the hold match or beat the published backbone? |
| | `cols_lam0_race_h2` vs `base_ck` | with implicit columns |
| | `cols_on_race_h2` vs `base_ck` | with learned columns |
| | `cols_sealed_race_h2` vs `base_ck` | with sealed columns |
| secondary (`secondary`, 4 tests) | `cols_sealed_race_h2` vs `base_race_h2`; `cols_on_race_h2` vs `base_race_h2`; `cols_sealed_race_h2` vs `cols_lam0_race_h2`; `cols_on_race_h2` vs `cols_lam0_race_h2` | do columns help the race, and does the regime matter? |

For each headline contrast: **improves** = Holm-adjusted p < 0.05 and a positive shift; **keeps performance** = the lower end of the 95 % CI above −Δ with Δ = 25 AUC; **worse** = a significant negative shift.

**Pre-registered expectations** (from the hold pilot, n = 10; expectations, not results): sealed and learned columns level with `base_ck` (shifts within ± 10, non-inferior, with better last-100 and a few solved runs); implicit columns and the race without columns worse by 35–45; the regime ordering sealed ≈ learned > implicit > none, with sealed vs none and sealed vs implicit significant; no run at the floor; 0 % deadline decisions; actor spikes per decision about 14 % below `base_ck`'s.

**Secondary (reported, not tested for the claims):** T_s, T_f; `actor_spikes_per_decision`, `actor_spikes_to_decision`; `race_log` per log bin (realised race entropy, `action_repeat_probability`, column-rate minimum); bound / deadline / tie counts; actor and critic rates (the critic must stay alive: its rate per bin); the column weight tables `rec.RK` and the within-column share; stability monitors (max Re λ ≤ 0.5, τ_loop ≤ 40 ms); collapsed runs (last-100 = −500); ms per frame.

**Not done:** no settings change after launch; no seeds added or dropped. A crashed run is relaunched from its checkpoint; a group with fewer than 20 runs is reported with its n.

## Run conditions

- **Machine:** the local workstation (72 cores), threads set to 1, 50 workers (the LunarLander hold pilot holds 9 cores). **Projected time:** about 10 min (5–7 min per run).
- **Code state:** git worktree `../map-stdp-racr` (branch `race-acrobot-study`) at the commit recorded in `manifest.json`; the runner refuses to relaunch if the code differs.

## Commands

```bash
python -m experiments.report run race-acrobot --workers 50    # launch or resume every declared run; then summarize
python -m experiments.report summarize race-acrobot           # summary.csv, pairs.csv and summary.md from runs/
```

For example:

```bash
python -m experiments.tdstdp --task Acrobot --seed 7100 --engine nlif --rate-target 40 --block 20 --fast --rate-match-rec --eta-cal --beta-c -1.5 --readout race --rate-cost 5e-4 --rate-floor 0.5 --race-theta 16 --freeze 40 --chi-rec 0.3 --rec cols --rec-seal --lam 0 --out reports/race-acrobot/runs/Acrobot_cols_sealed_race_h2/s7100.json
```
