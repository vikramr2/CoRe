# Study `race-cartpole`: the race readout on CartPole against the published backbone

**Pre-registered 2026-10-29 18:02 EDT, before any run of this study** (approved by the user; code at commit `23cc0b9`, the stamp committed on branch `race-cartpole-study`, worktree `../map-stdp-rcp`). The design and tests below are fixed.

Nothing in this directory is written by hand except this README. `manifest.json`, `code.patch`, `runs/`, `summary.*` and `pairs.csv` come only from `python -m experiments.report`.

## What the study is

- **Question (user, 2026-10-29):** on CartPole, does the TD-STDP backbone with its actor read out by a race on action-column spike counts, learning through the population surrogate with its gain re-estimated on a schedule, match or beat the backbone as published (Chung & Kozma's softmax), and do recurrent action columns help it?
- **Why these arms:** they are the best CartPole settings found in the development pilots of iteration 016 (§18 re-pilot, §20 grids; dev seeds 20–29, `experiments/results/pilot016/`). This is stated plainly: the race's settings were **selected per task on dev seeds**, as the backbone's own authors set theirs per task; this study runs them on fresh seeds. The dev-pilot numbers are development results and are not reported.
- **Design record:** `docs/iterations/016-columns-more-tasks.md` §16, §18, §20; the procedure in `docs/columns-recipe.md`.

## Design

- **Common flags, every arm:** `--engine nlif --rate-target 40 --block 20 --fast --rate-match-rec`; CK's preset η (CartPole's preset η are SGD rates; no `--eta-cal`).
- **Arms:**

| Arm | Flags | What it is |
| --- | --- | --- |
| `base_ck` | `--ent-match-rec` | CK's softmax on the unnormalised scores: the published backbone at the matched start (the reference; `hybrid-columns-3-ck`'s arm) |
| `base_nl` | `--ent-match-rec --norm-logits` | the normalised softmax (`hybrid-columns-3`'s base) |
| `base_race_rp` | `--readout race --race-reprobe 50` | the race readout, no recurrence: columns' E spike counts race to θ = 16 (the rule: n_ae r₀ T), the leader at the 20-ms deadline otherwise; credit = the population surrogate softmax(β log r) with β by the contrast probe at init and **re-probed every 50 episodes** |
| `cols_lam0_race_rp` | `… --chi-rec 0.3 --rec cols --lam 0` | the race on implicit columns (recurrent E→E within and across columns, task term only) |
| `cols_on_race_rp` | `… --chi-rec 0.3 --rec cols --lam 1` | the race on learned columns (plus the map term, r = 1) |
| `cols_sealed_race_rp` | `… --chi-rec 0.3 --rec cols --rec-seal --lam 0` | the race on sealed columns (no cross-column synapses) |

- **Settings selected on dev seeds (016 §18, §20):** the β re-probe period (50 episodes; one re-probe per log bin) over the alternatives tried on CartPole (the rate term at c_t 5e-4, both, no fix, the per-neuron score credit at three step multipliers); θ and the matched start are the pre-set rules of §16 and §10. Known and accepted: under the re-probe the actor rate falls after solving (to about 6 Hz), as under the normalised softmax; the race arms' energy per decision is reported as a secondary.
- **Episodes:** 400 (CartPole's preset). **Seeds:** 7000–7019, fresh (no earlier run used seeds at or above 7000). 6 groups × 20 seeds = 120 runs.

## Pre-registered tests

**Primary metric:** AUC (the mean return over the 400 episodes). Also reported: last-100 return and the solved count (CK's criterion), which the user named as the quantities of interest.

**Contrasts:** two-sided Mann–Whitney on AUC, Holm within each family; Hodges–Lehmann shift with a 95 % CI.

| Family | Contrast | Asks |
| --- | --- | --- |
| **headline** (`primary`, 4 tests) | `base_race_rp` vs `base_ck` | does the race readout match or beat the published backbone? |
| | `cols_lam0_race_rp` vs `base_ck` | with implicit columns |
| | `cols_on_race_rp` vs `base_ck` | with learned columns |
| | `cols_sealed_race_rp` vs `base_ck` | with sealed columns |
| readout (`readout`, 1 test) | `base_nl` vs `base_ck` | the normalised softmax against the published one |
| secondary (`secondary`, 4 tests) | `cols_sealed_race_rp` vs `base_race_rp`; `cols_lam0_race_rp` vs `base_race_rp`; `cols_sealed_race_rp` vs `cols_lam0_race_rp`; `cols_on_race_rp` vs `cols_lam0_race_rp` | do columns help the race, and does the regime matter? |

For each headline contrast: **improves** = Holm-adjusted p < 0.05 and a positive shift; **keeps performance** = the lower end of the 95 % CI above −Δ with Δ = 25 AUC; **worse** = a significant negative shift.

**Pre-registered expectations** (from the dev pilots, n = 10; expectations, not results): every race arm solves 20/20 and is above `base_ck` on AUC, the sealed and learned column arms by 30–40 points with CIs clear of zero, the base race by 10–20; the ordering sealed ≥ learned ≥ implicit ≥ no columns; `base_nl` non-inferior to `base_ck`.

**Secondary (reported, not tested for the claims):** T_s and T_f; `actor_spikes_per_decision` and `actor_spikes_to_decision` (the energy claim: the race decides with fewer actor spikes); the β sequence per run (`race_cal_history`) and the realised race entropy per log bin (`race_log`); bound / deadline / tie counts; the actor rate per log bin; the column weight tables `rec.RK` and the within-column share; stability monitors; ms per frame.

**Not done:** no settings change after launch; no seeds added or dropped. A crashed run is relaunched from its checkpoint; a group with fewer than 20 runs is reported with its n.

## Run conditions

- **Machine:** the local workstation (72 cores), threads set to 1, 60 workers. **Projected time:** about 15 min (3–5 min per run).
- **Code state:** git worktree `../map-stdp-rcp` (branch `race-cartpole-study`) at the commit recorded in `manifest.json`; the runner refuses to relaunch if the code differs.

## Commands

```bash
python -m experiments.report run race-cartpole --workers 60   # launch or resume every declared run; then summarize
python -m experiments.report summarize race-cartpole          # summary.csv, pairs.csv and summary.md from runs/
```

For example:

```bash
python -m experiments.tdstdp --task CartPole --seed 7000 --engine nlif --rate-target 40 --block 20 --fast --rate-match-rec --readout race --race-reprobe 50 --chi-rec 0.3 --rec cols --rec-seal --lam 0 --out reports/race-cartpole/runs/CartPole_cols_sealed_race_rp/s7000.json
```
