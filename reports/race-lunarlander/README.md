# Study `race-lunarlander`: the race readout at the backbone's control rate on LunarLander against the published backbone

**Pre-registered 2026-10-30 02:45 EDT, before any run of this study** (approved by the user; code at commit `bcdd452`, the stamp committed on branch `race-lunarlander-study`, worktree `../map-stdp-racl`). The design and tests below are fixed.

Nothing in this directory is written by hand except this README. `manifest.json`, `code.patch`, `runs/`, `summary.*` and `pairs.csv` come only from `python -m experiments.report`.

## What the study is

- **Question (user, 2026-10-29):** on LunarLander, does the TD-STDP backbone with its actor read out by a race on action-column spike counts, deciding within 20 ms and holding each decision at the backbone's own 40-ms control rate, match or beat the backbone as published (Chung & Kozma's softmax), and do recurrent action columns help it?
- **Why these arms and settings:** the development pilots of iteration 016 (§20–21; dev seeds 20–29, `experiments/results/pilot016/`). Stated plainly: the race's settings were **selected per task on dev seeds**, as the backbone's own authors set theirs per task; this study runs them on fresh seeds. Two findings shape them: the race's learning surrogate must read the quantity the race tallies (the excitatory column rate; reading CK's net rate left the credit dead on this task, whose actor has inhibitory posts), and the race must run at the backbone's control rate (its preset holds each action for 40 ms). The dev-pilot numbers are development results and are not reported.
- **Design record:** `docs/iterations/016-columns-more-tasks.md` §16–21; the procedure in `docs/columns-recipe.md`.

## Design

- **Common flags, every arm:** `--engine nlif --rate-target 40 --block 20 --fast --rate-match-rec --eta-cal` (the preset's Adam steps as the calibrated step; under the race the actor's calibration RMS is taken over the credited E rows).
- **Arms** (the preset's 16 copies, 40-ms freeze, Fourier encoder with the constant unit, entropy bonus c_e 1e-4, β_c −2.0 in every arm):

| Arm | Flags | What it is |
| --- | --- | --- |
| `base_ck` | `--ent-match-rec` | CK's softmax on the unnormalised scores at the matched start: the published backbone (the reference; `hybrid-columns-3-ck`'s arm on this study's seeds) |
| `base_race_e` | `--readout race --rate-cost 5e-4 --rate-floor 0.5 --race-theta 26` | the race, no recurrence: columns' E spike counts race to θ = 26 (the 20-ms rule, n_ae r₀ · 20 ms, n_ae 32) within the first 20 ms, the leader at the 40-ms deadline otherwise; the winner is held for the preset's 40-ms window (2 env steps, the backbone's control rate); credit = the population surrogate softmax(β log r_E) on the **E column rate**, β by the contrast probe at init under the 0.05 fit gate, no re-probe; the band: CK's target-rate term on the copy mean (c_t 5e-4, ρ̂ = r₀) plus the one-sided per-column floor at r₀/2 |
| `cols_lam0_race_e` | `… --chi-rec 0.3 --rec cols --lam 0` | the race on implicit columns (recurrent E→E within and across columns, task term only) |
| `cols_on_race_e` | `… --chi-rec 0.3 --rec cols --lam 1` | the race on learned columns (plus the map term, r = 1) |
| `cols_sealed_race_e` | `… --chi-rec 0.3 --rec cols --rec-seal --lam 0` | the race on sealed columns (no cross-column synapses) |

- **Settings selected on dev seeds (016 §21):** θ from the 20-ms rule (26; the 40-ms rule's 51 put half the decisions at the deadline); the decision at 20 ms with the hold at the preset's control rate (the race at 20 ms without the hold, and the hold with the old surrogate, each failed alone); the band's coefficients from §18 and §20; no re-probe. The matched start and the loop gain are the pre-set rules of 016 §10 and 015. The pilot ran base, sealed and learned columns; implicit columns are included for the regime contrast as in the other two race studies.
- **Episodes:** the preset's: pooled over 16 copies, at least 3000 and at most 10000, stopping once solved (CK's criterion, trailing-100 mean ≥ 200) after 3000. **Seeds:** 7200–7219, fresh. 5 groups × 20 seeds = 100 runs.

## Pre-registered tests

**Primary metric:** last-100 return (as in every LunarLander study of this series). Also reported: AUC over the first 3000 episodes, the solved count, T_s.

**Contrasts:** two-sided Mann–Whitney on the primary metric, Holm within each family; Hodges–Lehmann shift with a 95 % CI.

| Family | Contrast | Asks |
| --- | --- | --- |
| **headline** (`primary`, 4 tests) | `base_race_e` vs `base_ck` | does the race match or beat the published backbone? |
| | `cols_lam0_race_e` vs `base_ck` | with implicit columns |
| | `cols_on_race_e` vs `base_ck` | with learned columns |
| | `cols_sealed_race_e` vs `base_ck` | with sealed columns |
| secondary (`secondary`, 4 tests) | `cols_sealed_race_e` vs `base_race_e`; `cols_on_race_e` vs `base_race_e`; `cols_sealed_race_e` vs `cols_lam0_race_e`; `cols_on_race_e` vs `cols_lam0_race_e` | do columns help the race, and does the regime matter? |

For each headline contrast: **improves** = Holm-adjusted p < 0.05 and a positive shift; **keeps performance** = the lower end of the 95 % CI above −Δ with Δ = 25 last-100 points; **worse** = a significant negative shift.

**Pre-registered expectations** (from the full-length dev pilot, 10 seeds per arm against the backbone's 20 of `hybrid-columns-3-ck`, 016 §21; expectations, not results): every race arm non-inferior to `base_ck` on last-100 with no collapsed run; sealed columns above `base_ck` by about 30 last-100 points and learned columns by about 17, both with CIs clear of zero (pilot: +31.2 [+13.9, +48.6], +16.9 [+7.4, +26.9]); the race without columns level (pilot +6.4 [−2.8, +15.8]); implicit columns untested in the pilot and not predicted; early learning level (AUC over the first 3000 episodes within ± 10); sealed and learned columns above the race without columns (pilot p 0.01, 0.03); actor spikes to decision about 30 % of the backbone's per window at a held rate near 38 Hz where the backbone's rises to about 58 Hz; no run solved by CK's criterion in either readout.

**Secondary (reported, not tested for the claims):** AUC over the first 3000 episodes; `actor_spikes_per_decision`, `actor_spikes_to_decision`; `race_log` per log bin (realised race entropy, `action_repeat_probability`, column-rate minimum); bound / deadline / tie counts; the init probe's β and fit; actor and critic rates; the column weight tables `rec.RK` and the within-column share; stability monitors (max Re λ ≤ 0.5, τ_loop ≤ 40 ms); ms per frame.

**Not done:** no settings change after launch; no seeds added or dropped. A crashed run is relaunched from its checkpoint; a group with fewer than 20 runs is reported with its n.

## Run conditions

- **Machine:** the local workstation (72 cores), threads set to 1, 60 workers. **Projected time:** 10–14 h (each run 6–8 h; unsolved runs go to 10000 episodes).
- **Code state:** git worktree `../map-stdp-racl` (branch `race-lunarlander-study`) at the commit recorded in `manifest.json`; the runner refuses to relaunch if the code differs.

## Commands

```bash
python -m experiments.report run race-lunarlander --workers 60   # launch or resume every declared run; then summarize
python -m experiments.report summarize race-lunarlander          # summary.csv, pairs.csv and summary.md from runs/
```

For example:

```bash
python -m experiments.tdstdp --task LunarLander --seed 7200 --engine nlif --rate-target 40 --block 20 --fast --rate-match-rec --eta-cal --readout race --rate-cost 5e-4 --rate-floor 0.5 --race-theta 26 --chi-rec 0.3 --rec cols --rec-seal --lam 0 --out reports/race-lunarlander/runs/LunarLander_cols_sealed_race_e/s7200.json
```
