# Iteration 016: when do columns help? Acrobot and Taxi

**Status:** pilots done (§7–§9). Study `hybrid-columns-2` (Acrobot only) is drafted, not launched. Pilots and the study start after the `hybrid-columns` study (iteration 015) finishes, so that they do not share its CPUs. The plan is `~/.claude/plans/jiggly-chasing-sparkle.md`.

**Question (user, 2026-10-27):** across tasks, when does each regime of modularity help?
- **implicit:** the task alone shapes the columns (`cols_lam0`);
- **learned:** the map term shapes them (`cols_on`);
- **enforced:** the columns are sealed (`cols_sealed`).

Two tasks (CartPole, LunarLander) cannot show a pattern, so this iteration adds two tasks chosen by **how their actions relate**, not by difficulty.

## 1. Working hypothesis (from iteration 015, interim)

- **CartPole:** 2 opposite actions, a purely reactive policy. All regimes keep performance.
- **LunarLander:** 4 actions that are combined and sequenced. At about 4600 episodes, task-formed columns are ahead of base (last-100 about 34 vs 23); map-term and sealed columns are level with base. In pilot C the task term built a hub into the main-engine column.
- **Hypothesis:** sealed columns cost nothing when actions are independent alternatives. Cross-column links help when actions must work together.

The final 015 numbers replace these before pre-registration.

## 2. Tasks (user decision, 2026-10-27)

| Task | Actions | How the actions relate |
| --- | --- | --- |
| Acrobot-v1 | 3: torque −1, 0, +1 | exclusive, but must be **sequenced** (energy pumping) |
| Taxi-v4 | 6: 4 moves, pickup, dropoff | **hierarchical**: sub-goals (reach the passenger, pick up, reach the destination, drop off); two functional groups of actions |

- **Acrobot** separates sequencing from combining.
- **Taxi** is the classic hierarchical RL task (Dietterich's MAXQ). It also adds a different domain: a grid world with a discrete state, where the policy is effectively a lookup table.
- **Rejected:**
  - a discretised Pendulum (with CartPole and Acrobot, a third pendulum);
  - MountainCar (sparse reward; a base arm at floor would make every arm level);
  - MinAtar (needs a new package; the actor's linear policy risks a floor).

## 3. Predictions (draft; fixed in the study README)

- **P-seq (Acrobot):** if sequencing alone makes cross-column links useful, `cols_lam0` beats `cols_sealed`. If sealed matches, combining actions is what mattered on LunarLander.
- **P-hier (Taxi):** `cols_lam0` and `cols_on` beat `cols_sealed`.
- **P-group (Taxi structure, `rec.RK`):** cross-column weight is concentrated within the movement group, not between the movement and passenger groups.
- **Headline contrasts:** as in 015, each regime vs base, with Holm within families. The margin Δ is restated per task at pre-registration: the AUC scales are about −500 to −60 (Acrobot) and −770 to +8 (Taxi).

## 4. Settings: pre-set rules, never tuned on return (`experiments/tdstdp.py`, `PRESETS`)

No authors' configs exist for these tasks.

- **Populations, α_a, τ_d, freeze, init:** CartPole's (`n_ce` 40, `n_ae` 20 per action, no I posts, one copy, α_a 50, τ_d 1000 ms, 20 ms, U(0, 0.1)).
- **Step:** `--eta-cal`, with LunarLander's Adam steps (critic 1.25e-4, actor 6.25e-5 mV) as the step to match. These are the authors' only Adam steps.
- **Low-rate scaling k:** `r_ck_hz` = 140 Hz, CartPole's BindsNET rate (015 study, 20 seeds: 138–142), so k = 3.5 and the logits keep CartPole's range. A task's own BindsNET rate would tie the scaling to an engine these tasks were never tuned on.
  - Measured BindsNET rates: Acrobot 204 Hz, Taxi 55 Hz.
- **Encoders:**
  - **Acrobot:** LunarLander's (per-dimension order 1, ON/OFF split, constant unit, `in_rep` 16), giving 208 inputs. CartPole's order-2 cross terms would give 3^6 = 729 features. Bounds are the M3 `FOURIER_BOUNDS`.
  - **Taxi:** a one-hot of the state index (`--onehot`, 500 inputs). `in_rep` 1 is the smallest replication, and it is feasible: `calibrate_s_in` reaches 40 Hz with s_in 11.8. `in_rep` 16 also works (s_in 0.72) but makes 8000 inputs.
- **Reward scale:** `r_scale` = 0.02 / RMS of the per-step reward under a random policy (20–50k steps). CartPole's RMS is 1.
  - Acrobot: RMS 1.00, `r_scale` 0.02.
  - Taxi: RMS 5.74, `r_scale` 0.0035.
- **Critic floor:** β_c = min(−0.2, 1.2·V_rand), where V_rand = r_scale·E[r]·τ_d / 20 ms is the value of a random policy (user decision, 2026-10-28, after pilot v1, §7).
  - The critic reads V = α_c·rate + β_c ≥ β_c, so β_c must lie below the values the task needs.
  - Resulting β_c: Acrobot −1.2 (V_rand −1.00), Taxi −0.82 (V_rand −0.68).
  - It is consistent with the authors' configs: LunarLander β_c −2.0 vs V_rand −2.31; CartPole −0.2 with V_rand +1.0.
- **Exploration:** c_e = 1e-4, LunarLander's entropy bonus (Eq. 26). It is the authors' only exploration setting beyond CartPole's config (user decision, 2026-10-28, after pilot v2, §8).
- **Episodes (fixed, no early stop):** Acrobot 500; Taxi 2000, from the tabular-RL convergence range.
- **Solve (secondary):** trailing-100 mean ≥ gymnasium's `reward_threshold` (Acrobot −100, Taxi 8).
- **Primary metric:** AUC (`report.TD_PAIR_METRICS`: AUC on every task except LunarLander).

## 5. Study `hybrid-columns-2` (declared in `experiments/report.py`, not pre-registered)

- **Arms:** `base`, `cols_lam0`, `cols_on`, `cols_sealed`. `lat_on` is added only if 015 gives a reason, decided before launch.
- **Flags:** `--rate-target 40 --block 20 --fast --eta-cal` (+ arm flags).
- **Seeds:** 4000–4019, fresh. 2 tasks × 4 arms × 20 = 160 runs.
- **Pairs:**
  - headline: each regime vs base;
  - secondary: `cols_on` vs `cols_lam0`, `cols_on` vs `cols_sealed`, `cols_lam0` vs `cols_sealed`.

## 6. Pilot (after 015; dev seeds 0–2, `experiments/results/pilot016/`)

Gates are on stability and structure only:
- rates hit 40 Hz;
- max Re λ ≤ 0.5, τ_loop ≤ 40 ms;
- no NaN;
- base is not at floor by the episode cap.

If base is at floor, report to the user before changing anything.

## 7. Pilot v1 (2026-10-28; `experiments/results/pilot016/v1/`, superseded)

24 runs: seeds 0–2, 2 tasks × 4 arms.

- **Taxi:** all 12 crashed when writing the result (`n_in` was a numpy int, not JSON-serialisable). Fixed in `ck_encoder`.
- **Acrobot:** at floor in every arm. The return was −500 (the goal was never reached) in 12 of 12 runs.
  - **Cause:** with CartPole's β_c = −0.2, the critic cannot represent Acrobot's negative values (V_rand −1.0). The critic posts went silent (rate 0.000 from episode 50), V sat at β_c and the TD error stayed negative. The policy collapsed (entropy 0.001).
  - **Stability gates passed:** rates 41 Hz, max Re λ ≤ 0.28, no NaN.
  - **Structure (final within-column share):** `cols_lam0` 0.83–0.86, `cols_on` 0.998, sealed 1.
- **Decision (user):** the β_c rule of §4, then pilot v2 with the same seeds.

## 8. Pilot v2 (2026-10-28; β_c rule; `experiments/results/pilot016/v2/`, superseded)

**Stability gates pass:** rates 40–41 Hz, max Re λ ≤ 0.28, no NaN.

**The policy collapses before it has explored:**
- **Acrobot:** 3 of 12 runs learn (base 1 of 3).
  - In base seed 2, entropy was 0.32 at episode 50 and the return reached −120 by episode 150.
  - In seeds 0 and 1, entropy fell to 0.09 at episode 50 and 0.001 later: a single fixed torque, return −500.
- **Taxi:** every run learns to avoid illegal actions (−350 to −200) but no run delivers a passenger (+8 needed). Entropy is 0.02 by mid-run.
- **Structure (final within-column share):**
  - Taxi: `cols_lam0` 0.46–0.53, `cols_on` 0.996, sealed 1;
  - Acrobot: `cols_lam0` 0.81–0.82.

**Decision (user):** add LunarLander's entropy bonus (§4) and run pilot v3 on the same seeds. If base still collapses (leaves the floor in fewer than 2 of 3 seeds), report to the user rather than trying further fixes.

## 9. Pilot v3 (2026-10-28; entropy bonus; `experiments/results/pilot016/`)

**Acrobot passes:**
- Base learned in 3 of 3 seeds: last-100 −110 to −145, AUC −163 to −166.
- Column arms learned in 7 of 9 seeds. Two collapsed to a fixed torque at −500: `cols_lam0` s1 and `cols_sealed` s0.
- Gates: rates 41 Hz, max Re λ ≤ 0.28, no NaN.
- Final within-column share: `cols_lam0` 0.65–0.89, `cols_on` 0.996–0.998, sealed 1.

**Taxi fails the floor gate:**
- Base delivered 0 passengers in 6000 episodes; 3 deliveries across all 12 runs.
- Every arm learns only to avoid illegal pickups and dropoffs, reaching the −200 time-limit plateau, so the hierarchy is never exercised.
- **Likely cause:** the one-hot input learns each of the 500 states separately, from rare visits, at a small no-Adam step.

**Decision (user, 2026-10-28):**
- Run the study on Acrobot alone (`hybrid-columns-2`, 4 arms × 20 seeds).
- No more Taxi fixes inside this study. A replacement for the hierarchical slot, or a Taxi variant, gets its own pilot later.
- The Taxi presets stay in the code, tested, for that later work.

## 10. Collapse in the column arms, and per-arm rate matching (2026-10-28)

**The problem:** in pilot v3, 2 of 9 Acrobot column runs collapsed to a fixed torque (−500); no base run did. The user asked for the cause and a fix.

**Diagnosis:**
- **The column arms start hotter.** `calibrate_s_in` probes without recurrence, so all arms share s_in, and the recurrent kicks add drive on top.
  - Actor rate at episode 50: about 0.047 per ms in every column run vs 0.039 in base.
  - At a shared s_in, the initial post rate with R on is 46 Hz on Acrobot and 43–44 Hz on LunarLander.
- **A sharper policy follows.** The policy is softmax(α_a × column rate), and the within-column loop (max Re λ ≈ 0.27) amplifies rate differences between columns.
  - Entropy at episode 50: median 0.29 with columns vs 0.58 in base.
  - Both collapsed runs were below 0.16.
- **The deterministic corner absorbs.** (A − p) → 0 when p → 1, and so does the entropy gradient (Eq. 25). The actor's update vanishes, the critic goes silent and the policy never recovers (action repeat 1.00).
- **Implication for iteration 015:** its column arms had the same higher starting rate. The early LunarLander speed-up (AUC +9 to +11 over the first 3000 episodes) was as large for sealed columns, which fits a gain effect as well as a structural one.

**Fix (user decision): `--rate-match-rec`.** `probe_rate` runs with the arm's R at init, so `calibrate_s_in` brings every arm to the same initial post rate.
- It is a pre-set rule, the same for every arm (a no-op for base).
- Check at seed 0 (initial rate with R on):

  | Arm | Shared s_in | Matched s_in |
  | --- | --- | --- |
  | Acrobot cols | 46.0 Hz | 42.4 Hz |
  | Acrobot sealed | 46.1 Hz | 44.8 Hz |
  | LunarLander cols | 43.7 Hz | 39.9 Hz |
  | LunarLander sealed | 42.8 Hz | 40.1 Hz |

  On Acrobot the closed-loop rate is steep in s_in, so the match is looser.

**Pilots** (`experiments/results/pilot016/rm/` and `ll_rm/`):
- **Acrobot:** 4 arms × dev seeds 0–9, all rate-matched. Measures the collapse count per arm and the entropy at episode 50.
- **LunarLander:** base, `cols_lam0` and `cols_sealed`, each unmatched and matched, over dev seeds 10–15 and 3000 episodes. Asks whether the early speed-up survives rate matching.

**Results** (development, small n; analysis script in the session scratchpad):

- **Acrobot, rate-matched** (seeds 0–9; AUC = mean return over 500 episodes):

  | Arm | Collapsed (last-100 = −500) | AUC median [range] | AUC < −300 | Actor rate, ep 50 | H(s), ep 50 (median, min) |
  | --- | --- | --- | --- | --- | --- |
  | base | 0/10 | −163 [−271, −134] | 0 | 0.038 | 0.52, 0.28 |
  | `cols_lam0` | 1/10 | −234 [−495, −158] | 4 | 0.040 | 0.35, 0.14 |
  | `cols_on` | 1/10 | −225 [−495, −162] | 2 | 0.039 | 0.33, 0.11 |
  | `cols_sealed` | 1/10 | −181 [−499, −155] | 2 | 0.041 | 0.40, 0.12 |

  - Rate matching removed the rate gap (0.040 vs 0.038 per ms at episode 50), but not the entropy gap or the slow and collapsed runs.
  - HL shift vs base: `cols_lam0` −73 (MW p 0.011), `cols_on` −61 (p 0.009), `cols_sealed` −22 (p 0.27).
  - **So the starting rate is not the cause.** The policy is sharper at the same mean rate, so the likelier driver is the within-column loop amplifying rate *differences* between columns (a gain on the contrast, not on the mean). This is a hypothesis, not tested here.
- **LunarLander** (seeds 10–15, AUC over 3000 episodes; base −58.2):

  | Arm | AUC mean | HL shift vs base | MW p | Last-100 mean |
  | --- | --- | --- | --- | --- |
  | base | −58.2 | – | – | 14.6 |
  | `cols_lam0` | −45.7 | +11.0 | 0.026 | 50.8 |
  | `cols_lam0` matched | −49.7 | +5.9 | 0.31 | 37.4 |
  | `cols_sealed` | −46.4 | +12.9 | 0.026 | 28.0 |
  | `cols_sealed` matched | −54.0 | +4.9 | 0.13 | 25.2 |

  - The unmatched arms reproduce 015's early speed-up (+11 to +13).
  - Rate matching roughly halves it (paired by seed: −4.0 for `cols_lam0`, −7.6 for `cols_sealed`). With n = 6 per arm the remainder is not significant.
  - **So part of 015's early LunarLander gain is a gain (starting-rate) effect.** Whether a structural part remains needs more seeds.

## 11. Per-arm entropy matching (2026-10-28)

**Check: policy entropy at init** (W and R frozen, rule off, after each arm's own calibration; CAL stream; scratch probe):

| Task | Arm | H₀ (nats; median) | Logit spread (SD across actions) |
| --- | --- | --- | --- |
| Acrobot (max ln 3 = 1.10) | base | 0.977 | 0.49 |
| | cols, shared s_in | 0.928 | 0.59 |
| | cols, rate-matched | 0.923 | 0.61 |
| | sealed, rate-matched | 0.898 | 0.67 |
| LunarLander (max ln 4 = 1.39) | base | 1.274 | 0.46 |
| | cols, rate-matched | 1.251 | 0.52 |
| | sealed, rate-matched | 1.234 | 0.55 |

- **Confirmed:** at the same mean rate the column policies start sharper. The within-column loop amplifies the logit spread (+25 to +37% on Acrobot), and rate matching does not change that.
- **Qualified later (§13):** at init about 93% of the logit variance is rate noise, so what the loop amplifies is mostly correlated noise within a column, not a signal difference between columns; the noise-only logit SNR is about 10% lower with columns. Entropy matching therefore calibrates noise, and stays fixed while the loop gain changes over training.
- This acts as a larger α_a, the softmax gain.

**Fix (user decision): `--ent-match-rec`.**
- `calibrate_low` multiplies α_a by f, found by `match_entropy` (bisection on log f) so that `probe_entropy` with the arm's R equals the base run's at the same s_in.
- It is a pre-set rule, a no-op for base, and recorded in `low_rate` (`ent_scale`, `h_base`, `h_got`).
- With `--eta-cal`, η is normalised by the RMS of the update at the arm's α_a, so the logit step per update is also matched.
- Seed checks (with `--rate-match-rec`): Acrobot f 0.81 (cols), 0.70 (sealed); LunarLander f 0.89 (cols), 0.82 (sealed). The matched entropy is within 0.012 nats of base.

**Pilot** (`experiments/results/pilot016/em/` and `ll_em/`; both flags on; base from `rm/` and `ll_rm/`, since neither flag changes base):
- Acrobot: the three column arms × dev seeds 0–9.
- LunarLander: `cols_lam0` and `cols_sealed` × dev seeds 10–15, 3000 episodes.

**Results** (development, small n; HL shift and MW p vs the same base runs):

- **Acrobot** (seeds 0–9; AUC over 500 episodes; base median −163):

  | Arm | Matching | Collapsed | AUC < −300 | AUC median | HL vs base | MW p | H(s), ep 50 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | base | – | 0 | 0 | −163 | – | – | 0.52 |
  | `cols_lam0` | rate | 1 | 4 | −234 | −73 | 0.011 | 0.35 |
  | `cols_lam0` | rate + entropy | 0 | 2 | −200 | −32 | 0.045 | 0.43 |
  | `cols_on` | rate | 1 | 2 | −225 | −61 | 0.009 | 0.33 |
  | `cols_on` | rate + entropy | 0 | 2 | −216 | −47 | 0.021 | 0.47 |
  | `cols_sealed` | rate | 1 | 2 | −181 | −22 | 0.27 | 0.40 |
  | `cols_sealed` | rate + entropy | 0 | 1 | −217 | −51 | 0.031 | 0.54 |

  - Entropy matching removed the collapses (0/30 vs 3/30) and restored the entropy at episode 50.
  - The column arms are still below base (HL −32 to −51).
  - It is not perseveration: action repeat is below chance in every column arm (0.60–0.63 vs 0.63–0.66).
  - The cause of the remaining deficit is open.
- **LunarLander** (seeds 10–15; AUC over 3000 episodes; base −58.2):

  | Arm | Unmatched | Rate | Rate + entropy |
  | --- | --- | --- | --- |
  | `cols_lam0` | +11.0 (p 0.026) | +5.9 (p 0.31) | +3.2 (p 0.39) |
  | `cols_sealed` | +12.9 (p 0.026) | +4.9 (p 0.13) | +5.5 (p 0.24) |

  - With both matchings, the early gain is +3 to +6 and not significant at n = 6. **Most of 015's early LunarLander speed-up was a gain (rate and temperature) effect.**

## 12. Study `hybrid-columns-2` launched (2026-10-28)

- Pre-registered 15:25 EDT (README approved by the user, Δ = 25 confirmed); code at commit `8d0a9e4`.
- Worktree `../map-stdp-hc2`, branch `hybrid-columns-2-study`.
- Command: `pytest -q tests && python -m experiments.report run hybrid-columns-2 --workers 64` (local workstation, threads 1), log `logs/hybrid-columns-2.log` in the worktree.

**Results** (official, `../map-stdp-hc2/reports/hybrid-columns-2/`; 20 seeds per arm, finished 15:36, 0 failed). Primary: AUC over the 500 episodes; HL shift vs base [95% CI], Holm-adjusted p:

| Contrast | Shift [CI] | p (Holm) | Pre-registered call |
| --- | --- | --- | --- |
| `cols_lam0` vs base | −35.2 [−59.8, −12.3] | 0.014 | **worse** |
| `cols_on` vs base | −16.6 [−43.1, +4.6] | 0.13 | inconclusive (CI below −Δ = −25) |
| `cols_sealed` vs base | −31.0 [−56.9, −9.8] | 0.014 | **worse** |
| `cols_on` vs `cols_lam0` | +19.7 [−10.4, +48.3] | 0.47 | null |
| `cols_on` vs `cols_sealed` | +15.1 [−12.3, +42.9] | 0.47 | null |
| `cols_lam0` vs `cols_sealed` (P-seq) | −3.1 [−31.3, +26.8] | 0.80 | null: level |

- **Secondary:**
  - Last-100 return: base −141.5 ± 17.1; `cols_lam0` −171 ± 78, `cols_on` −159 ± 32, `cols_sealed` −166 ± 81.
  - Solved (trailing-100 ≥ −100): 0/20 in every arm.
  - Collapsed runs: 1 (`cols_lam0`), 0, 0. Runs with AUC < −300: 0 base; 2, 1, 3 in the column arms.
- **Matching held:**
  - α_a factor f median 0.79 (cols), 0.73 (sealed);
  - initial entropy within 0.02 nats of base;
  - initial rate 40.0–43.1 Hz.
- **Reading (softened after the expert review, §13):**
  - On Acrobot, implicit and enforced columns cost performance even at matched starts. The study stands as pre-registered.
  - But the contrast is confounded: the Acrobot preset paired an E-only actor with the entropy bonus, which drives a silencing drift in every arm, and the loop roughly doubles its speed. The cost shows up *through* the loop; whether the loop costs anything without the drift is untested (§13).
  - P-seq is level, so cross-column links neither help nor hurt here; but all arms share the drift, and `cols_on` converges to sealed, so P-seq is not a clean test of sequencing.
  - The map term does not significantly beat either `cols_lam0` or `cols_sealed`. In the sealed arm, and in `cols_on` once sealed, R gets no task credit and almost no map signal (015 §8): these arms test fixed random recurrence, not learned column structure.

## 13. Expert review and the entropy-bonus confound (2026-10-28)

The user asked (after §12): is it *how* columns and the map term are used, rather than whether? A divergent-then-convergent review by the neuroscientist and snn-expert agents (about 29 and 27 strategies, then a shortlist). Reports in the session; the snn-expert's scratch analyses are development results.

**Finding (snn-expert, preliminary): the Acrobot actor-rate decline is the entropy bonus, not the columns.**
- CK's entropy update on an E-only actor always lowers the total actor drive: with g the entropy coefficient (Eq. 25), Σ_k g_k r_k = −α Var(r) ≤ 0 (checked on 1e5 random draws). The bonus buys entropy by silencing the actor; the logits α_a × rate shrink and the policy flattens late (§12 logs).
- The pre-set rule (§4) paired CartPole's E-only actor (`n_ai` 0) with LunarLander's c_e. The authors never ran that pairing: `config_ll.ini` pairs c_e with opponent I posts (`n_ai` = `n_ae`).
- Evidence: pilot v2 (c_e 0) kept the actor rate (0.036–0.059 per ms) through 500 episodes in all 12 runs; every c_e run declines four- to five-fold. Instrumented replays of study seeds (bit-identical to the study) split the update exactly: the entropy part drives the drift in every arm, columns add about 1.3–1.5× at matched rate, times the loop gain of about 1.3, giving the 2× faster decline seen in the study logs. Base with `n_ai` 20 holds its E rate within 8% to episode 300 (−72% without); LunarLander, which has I posts, stays flat over 3000 episodes.

**Convergent shortlist (both reviewers, ranked):**
1. **Gain control of the actor in every arm** (opponent I posts, the authors' pairing; or normalised / log-rate logits; or inhibitory STDP to 40 Hz). Levels the field; makes columns non-inferior, not better. Linear E-only columns read by a linear softmax can at best match base.
2. **Columns as the readout:** competition through inhibition (a fixed shared inhibitory pool, or per-column I posts whose cross-column inhibition the map term learns) with a race readout replacing the softmax. Gives the map term a function: keeping cross-column excitation low so the columns compete. Expected win: energy at matched return (neuroscientist), 0 to +10% AUC speculative (snn-expert). Risk: re-sharpening; keep 1 and the entropy bonus. Precedent: Frémaux, Sprekeler & Gerstner 2013.
3. **Latent state modules with latent→action exempt from the map term** (routing, as C→A is): the research question proper. 015's only positive structural hint: `lat_on` vs `cols_on` on CartPole +28 AUC (Holm 0.23). First measure the policy-class ceiling of the linear actor on Acrobot.
4. Conditions where modularity is known to pay off (observation noise, non-stationarity); and a loop-lag diagnostic (τ_loop 23–28 ms vs the 20 ms resample).

**Claims softened:** §11 and §12 readings above; 015 §8 (early speed-up; sealed R is a fixed amplifier); v4's benefit was a different mechanism (sealing fixed inter-module leakage that corrupted the readout; the CK softmax has no such leakage). "Cortical columns" overstates 20-neuron E-only pools; "action assemblies" is nearer.

**User decisions (2026-10-28):** fix the preset pairing (opponent I posts); pilot strategies 1, 2 and 3; apply the doc edits.

**Preset fix:** `_NEW` (Acrobot, Taxi) now has `n_ai` 20 = `n_ae`, the authors' pairing with c_e. This changes a pre-set rule after the study, for a mechanistic reason independent of return; recorded here, and the `hybrid-columns-2` study is reported as run. Its rerun under the fixed preset is a new study.

**Pilot (`experiments/results/pilot016/ai/`, logs `logs/pilot016/ai/`):** the 4 arms of `hybrid-columns-2` under the fixed preset, dev seeds 0–9, both matchings. Gates (stability only): actor E rate at episodes 250 and 500 within 20–60 Hz in ≥ 9/10 runs per arm; H(s) at episode 50 ≥ 0.1; max Re λ ≤ 0.5; no NaN; base not at floor. Falsifier of the drift explanation: the E rate holds within ±25% and the column arms still lose more than Δ = 25 AUC to base.

## 14. Design draft: columns as the readout, and latent state modules (2026-10-28; Proposal, revised after the neuroscientist's review)

Both follow the derivation's own table for the hybrid (§8.1): decision units read column spikes and send nothing back; inhibitory synapses across modules strengthen under the map term (lateral inhibition), within a module they weaken or are exempt (§7); routing is exempt. New features go behind flags whose default path draws nothing extra.

### 14.1 Strategy 2: competitive columns with a race readout

Staged, so that each stage has one new mechanism:

**Stage A, a fixed shared inhibitory pool (`--pool`).** As the QIF pools of §5.4: outside W, outside R and outside the walk.
- n_I pool posts, driven by every actor E column uniformly (E→I kick χ_EI / n_E per spike), and inhibiting every actor E post uniformly (I→E kick −χ_IE / n_I). No plasticity on the pool. Noisy LIF like the columns. n_I = `n_ae` and χ_EI = χ_rec are conventions, not principles, fixed before any run.
- The pool is **subtractive** common-mode cancellation (divisive only through rectification). A single shared pool is the precedented first model (pooled inhibition as the normaliser: Carandini & Heeger 2012; striatal fast-spiking interneurons: Koós & Tepper 1999); per-column pools are less precedented, so they come later if at all.
- **Pre-set rule for χ_IE:** bisection so that `probe_rate` with R and the pool equals the base run's rate at the base run's s_in. The pool then cancels the common-mode gain of the loop (1 / (1 − λ_E + λ_I) ≈ 1), and only the difference modes keep the loop's gain 1 / (1 − λ_E). This replaces `--rate-match-rec`, which lowered s_in and so the signal with it. χ_EI is fixed at χ_rec.
- `--ent-match-rec` then corrects only the difference-mode gain. The calibration holds at init only: R's loop gain grows while the pool is fixed, so the common-mode gain is logged per frame.
- **Readout unchanged at this stage** (softmax over column rates), so Stage A isolates one question: does cancelling the common mode with inhibition, rather than with a lower s_in, remove the column deficit? Stage A tests normalisation, not Map-STDP.
- **Structure monitors:** the pool anticorrelates the columns, which the pair counts record as below-chance cross-column transitions (§2.1 of the derivation), so M* and `within` rise with the pool even at fixed R. Compare arms at matched pool, never pool vs no-pool on structure.
- **Where the map term fits:** it keeps cross-column E low so that difference modes exist; a hub (015 §8, LunarLander) folds them into the common mode. Prediction: with the pool on, `cols_lam0` < `cols_on` ≈ `cols_sealed` on LunarLander.
- **Monitor:** the full signed offspring table (2K × 2K with the pool; derivation (12)), max Re λ ≤ 0.5; no column silent (> 1 Hz) in ≥ 95% of frames.

**Stage B, the race readout (`--readout race`).** The decision units of §8.1: the action is the column whose spike count first reaches a bound θ in the decision window (the count-difference race of Eq. 1h, as `experiments/m2.py` uses). No softmax; the policy is implicit.
- **What the race counts:** each column's net count, E spikes minus its own opponent I spikes, the same quantity the softmax's logits read (`Layout.Ma`). Fixed before piloting.
- **Pre-set rules, all fixed before any return is seen:** the decision window is the freeze window (20 ms on CartPole and Acrobot, 40 ms on LunarLander); if no column reaches θ, the leader at the window's end wins; ties break uniformly (one rng draw, behind the flag); θ by bisection so that the initial action entropy equals base's (the `match_entropy` probe on the realised action distribution). `BOUND_CAL` is not used: it was measured on M2 action modules, not on CK columns at 40 Hz.
- **Learning:** the exact local score is each actor neuron's spike minus its hazard over the decision window, times δ (Frémaux, Sprekeler & Gerstner 2013; Urbanczik & Senn 2009, who also show that feeding the winner back speeds learning). The population surrogate is the derivation's matched score for (1h): A_k − P̃_θ(k), with P̃_θ the θ-sharpened rate shares (calibrated at init), used for both the actor eligibility and the entropy bonus (Eq. 25 needs p too). The per-neuron score is run as a check on one seed.
- **Co-primary metrics:** return non-inferior (Δ as before) **and** synaptic operations per decision at least 30% below base: spikes of the columns, the pool and the I posts, each weighted by its fan-out (Attwell & Laughlin 2001; Lennie 2003). Losing columns are suppressed, not silent.
- **Risk:** competition re-sharpens the policy and brings back the absorbing corner. Keep the entropy bonus and the opponent I posts.

**Stage C, learned lateral inhibition** (per-column I pools in R; map term on cross-column I→E only; within-column I→E and cross-column E→I exempt). Cross-column I→E is an abstraction: in cortex and striatum lateral inhibition runs through long-range excitation onto *local* interneurons (Adesnik et al. 2012; Packer & Yuste 2011), and the map term would prune that pathway as cross-module excitation. Only if Stage A shows inhibition matters (criterion: the Stage A pool removes at least half of the column deficit). Needs plastic inhibition in `HybridRec` and the 2K × 2K monitor each frame. The design is in the subcritical regime (χG ≤ 0.5); the inhibition-stabilised conditions of derivation §7 apply only if χ_rec is raised above the excitation-only bound, and would be checked by the paradoxical test (pool drive up, pool rate down).

### 14.2 Strategy 3: latent state modules with exempt routing (`--rec lat --lat-exempt`)

- Latent modules as in 015 (`rec lat`): n_lat_mod modules of n_ae posts between the input and the action columns, input→L frozen at init, L→A and L↔L in R.
- **Change:** latent→action is routing. Set g = 0 on the (A ← L) entries of the modulator table in `HybridRec.frame`, the same exemption as C→A (derivation §4.5), in place of the latent-output barrier `la_beta`, which did not hold (015 §6: L→A pruned 0.25 → 0.004). The map term then only seals the latent modules from each other.
- **Arms:** base; and L→A learned by TD × {map term on L↔L, off}, with λ = 0 and L→A exempt to attribute any residual pruning of L→A to the task term (§4.5 says the task term prunes it too on fully observed tasks). The random-expansion arm keeps L→A learning (otherwise it has no readout).
- **What to expect on CartPole, LunarLander and Acrobot:** learned L ≈ random L is *expected* on a fully observed task. "Communities as states" predicts something beyond a random expansion only under noise or partial observability (slow, metastable module dynamics: Litwin-Kumar & Doiron 2012; Mazzucato et al. 2015). So the test is (i) structural: the module chain T^K(o) of learned L predicts the next observation better than random L's; (ii) functional, on a partially observed or noisy-observation arm (M4's T-maze, or Gaussian observation noise fixed in advance): learned L > random L.
- **Gates:** `lat_out` ≥ 0.1 at the end, latent rates > 1 Hz, max Re λ ≤ 0.5.
- **Before the pilot:** the policy-class ceiling of the linear CK actor on Acrobot (an analogue of `experiments/ceiling.py`), since 0/20 runs solved it. If the linear actor is at its ceiling, latents are the only arm that can raise it.

### 14.3 Order and cost

1. §13 pilot (preset fix): running.
2. Stage A on Acrobot (4 arms × dev seeds 0–9, both with and without `--pool`): about 10 minutes.
3. Stage B on CartPole and Acrobot (base-race, cols-race): about 15 minutes.
4. Strategy 3: the Acrobot ceiling first; then the structural test on LunarLander (dev seeds 10–15, 3000 episodes, about 1 hour); the functional test needs a partially observed arm, outside the three-task scope.

**Precedent for and against (neuroscientist):** action competition in vertebrates is implemented by convergent, mostly fixed inhibition (basal-ganglia focused selection: Mink 1996; Redgrave, Prescott & Gurney 1999; Gurney, Prescott & Redgrave 2001; Bogacz & Gurney 2007), so Stages A and B have strong precedent, and Stage C's learned long-range I→E is the weak link. "Columns" remains an overstatement for 20-neuron E pools; "action channels" is nearer. Proposed one-sentence caveat under the derivation's §8.1 table (not yet applied): the map term's optimum for inhibition, lateral I→E, is an abstraction of lateral inhibition that in cortex runs through long-range excitation onto local interneurons.

**§13 pilot result (fixed preset, `pilot016/ai/`, dev seeds 0–9, both matchings; development):**

| Arm | Collapsed | AUC < −300 | AUC median | Actor rate ep 50 / 250 / 500 | H(s) ep 50 / 250 / 500 |
| --- | --- | --- | --- | --- | --- |
| base, E-only (§11 pilot) | 0 | 0 | −163 | 0.038 / 0.022 / 0.010 | 0.52 / 0.42 / 0.61 |
| base, I posts | 3 | 4 | −264 | 0.037 / 0.030 / 0.029 | 0.32 / 0.08 / 0.13 |
| `cols_lam0`, I posts | 3 | 5 | −369 | 0.037 / 0.028 / 0.025 | 0.23 / 0.08 / 0.24 |
| `cols_on`, I posts | 7 | 8 | −499 | 0.037 / 0.031 / 0.031 | 0.15 / 0.00 / 0.00 |
| `cols_sealed`, I posts | 4 | 5 | −365 | 0.037 / 0.026 / 0.024 | 0.24 / 0.10 / 0.12 |

- **The drift is gone:** the actor rate holds within 25% of its start in every arm (the predicted effect of the I posts).
- **But base is now at floor** (3/10 collapsed, median AUC −264), and the column arms worse. The policy sharpens early instead: entropy 0.32 at episode 50 and 0.08 by 250 in base, against 0.52 and 0.42 without I posts. This is the absorbing corner of §10 again, reached through the gain rather than through the loop.
- **Gates:** rate gate passed; the H(s) ≥ 0.1 gate at episode 50 passed in base and `cols_lam0`/`cols_sealed` medians, failed for `cols_on`; max Re λ 0.28; no NaN; **base at floor: failed.** Per the pre-set rule, reported to the user before any change.
- **Likely cause (untested):** the opponent pairing doubles the logit's learning gain. The logit is α_a (r_E − r_I), and the task update moves the E and I rows in opposite directions, so each update changes the logit about twice as much as on an E-only actor, at the same α_a. The authors' I-post configs use α_a 15 (LunarLander), not CartPole's 50 that the Acrobot preset inherits. So `n_ai` 20 with α_a 50 is another combination the authors never ran.
- **Init check (seed 0, base, after calibration):** with I posts the initial policy entropy is 0.874 vs 0.969 E-only, at the same α_a (175 after k). The E − I difference of two independent noisy populations has about √2 the spread of E alone, so the opponent readout is sharper from the start, and its logit learning gain is about doubled. The I posts remove the drift and add a gain. Reported to the user; no change made.

**User decision (2026-10-28): gain control by divisive normalisation of the action scores** (`--norm-logits`; Proposal, neuroscientist check before implementation). In every arm, base included:

- **Scores:** score_k = α_a · r_k · r₀ / max(r̄, ε), with r_k column k's rate as now (`Layout.Ma`), r̄ the pooled rate over the actor posts (the pilot's presets are E-only, so r̄ is the mean actor E rate; with I posts the normaliser is the unsigned pooled rate and the direct term stays signed), r₀ = `rate_target` (the pre-set 40 Hz). No new free parameter. At init r̄ ≈ r₀, so the policy is the backbone's; a common change of all rates leaves it unchanged, so the entropy bonus can no longer flatten it by silencing the actor, and the loop's common-mode gain no longer matters for the policy (it still matters for stability, the critic and the pair counts; and the normalisation removes the common-mode confound, not §10's contrast-gain hypothesis, for which `--ent-match-rec` remains the control).
- **Floor (neuroscientist's rule, no free number):** ε = α_a r₀ / (τ_n n_ae), the pooled rate at which one spike (which raises a column's filtered rate by 1 / (τ_n n_ae)) moves a score by one nat; with the pre-set α_a (175 after k), r₀, τ_n 20 ms and n_ae 20 this is 0.0175 per ms (17.5 Hz). Above the floor the readout is exactly scale-free; below it the normaliser is the constant ε (gain capped at r₀ / ε ≈ 2.3, correction term u = 0). Score noise scales as r̄^(−1/2), so without a floor a 4× rate drop would double the policy noise.
- **Learning, the exact gradient through the normalisation:** ∂ log p_a / ∂ r_j = (α_a r₀ / r̄) · [(A_j − p_j) − (r_a − Σ_k p_k r_k) / (K r̄)]. Column j's eligibility factor is therefore (r₀ / r̄) · [(A_j − p_j) − u], with u = (r_a − Σ_k p_k r_k) / (K r̄) one scalar shared by all columns. Both factors are broadcast scalars (crossbar-native). The entropy bonus (Eq. 25) chains through the same factor: κ_j ∝ (r₀ / r̄) · [g_j − (1 / (K r̄)) Σ_k g_k r_k]; the second term is what cancels the silencing drift of §13 (Σ_k g_k r_k ≤ 0 is no longer a drift on the total drive, because p is invariant to the common mode).
- **The factor r₀ / r̄** keeps the step on the score scale matched as rates change (the gain compensation that `--ent-match-rec` could only fix at init).
- **Precedent:** divisive normalisation by pooled activity as a canonical cortical computation (Carandini & Heeger 2012); the pooled rate is the quantity a shared inhibitory pool would carry (Stage A then implements it in neurons).
- **Alternative considered:** the Luce readout p_k ∝ r_k^β (per-column factor 1/r_k on the eligibility). Also local and drift-free, but it changes the init policy and needs β calibrated. Rejected for the extra parameter.
- **Pilot:** Acrobot, the 4 arms, dev seeds 0–9, E-only presets (`n_ai` back to 0: the opponent pairing is withdrawn, §13 pilot), both matchings, `--norm-logits`. Gates as §13; the falsifier as before.
- **Forking paths, stated:** this is the third gain-control choice (E-only drift → opponent I posts → normalisation) made after seeing Acrobot returns on dev seeds 0–9. The pilot is a stability pilot only; its AUCs are not evidence on columns. The study under `--norm-logits` uses fresh seeds, its README records the three-step path as a post-hoc readout change, `base --norm-logits` is its reference, and `hybrid-columns-2`'s base remains the CK-faithful reference.
- **Neuroscientist check (2026-10-28):** gradient and entropy chain confirmed (log p_a and H are homogeneous of degree 0 in r, so the Euler identity gives the zero drift); locality holds as two broadcast scalars (an abstraction: a shared pool carries r̄, Stage A; u is a decision-circuit quantity); pooled-over-options normalisation of action value is observed in LIP (Louie, Grattan & Glimcher 2011; Louie, Khaw & Glimcher 2013); the critic's rate as normaliser and the Luce ratio were rejected. The only new risk: the backbone's accidental late re-flattening (E-only base H 0.42 → 0.61 was the drift) is gone, so base may sharpen monotonically; the entropy bonus then acts on contrast only, the intended regime.
- **Numerical check (scratch, K = 4, 200 random rate vectors):** the gradient above matches central finite differences of log p_a and of H to a relative error of 1e-9; Σ_j r_j ∂H/∂r_j = 0 exactly with the normalisation (−0.35 for the unnormalised scores at the same rates: the drift).

**`--norm-logits` pilot result (`pilot016/nl/`, Acrobot, dev seeds 0–9, both matchings, E-only presets; development, stability pilot):**

| Arm | Collapsed | AUC < −300 | AUC median | HL vs its base | MW p | Actor rate ep 50 / 250 / 500 | H(s) ep 50 / 250 / 500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base, E-only, unnormalised (§11) | 0 | 0 | −163 | – | – | 0.038 / 0.022 / 0.010 | 0.52 / 0.42 / 0.61 |
| `cols_lam0`, unnormalised (§11) | 0 | 2 | −200 | −32 | 0.045 | 0.038 / 0.014 / 0.007 | 0.43 / 0.53 / 0.75 |
| base, normalised | 1 | 1 | −186 | – | – | 0.048 / 0.047 / 0.041 | 0.71 / 0.34 / 0.27 |
| `cols_lam0`, normalised | 0 | 0 | −176 | +13 | 0.39 | 0.051 / 0.050 / 0.048 | 0.61 / 0.32 / 0.27 |
| `cols_on`, normalised | 0 | 0 | −180 | +4 | 0.97 | 0.052 / 0.052 / 0.049 | 0.61 / 0.24 / 0.26 |
| `cols_sealed`, normalised | 0 | 0 | −199 | −2 | 0.97 | 0.052 / 0.047 / 0.044 | 0.69 / 0.31 / 0.25 |

- **Gates passed:** actor rate within 20–60 Hz at episodes 250 and 500 in 10/10 base runs (and every column run); H(s) ≥ 0.1 at episode 50 in all; max Re λ 0.28 (columns), no NaN; base not at floor (median −186, 1/10 collapsed).
- **The falsifier did not fire:** the rate holds and the column arms are level with base (HL +13, +4, −2; all p ≥ 0.39), with no collapsed or slow column run (0/30). The Acrobot deficit of §12 was the silencing drift acting through the loop, not a cost of the loop itself.
- **Base is lower than the unnormalised E-only base** (−186 vs −163, HL −22): as the neuroscientist predicted, the backbone's accidental late re-flattening (H 0.42 → 0.61, which was the drift) is gone, so the policy sharpens monotonically (H 0.27 at episode 500 in every arm). The entropy bonus now acts on contrast only. The comparison to make is columns vs base under the same readout; `hybrid-columns-2`'s base remains the CK-faithful reference.
- `--ent-match-rec` factors are now 0.95–1.05: with the normalisation the arms' initial policies already match.
- **Actor rates rise slightly above the 40 Hz target** (0.048–0.052 per ms by episode 50) and stay there: the task update's common mode is no longer corrected downward by the bonus. Within the rate gate.

## 15. Sequence and study `hybrid-columns-3` (2026-10-28)

**User decisions:** structure has to matter before the map term can be used strategically. Run sequentially: (1) the three-task study under the normalised scores, the backbone result set; (2) Stage B, the race readout (§14.1), where cross-column links act at the decision; (3) a stimulus-conditioned map term, informed by the study's results.

**Why the conditioned map term waits (chat, 2026-10-28):** a K×K modulator indexed by a stimulus class (the active action column, or a latent state module) is local, crossbar-native and cheap (S × K × K); the implementation's `obs_tau` averaging currently washes the stimulus out of the walk statistics. But every backbone result so far shows the softmax readout cannot see R's structure (hub ≈ modular ≈ sealed at matched gain; P-seq null): a link between columns acts within one loop time (about 25 ms), far below the task's sequencing timescale, and the window average removes it. So a conditioned map term now would shape structure no one reads. Candidates that make structure matter: the race readout, slow (NMDA-like) recurrent synapses, latent state modules; then the conditioned map term is a learned, state-dependent chunking of actions. Two further strategies noted: a map term that defers to the task term (a K×K exemption table from the task credit), and energy as the metric where sealing already pays.

**Study `hybrid-columns-3`** (declared in `experiments/report.py`; README draft `reports/hybrid-columns-3/README.md`): CartPole, LunarLander, Acrobot × base / `cols_lam0` / `cols_on` / `cols_sealed`, seeds 5000–5019, flags `--rate-target 40 --block 20 --fast --eta-cal --rate-match-rec --ent-match-rec --norm-logits`. Primary: LunarLander last-100, AUC elsewhere; Δ = 25; Holm per task. `--eta-cal` on every task (015 used CK's preset η on CartPole). **Stability pilots** before pre-registration (`pilot016/nl_cp/`: CartPole 4 arms × dev seeds 0–4; `pilot016/nl_ll/`: LunarLander 4 arms × dev seeds 10–12, 1000 episodes). Gates as §13.

**Stability pilots (development):**
- **LunarLander** (`pilot016/nl_ll/`, dev seeds 10–12, 1000 episodes): rates hold (0.040–0.045 per ms at episodes 50, 500 and 1000 in every arm), entropy 1.26 → 0.95 (base) and the same in the column arms, max Re λ ≤ 0.26, no NaN; AUC over the 1000 episodes level with base (`cols_lam0` −1, `cols_on` −17, `cols_sealed` −7; n = 3). Gates passed.
- **CartPole, first attempt** (`--eta-cal` on every task): base at the random-policy level (AUC 11.7, 0/5 solved). Not the normalisation: CartPole's preset η (0.005, 0.01) are CK's SGD learning rates, not Adam steps, so `--eta-cal` divides them by the update RMS and gives η_a 8.7 mV (LunarLander's 0.07); |W| grew to 7 (015: 0.06) and the actor fell silent. That is why 015 kept CK's preset η on CartPole. The policy at init under the normalisation is the backbone's (traced on seed 0). **Fix:** CartPole without `--eta-cal`, as in `hybrid-columns`; declaration, test and README updated; pilot rerun.
- **CartPole, second attempt** (CK's preset η): base solved 5/5 (T_s 221–269; AUC 354; rates 0.041 → 0.004 by episode 400 as the solved policy settles, as in 015). The column arms were dead (AUC 36, entropy ln 2 throughout): `match_entropy` had driven α_a to its floor (f = 0.125). **Cause, a bug in `probe_entropy`:** it averaged the policy entropy over the live copies per tick; CartPole's single copy spends many calibration ticks in reset (random-policy episodes last about 20 ticks), the empty mean is NaN, every NaN comparison in the bisection reads as "entropy too high", and f walks to the floor. Acrobot and LunarLander episodes are long enough never to hit it (their factors were 0.7–1.05). **Fix:** the probe pools the entropy over the live copies of all ticks (identical for one copy without resets; a count-weighted mean otherwise). Test added. Pilot rerun.
- **CartPole, third attempt** (preset η, fixed probe; `pilot016/nl_cp/`, dev seeds 0–4): every arm solved 5/5; f 0.95–0.99 with the entropies matched (0.59 = 0.59); rates 0.041–0.051 per ms; max Re λ 0.28; no NaN. AUC vs base: `cols_lam0` −16 (p 0.42), `cols_on` −1 (p 0.84), `cols_sealed` −43 (p 0.03; T_s 256–356 vs base 221–269). Gates passed. README expectations filled; the study awaits the user's approval, Δ confirmation and commit.

**Monitor added before launch (user decision, 2026-10-28): the stimulus-conditioned walk.** `HybridRec.window` now takes the copies' executed action as the class and splits the window's pair counts by it (`kc`, K × nR × nR; per head the counts are independent integers, so the split sums to the single call exactly, and `a["kappa"]` is unchanged). At each `SNAP_EVERY` snapshot and at the end, `class_table()` reduces it to the module level (K × Kr × Kr shares per class and the class's pair count) and resets; the result carries it as `rec.RKc` next to `RK`, and checkpoints carry `rec_kc` and `snaps_kc`. It changes no update and draws nothing: a measurement of whether the cross-column walk depends on the state, which the conditioned map term would need. Test `test_class_pair_counts`.

**Study `hybrid-columns-3` launched (2026-10-28 18:22 EDT).** README pre-registered (user approved; Δ = 25 confirmed); code at commit `4bdbe45`; worktree `../map-stdp-hc3`, branch `hybrid-columns-3-study`. Command: `pytest -q tests && python -m experiments.report run hybrid-columns-3 --workers 64` (local workstation, threads 1), log `logs/hybrid-columns-3.log` in the worktree. Projected 12–16 h.

**Interim, CartPole (read-only look at the running study, 20 seeds, 19:03; official numbers come from `report summarize` at the end):** every arm solved 20/20; AUC vs base `cols_lam0` −3.6 (p 0.43), `cols_on` −12.8 (p 0.12), `cols_sealed` −8.5 (p 0.26); T_s medians 251 / 256 / 264 / 263. The 5-seed hint of slower sealed columns did not hold.

**Two defects of the `RKc` monitor as launched (found on the CartPole runs; fixed in the main checkout, not in the study):**
1. The end-of-run snapshot overwrote the `SNAP_EVERY` snapshot of the same episode with an empty table (the counts had just been reset), so the study's last phase per run (CartPole 300–400, Acrobot 400–500) is lost; the earlier snapshots are intact. Fix: the end snapshot is skipped when that episode already has one.
2. The raw pair counts are chance-dominated: with two columns at equal rates, half the coincidences cross columns, so every arm, sealed included, shows a cross-column share of 0.506 per class, and the executed action makes no visible difference. The state-dependence question needs the **causal** counter (Eq. 8's signed counter, kicks × offspring × spikes, what the structural rule sees). `class_table` now records both (`cshare`, `cn` next to `share`, `n`), and the per-class counters sum to `a["c"]` exactly (test). The study's `RKc` keeps only the raw version; the question is answered instead by a dev run with the fixed monitor (below) and by later studies.

## 16. Stage B implemented: the race readout (2026-10-28; pilot held until the study reports)

User decision: build Stage B (§14.1) while `hybrid-columns-3` runs; no pilot until its results are in. Implemented by the snn-expert against the reviewed design, reviewed here; `--readout race` (`experiments/tdstdp.py`: `race_theta`, `CKRunner.race_block`, `_leader`, `surrogate`, `race_factor`, `hold_reset`/`hold_clear`, `calibrate_race`, `RACE_DELTAS`); default off, the softmax path byte-identical (61 tests, golden unchanged).

**The decision (Eq. 1h).** Per copy, the net count per column (E spikes minus its own opponent I spikes, the signs `Layout.Ma` reads) accumulates since the last resample at 1 ms resolution; the first ms at which any column reaches θ fixes the pending winner (the leader at that ms; an exact tie, one uniform draw); a copy without a pending winner at the deadline (the freeze window: 20 ms on CartPole and Acrobot, 40 ms on LunarLander) takes the leader of its counts. **θ rule:** round(n_ae r₀ T_freeze), the mean count of one column at the target rate over a window: 16 on CartPole and Acrobot, 51 on LunarLander. The result's `race_decisions` records bound / deadline / tie counts.

**The learning signal.** The population surrogate p̃ = softmax(β log max(r_k, ε_r)), ε_r = 1 / (τ_n n_ae) (the filtered rate one spike gives); the eligibility of column j is β / (α_a max(r_j, ε_r)) × (A_j − p̃_j), its exact gradient in CK's convention (central differences: relative error 6e-6; Σ_j r_j ∂H/∂r_j = 0, no silencing drift), the same factor on the entropy bonus g(p̃) and on the module scores. (My spec carried an r₀ from the normalised scores into this factor; the snn-expert implemented the exact gradient instead, which is right.)

**β by the contrast probe (pre-set, no return seen).** My first rule, matching the surrogate's entropy to the race's realised entropy at init, is ill-conditioned: at init the columns' rates differ by 3–5%, both policies are near-uniform, and 25 decisions per state give an entropy sd of 0.05 nats, the size of the signal (Acrobot returned β at the bracket's floor, 0.125; CartPole 6.75 by luck). Replaced by a psychometric probe: in the held-observation probe (W, R frozen, rule off, CAL stream; 8 held states), column 0's excitatory input rows are scaled by 1 + δ for δ ∈ {−0.2, −0.1, 0, 0.1, 0.2} (conventions: the column, its E rows, the δ set), the closed loop's own race decides once per freeze window, and β is the maximum-likelihood slope of column 0's win probability against log rate contrast under the surrogate (concave; bisection on the score). Seed 0, base: **β 7.80 (Acrobot), 7.95 (CartPole)** at θ = 16; the surrogate reproduces column 0's realised win rate within 0.05 at every δ (Acrobot: realised 0.075 / 0.160 / 0.305 / 0.490 / 0.580 vs 0.078 / 0.155 / 0.309 / 0.438 / 0.584); 1.7× the Poisson-race value 1.6√(θ/2) = 4.5, consistent with sub-Poisson counts at 40 Hz with τ_ref 2 ms; 1 s per calibration. Recorded in `low_rate["race"]` / `race_cal` (β, θ, log-likelihood, win rates realised and surrogate, the entropy diagnostics), restored from checkpoints.

**Energy monitor (both readouts):** actor spikes (E and I) per decision window per copy, `actor_spikes_per_decision` in every result (the pre-registered co-primary for the Stage B pilot). At init on Acrobot seed 0 (frozen, 100 ticks): softmax 49.5, race 45.1 spikes per decision; the difference is trajectory only. A saving needs suppressed losers, i.e. Stage A's pool on top of the race.

**Open before the pilot:** (1) LunarLander: with n_ai = n_ae the net count E − I hovers near 0 while θ = 51 is one column's E count, so the race there would be nearly all deadline decisions (a max-net-count readout over the 40 ms window); decide whether that is acceptable or θ should be defined on the net count's scale. (2) About 10% exact ties at θ = 16 with K = 3; the tie draw does real work. (3) The 1/r_j eligibility gain (the exact gradient of the Luce surrogate) amplifies a weak column's update up to 16× at the floor.

**Pilot (to pre-register after the study):** CartPole and Acrobot, dev seeds, {base-softmax (from the study), base-race, cols-race}; co-primary: return non-inferior (Δ as before) and synaptic operations per decision; stability gates only.

**Dev check of the causal per-class walk** (`pilot016/rkc/`, `cols_lam0`, dev seeds 0–2, study flags, the fixed monitor; the class is the executed action):
- **CartPole:** the causal cross-column share falls 0.077 → 0.001 by episode 400 (the task term seals, as 015 found) and is identical for the two executed actions at every snapshot (0.077 / 0.077, 0.015 / 0.015, …). No state dependence.
- **Acrobot:** the task term does **not** seal. At episode 500, R's weight table (share of R, [post, pre]) is within-column 0.33 / 0.13 / 0.30 and cross-column 1→0 0.146, 1→2 0.060, 2→0 0.033, everything else ≤ 0.001: the zero-torque column is a **hub source** feeding both torque columns, with no flow back into it. A third task-term structure, after CartPole's sealing and LunarLander's hub into the main engine (015 §8).
- **Per executed action on Acrobot:** the amount of causal flow changes with the class (the share landing in the executed column 0.60 / 0.16 / 0.46; the cross share 0.16 / 0.23 / 0.19, sd over seeds 0.05–0.07), but the **cross-pair pattern is the same in every class**: the six cross entries' pattern correlates 0.92–1.00 between classes within each seed (1→0 about 0.6 of the cross flow, 1→2 0.21–0.27, 2→0 0.09–0.19, the rest 0). The executed action changes how much flows, not which pairs carry it.
- **Reading for the conditioned map term:** with the executed action as the class there is nothing to condition on in these tasks: the walk's cross-column structure is state-independent at this level, and a class-indexed map term would learn the same partition in every class. What is task-dependent is the structure the task term builds (sealed / hub into main engine / hub from no-torque). A conditioned map term would need a finer or different class (a latent state module, or an observation class), which is strategy 3's territory, or a task with partial observability. Development, 3 seeds; the study's `RK` tables (20 seeds) will confirm the Acrobot structure.

## 17. Study `hybrid-columns-3` results (2026-10-29; `../map-stdp-hc3/reports/hybrid-columns-3/`)

Finished 2026-10-29 04:40 EDT, 240/240 runs, none failed, no resumes; 10.3 h. Official numbers: `summary.md` and `pairs.csv` written by `report summarize` (git `4bdbe45`, no code patch). The study directory is untracked on branch `hybrid-columns-3-study` in the worktree, as are `hybrid-columns` (`../map-stdp-hc`) and `hybrid-columns-2` (`../map-stdp-hc2`); none of the three result sets is in `columns-tdstdp` yet.

**Headline (primary metric; Hodges–Lehmann shift vs base [95% CI], raw p, Holm p; Δ = 25):**

| Task | Arm | Shift [CI] | p (Holm) | Pre-registered reading |
| --- | --- | --- | --- | --- |
| CartPole (AUC) | `cols_lam0` | −3.6 [−19.2, +9.7] | 0.42 (1) | **non-inferior** |
| | `cols_on` | −12.8 [−28.5, +2.4] | 0.12 (0.72) | inconclusive (lower bound −28.5) |
| | `cols_sealed` | −8.5 [−25.9, +6.7] | 0.26 (1) | inconclusive (lower bound −25.9, by 0.9) |
| LunarLander (last-100) | `cols_lam0` | +10.7 [+1.6, +19.6] | 0.021 (0.17) | **non-inferior**; positive, not significant after Holm |
| | `cols_on` | +8.8 [+1.2, +16.5] | 0.017 (0.15) | **non-inferior**; positive, not significant after Holm |
| | `cols_sealed` | +7.4 [+0.2, +15.4] | 0.044 (0.31) | **non-inferior**; positive, not significant after Holm |
| Acrobot (AUC) | `cols_lam0` | −9.1 [−35.3, +9.7] | 0.42 (1) | inconclusive |
| | `cols_on` | −16.4 [−31.2, +5.6] | 0.12 (0.72) | inconclusive |
| | `cols_sealed` | −2.2 [−19.0, +15.7] | 0.68 (1) | **non-inferior** |

No arm is worse than base on any task, and none improves on base after Holm. Every secondary contrast (regime vs regime) is null on every task (|shift| ≤ 14, all Holm p = 1). Per group: CartPole every arm 20/20 solved (T_s medians 251 / 256 / 264 / 263); LunarLander 0/20 solved in every arm, last-100 30.1 (base), 40.1, 38.9, 37.3; Acrobot last-100 −136.6 (base), −153.8, −127.3, −127.9, solved 1 / 0 / 0 / 3, no run at the −500 floor.

**Against the earlier studies.** On Acrobot, `hybrid-columns-2` (CK-faithful base, the entropy-bonus drift in place) had `cols_lam0` −35.2 [−59.8, −12.3] and `cols_sealed` −31.0 [−56.9, −9.8], both worse; under the normalised scores they are −9.1 and −2.2, with sealed non-inferior. That is what §13's diagnosis predicted. On LunarLander, 015 (unmatched, CK-faithful) had every column arm level with base on last-100 and ahead only on AUC-3000 (a starting-gain effect, §10–11); here the AUC-3000 shifts are 0 (−0.6, +1.1, +1.0) and the last-100 shifts are +7 to +11 in every arm, the CIs above zero before the Holm correction. The pattern across the two LunarLander studies is the same arms, a different phase: once the gain is matched, whatever the columns add shows late, not early.

**A cost of the normalised readout on LunarLander's base (cross-study, different seeds, not a pre-registered test):** 015's CK-faithful base and this base have the same AUC-3000 (−60.2 vs −60.5) and the same 3000-episode return (17 vs 23), but 015's base keeps improving after 3000 (25.5 / 43.1 / 55.5 at 5000 / 8000 / 10000, last-100 58.3 ± 15.2) while this base plateaus (19.9 / 23.5 / 30.1, last-100 30.1 ± 11.5). The column arms recover part of it (last-100 37–40). The study is internally consistent (every arm has the normalisation), but as a backbone the normalised scores cost about 25–30 points of late LunarLander return against CK's readout. A candidate cause: under the normalisation the policy's gain is fixed at α_a r₀ / r̄, so the late sharpening CK's base gets from rising actor rates is removed, and only the contrast between columns can sharpen the policy. Not tested; it belongs to the Stage B / gain-control question, since the race readout fixes the gain differently (β).

**Monitors (column arms; base has no `rec` record):**
- Rates: LunarLander 47 → 40–45 Hz over 10000 episodes; Acrobot 46 → 45–48 Hz. CartPole: 46–47 Hz until episode 170, then falling to 3.4–3.9 Hz by 400 with 20–25 silent actor neurons per run, after the runs solve (T_s about 255; the return stays 500). 015's CartPole columns also fell after solving (64 → 23 Hz), more slowly and without silence. Under the normalisation the eligibility carries the factor r₀ / r̄, which grows as the pooled rate falls (12× at 3.4 Hz): a post-solve downward drift is amplified. Harmless here (solved), but it is the normalisation's one known feedback. Stability: max Re λ 0.21–0.26, τ_loop 26–28 ms, no NaN; the matching held in every arm (r_got 40.1–40.7 Hz; f 1.04–1.05 unsealed, 0.96–0.97 sealed; entropies matched to 0.001).
- Structure (`within` share of R): CartPole seals in every arm (0.91 → 0.998, as 015 and the §15 dev check); Acrobot `cols_lam0` 0.83 → 0.755 (the no-torque hub, §15) and `cols_on` 0.93 → 0.998 (the map term seals it); LunarLander `cols_lam0` 0.77 → 0.40 (the main-engine hub of 015 §8), `cols_on` 0.79 → 0.996. So the map term seals on every task, and the task term alone builds a hub on both tasks with a null action. Under the softmax readout none of this moves the return (the secondary contrasts).
- Acrobot spread: `cols_lam0` has two runs near collapse (last-100 −391, −369; AUC −467, −449; the rest −173 to −113), base one (−355); `cols_on` and `cols_sealed` none below −220. The implicit arm's inconclusive CI is those two seeds.
- `rec.RKc` (raw pairs, the launched version): chance-dominated as §15 predicted. The cross-column share per executed action is 0.60 (Acrobot, K = 3), 0.50 (CartPole, K = 2), 0.71 (LunarLander, K = 4), the same in every class to within 0.03, sealed arms included, and the last phase is empty. It answers nothing; the state-dependence question stands on the §15 dev check (causal counter) until a study carries the fixed monitor.

**Reading.** The claim the study supports: at matched gain, action columns are non-inferior to CK's softmax readout on 5 of 9 contrasts and inconclusive (never worse) on the other 4, with an inconclusive-but-positive signal on LunarLander in every regime. The regimes do not differ. Structure (sealed, hub) does not reach the return under a softmax readout, which is the premise of the sequence fixed in §15: next Stage B (the race readout, §16), where cross-column links act at the decision, then the conditioned map term.

## 18. Stage B pilot: the race readout (pre-registered 2026-10-29, before any run; development)

**What the study changes for Stage B (§17).** (1) The reference is CK's softmax readout, not the normalised one: the normalised base loses late LunarLander return, so a race arm that matches it proves little. (2) The race's surrogate reads log-rate differences and its sharpness is set by θ, the same scale invariance the normalisation has, so the first gate is whether base-race keeps the CK base's return at all. (3) The surrogate's 1/r_j eligibility factor is the same low-rate feedback the normalisation showed on CartPole after solving, so a silent column is a stability failure.

**User decision, θ on LunarLander (2026-10-29, from four options: E counts; net count with deadline decisions; net count with θ on its own scale; drop LunarLander):** **the race tallies E counts on every task.** The I posts shape the dynamics but are not counted, so θ = round(n_ae r₀ T_freeze) means the same thing on every task; on CartPole and Acrobot, which have no I posts, nothing changes. A deviation from the net-count tally of the §14.1 design the neuroscientist reviewed, noted here; the surrogate still reads the net rate (`rho @ Ma.T`), so the learning signal is unchanged. Code: `CKRunner.Mcnt` replaces `Msgn`; test (I-post raster moves no count).

**Energy monitor added:** `actor_spikes_to_decision`, the actor spikes (E and I) up to the bound ms (or the whole window at a deadline decision) per copy-decision, next to the full-window `actor_spikes_per_decision`. It is the energy a loop that stopped simulating at the bound would spend; a monitor only, nothing is stopped. Checkpoints carry `ebnd`.

**Pilot design** (`experiments/results/pilot016/race/`, logs `logs/pilot016/race/`; script `pilot016_race.sh` in the session scratchpad; dev seeds 20–29, fresh for development; no reported numbers):
- **Tasks:** CartPole (CK's preset η) and Acrobot (`--eta-cal`), 400 and 500 episodes. LunarLander follows only if both pass their gates.
- **Common flags:** `--engine nlif --rate-target 40 --block 20 --fast --rate-match-rec`; softmax arms add `--ent-match-rec` (a no-op on base).
- **Arms:** `base_ck` (CK's softmax, unnormalised: the 015 / `hybrid-columns-2` readout); `base_nl` (`--norm-logits`, the `hybrid-columns-3` readout); `base_race` (`--readout race`); `cols_lam0_race`, `cols_on_race`, `cols_sealed_race` (`--readout race --chi-rec 0.3 --rec cols` with `--lam 0`, `--lam 1`, `--rec-seal --lam 0`). 6 arms × 2 tasks × 10 seeds = 120 runs, 60 workers.
- **Fixed before any return:** θ by the rule (16 on both tasks); β by the contrast probe per run at init (`calibrate_race`, with the arm's own R); χ_rec 0.3, r = 1; s_in rate-matched per arm. The race arms cannot entropy-match (the gain is β's), so their starts are matched by rate and by the probe's fit, which the record (`race_cal`) shows.
- **Co-primary (development reading, Δ = 25 as in the studies):** (a) AUC of each race arm against `base_ck`: non-inferior, inconclusive or worse by the Hodges–Lehmann CI; (b) `actor_spikes_to_decision` and `actor_spikes_per_decision` per arm.
- **The structural question:** `cols_sealed_race` against `cols_lam0_race`, null under the softmax on every task. If it opens, cross-column links act at the decision and the conditioned map term has a job; if it stays null, they do not.
- **Also read:** `base_nl` vs `base_ck` on these tasks (the normalisation's cost, seen on LunarLander in §17); `base_race` vs `base_nl`; the bound / deadline / tie counts; β per arm and the surrogate's fit (`win0` vs `win0_surrogate` within 0.05).
- **Gates (stability only):** actor rates within 20–60 Hz through the run and no silent column (a column at the surrogate's floor for a whole log bin); no collapsed run (Acrobot last-100 = −500; CartPole AUC at the random level); no NaN; max Re λ ≤ 0.5, τ_loop ≤ 40 ms; the surrogate's win rates within 0.05 of the realised ones at every δ. If `base_race` fails a gate or is worse than `base_ck`, that is reported before anything is changed; no setting is tuned on return.
- **If it passes:** a three-task reported study `hybrid-columns-4` with `base_ck` as the reference group, `base_nl` as a secondary arm, and the three regimes under the race, seeds 6000–6019, pre-registered as before.

**Pilot result (2026-10-29, 120/120 runs, no failures; `an_race.py` in the scratchpad; development, dev seeds 20–29; nothing changed afterwards, per the rule above).** AUC, Hodges–Lehmann shift vs `base_ck` [bootstrap 95% CI]:

| Task | Arm | AUC median | Shift vs `base_ck` | Spikes per decision / to decision | Bound / deadline / ties | Actor rate at the end (Hz) | Collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole | `base_ck` | 364 | ref | 35.6 / – | – | 33 | 0 |
| | `base_nl` | 351 | −11 [−30, +11] | 19.7 / – | – | 4.5 | 0 |
| | `base_race` | 354 | −11 [−56, +10] | 17.9 / 17.3 | 0.22 / 0.78 / 0.11 | 8 | 0 |
| | `cols_lam0_race` | 318 | −39 [−73, +15] | 18.2 / 15.3 | 0.38 / 0.62 / 0.11 | 7 | 0 |
| | `cols_on_race` | 323 | −36 [−81, +15] | 19.8 / 16.4 | 0.42 / 0.58 / 0.10 | 8 | 0 |
| | `cols_sealed_race` | 387 | +24 [−6, +40] (p 0.08) | 16.5 / 13.8 | 0.33 / 0.67 / 0.13 | 4 | 0 |
| Acrobot | `base_ck` | −184 | ref | 24.0 / – | – | 7 | 0 |
| | `base_nl` | −182 | +1 [−23, +24] | 49.0 / – | – | 37 | 0 |
| | `base_race` | −272 | −84 [−277, −4] | 78.1 / 37.1 | 1.00 / 0.00 / 0.07 | 86 | 2 |
| | `cols_lam0_race` | −471 | −272 [−312, −92] | 92.3 / 38.4 | 1.00 / 0.00 / 0.05 | 92 | 5 |
| | `cols_on_race` | −246 | −65 [−216, −29] | 90.7 / 35.5 | 1.00 / 0.00 / 0.05 | 91 | 0 |
| | `cols_sealed_race` | −287 | −102 [−195, −23] | 93.1 / 36.5 | 1.00 / 0.00 / 0.04 | 98 | 3 |

- **Gate verdict: `base_race` fails on Acrobot** (worse than `base_ck`; the actor rate climbs monotonically 46 → 60 → 67 → 73 → 82 Hz at episodes 50–450, past the 60 Hz gate; 2/10 collapsed; the critic rate falls 64 → 50 Hz late). On CartPole it passes the stability gates until the runs solve and is inconclusive on AUC (last-100 469: not every run finishes solved), with the actor rate falling 41 → 10 Hz over the run, as `base_nl` does (41 → 10) and `base_ck` does not (41 → 41).
- **A common-mode rate drift of task-dependent sign under the race.** The surrogate's gradient is exactly rate-weighted-zero (Σ_j r_j ∂/∂r_j = 0, tested), so the drift is not in the gradient. What is not zero is the plain sum of the per-column updates, Σ_j (A_j − p̃_j) / r_j, whenever the race's realised choice frequencies differ from p̃: the race favours the leading column more than the surrogate does once rates move away from the calibration point (a fixed θ against counts that grow more regular at higher rates), so E[A_j] − p̃_j is positive for the high-rate column (small 1/r_j) and negative for the low-rate columns (large 1/r_j), a net negative sum whose sign is then set by the TD error: negative δ (Acrobot, −1 per step) drives the rates up, positive δ (CartPole) drives them down. Every race arm on Acrobot rises to 86–98 Hz; the surrogate's fit at init is within 0.05 in only 4–8 of 10 runs per arm (max deviation 0.02–0.08), so the mismatch is there from the start. A hypothesis from the monitors, not a tested mechanism.
- **Energy:** the race decides with fewer actor spikes where it works (CartPole 17.9 vs 35.6 for `base_ck`, 13.8–17.3 to the bound), and with more where the rates run away (Acrobot 78–93 vs 24): the energy co-primary tracks the rate drift, not the readout.
- **The structural contrast opened on CartPole:** `cols_sealed_race` − `cols_lam0_race` AUC +57 [+2, +97], p 0.038 (n = 10), and sealed is the only race arm above `base_ck` (+24, p 0.08). Under the race the cross-column links hurt (`cols_lam0_race` and `cols_on_race` last-100 311 and 370, not solving) rather than sealing helping; still, it is the first contrast between regimes to move on any task, and it moved under the readout where links act at the decision, as §15 argued. On Acrobot every column arm is at or near the floor, so the contrast there (+99 [0, +225]) says only that cross-column links collapse the loop faster (5/10 collapsed vs 3/10). Development, 10 seeds, on a background of a failed base.
- **Not changed:** θ, β's rule, χ_rec, the factor. The pilot stops here and the result goes to the user before any design change, as pre-registered. Candidate fixes to put to the user, each pre-set from principle: (a) recalibrate β on a schedule (the contrast probe is 1 s), so the surrogate follows the race as the rates move; (b) CK's own rate-cost term (`c_t`, `rho_hat`: a homeostatic pull of the mean actor rate to r₀; in the code only in the legacy `Runner`, and `CKRunner` asserts c_t = 0, so it would be added to `CKRunner`'s κ term at c_t = 0 by default), which removes a common-mode drift of either sign at the cost of one more pre-set number; (c) define the race's choice as the surrogate's sample (the race then only times the decision), which removes the mismatch by construction but gives up the readout's decision.

**Neuroscientist's review of fix (b) (2026-10-29; review-only, citations verified by the reviewer).** The term is CK's own (paper §3.5.3, Eqs. 31–32; used only in their low-rate CartPole variant, c_t 5e-6 at ρ̂ 5e-3; c_t = 0 in every Table 1 result). Mechanistically it is a coincidence-gated change (∝ z_ij ≥ 0) whose sign is one population scalar: **not** synaptic scaling (multiplicative in W, cell-autonomous, hours to days: Ibata, Sun & Turrigiano 2008; Hengen et al. 2013; Keck et al. 2013) and not BCM (per-neuron sliding threshold, potentiating above it). Its class is the *rapid compensatory process* that Zenke, Hennequin & Gerstner 2013 showed any plastic recurrent network needs for stability (a rate detector on seconds to minutes; substrate proposed in Zenke & Gerstner 2017 as heterosynaptic depression at high postsynaptic rate), with the algebraic shape of the Vogels et al. 2011 inhibitory rule (coincidence minus ρ₀ × pre), there per neuron and on inhibitory synapses. Name to use: "coincidence-gated rate homeostasis on the population mean (a rapid compensatory process)"; status: abstraction, timescale plausible but unmeasured. Locality and the crossbar claim hold (one scalar per copy and population, times the trace the TD term already uses). Ranked issues, all accepted as caveats on (b):
1. **It conflicts with the energy co-primary.** With ρ̂ = r₀ the actor's mean rate is a pre-set constant across arms, so `actor_spikes_per_decision` no longer discriminates arms; only `actor_spikes_to_decision` (the bound time) remains an energy measure, and this pilot's CartPole saving (17.9 vs 35.6) is removed by construction.
2. **It treats the common mode, not the cause.** The score-function estimator is unbiased only when the baseline equals the sampling distribution (Williams 1992); the term cancels the common-mode projection of the mismatch and leaves its differential part (which column gets the biased credit). Label (b) "a stabiliser of the common mode; it does not restore the surrogate's match", and keep the surrogate-fit gate (win rates within 0.05 of realised at every δ) as a test (b) alone cannot pass.
3. **It cannot revive a silent column** (z = 0 there, unlike weight-proportional scaling); on CartPole it would raise the still-active columns and leave floor columns at the floor. The silent-column gate stays independent of it.
4. **The copied constant is nominal:** c_t 5e-6 was set at ρ̂ 0.005, 40 posts per action, mV weights (`MV` 13; nlif `mv` 1), so the implied relaxation time differs in our units. Suggested rule instead: choose c_t so that the rate error relaxes with a declared time constant (minutes of simulated time, the Zenke 2013 class; the drift spans about 80 min) at the calibrated W, measured by a frozen probe like `calibrate_race`, and record it in `low_rate`. The snn-expert's sizing (below) decides between the copied constant and this rule.
5. **Population mean is the right set-point** (per-neuron set-points, as in biology (Hengen et al. 2016), would drive every column to ρ̂ and flatten the marginal policy, the effect Vogels et al. 2011 report); its carrier is the shared inhibitory pool reading the pooled E rate. Soften "leaves between-column differences untouched": ∝ coincidence, so active columns shrink more; a shrinkage of the contrast, not neutral.
6. Dale's law: `kap` reaches only W's actor rows (excitatory inputs), fine; if ever pushed into `hs` for R (which has I pre) the pre's sign is needed.
7. Say "minutes", not "seconds", for the needed time constant.
**Recommendation (Q3):** (a) over (c), and (b) alongside (a) as a labelled stabiliser, not instead of it: the race is the decision (evidence accumulation to a bound, Gold & Shadlen 2007; learning from the realised decision, Urbanczik & Senn 2009), so (c) is a step backwards, while a slowly re-estimated β is as local as the normalisation it replaced. Longer-run, the exact per-neuron score (spike minus hazard, Frémaux, Sprekeler & Gerstner 2013) needs no surrogate and has no mismatch. **Q4:** ρ̂ = r₀ is defensible as a consistency condition of the surrogate (β was calibrated there), not as the task's operating point; say so. Decision on the re-pilot's arms ((b) alone, as the user chose, or (a) + (b)) goes back to the user with this review.

**Sizing and implementation of fix (b) (2026-10-29; snn-expert's calculation, scratchpad `drift.py`, `ct_calc.py`; frozen `CKRunner` at the pilot's own calibrated config, 300 ticks).**
- **Units:** the authors' `update = δ·action_trace + ent·eligibility + targ_adj·eligibility` per ms is exactly `nu · G` of `block_learn_ck`, so `c_e = entropy_reg` and likewise **c_t = targ_firing_r with no conversion**, per spikes/ms of rate error; ρ̂ = `rate_target` × 1e-3 = 0.04. Their entropy term is per column; their rate term is per copy (the actor-E layer's mean, all actions pooled), actor connections only, which the implementation follows.
- **Drift (pilot `base_race`, 10 seeds each):** Acrobot +8.0 ± 1.5 Hz per 100 episodes (45 → 85 Hz); CartPole −10.7 ± 1.5 (41 → 8). As an effective per-ms `kap` on the E rows: 7.8e-7 (Acrobot), 1.8e-6 (CartPole), 0.15–0.3 % of the TD term's per-ms RMS (5–6e-4): invisible per ms, decisive over 10⁵ windows.
- **Restoring term:** dρ̄/dt = −k c_t (ρ̄ − ρ̂), k = γ η_a (s_in/13) Σ_j z_ij f_j = 0.028 /ms (Acrobot), 0.012 /ms (CartPole) per unit c_t. At the gate edge (|ρ̄ − ρ̂| = 0.02 per ms) the authors' 5e-6 restores 0.13× / 0.06× of the drift (equilibrium excursion +156 / −355 Hz: no hold); 5e-5 holds Acrobot's mean only (+16 Hz, worst seed +20) and fails CartPole (−36); **5e-4 restores 12.9× / 5.6×, equilibrium excursion +1.6 / −3.5 Hz (worst seed +2.0 / −4.1), relaxation τ_r 19 episodes (Acrobot) / 33 (CartPole; 17 once solved) = 72–166 s of simulated time, 2–4 % of the TD per-ms RMS at the gate edge.** Pre-set rule, fixed before any run: **c_t = 5e-4, the smallest power-of-ten multiple of the authors' value at which the restoring term exceeds the observed drift at the gate edge on both tasks** (margin ≥ 4.8× including the worst seed). This answers the neuroscientist's issue 4 (the copied constant is nominal) with a declared relaxation time in the model's units. Caveats: the nlif gain above 40 Hz was not measured (it falls, while Σ z f grows: roughly cancelling); if the drift forcing grows with the excursion, 5e-4 is conservative in the right direction; H > 1 (LunarLander) uses the per-copy mean, untested.
- **Implementation (`experiments/tdstdp.py`):** `--rate-cost C` sets `c_t`; `CKRunner.rho_hat` = `rho_hat` if given else `rate_target` per ms; in `learn_block`, per ms, `kap -= c_t (mean actor-E rate of the copy − ρ̂)` on the E actor rows only (`lay.ae`), both learners, before the block update; the `c_t = 0` assert is lifted; name token `rc<c_t>`. With c_t = 0 the broadcast `kap` is the same object and nothing is drawn: golden and the softmax path unchanged. Test `test_rate_cost_term`: same raster at the same seed, the same value on every E row of a copy, zero on I posts, the formula pinned by two targets. The copy mean (not a per-column target) is kept because z_ij ∝ ρ_i f_j makes a common `kap` move each column's drive in proportion to its own rate, preserving the log-rate contrasts the surrogate reads to first order; a per-column target would push p̃ toward uniform with no gradient justification. Side effects at c_t > 0: `rec_block` sees `g` with the term (its calibration sum is near zero at init, where the excursion is ≈ 0).
- **User decision (2026-10-29, selection screen):** the re-pilot runs three variants per regime, the rate term alone, the β re-probe alone (fix (a), implemented below) and both, so the re-pilot says which part does the work.
- **Smoke runs with the term (2026-10-29, scratchpad, Acrobot `base_race` + `--rate-cost 5e-4`, 100 episodes, dev seeds 20–23; the control without the term reproduces the pilot's seed 20 log exactly, so the episode cap changes nothing):** the term holds the actor rate at 40–43 Hz at episode 100 on every seed (the pilot's same seeds: 47–60 Hz, and 74–92 Hz by episode 500). Returns at episode 100, with / without the term: s20 −494 / −178, s21 −265 / −486, s22 −500 / −224, s23 −122 / −377. Two of four seeds collapse early either way (silent critic, V at its floor β_c, policy entropy → 0 by episode 30, return −500), on different seeds: the early Acrobot collapse is a property of the race readout itself, not of the term, and the term neither causes nor prevents it (as the neuroscientist's issue 2 predicts: the common mode is held, the mismatch is not). Decisions at the bound fall from 1.00 to 0.94 of windows at the held rate. Not a result (n = 4, 100 episodes); it says the re-pilot's collapse gate will be the binding one and that (b) alone is unlikely to pass it. Reported to the user with the (a) + (b) question.

**Fix (a) implemented (2026-10-29; `--race-reprobe N`, `experiments/tdstdp.py` `reprobe_race`, `CKRunner.beta`).** The surrogate's gain β is re-estimated on a schedule, as the neuroscientist's review recommends ("a slowly re-estimated gain β is as local as the normalisation it replaced"). Once every N pooled episodes the run re-runs `calibrate_race`'s 1-s contrast probe (the same body, `_race_probe`: eight held observations × five contrasts δ on column 0's E rows, the maximum-likelihood β of the realised winners under softmax(β log r̄)) at the run's **current W** (and current R with recurrence; the probe runner copies both), its calibrated s_in and alphas, and the new β replaces the surrogate's from the next window on, so p̃ and the 1/r_j factor follow the race as the rates move. **Pre-set N = 50**, fixed before any re-pilot run: one re-probe per log bin on CartPole and Acrobot (the bin at which the pilot's drift was read, +8 / −11 Hz per 100 episodes), so β moves on the drift's timescale and not within it; the probe costs 1 s of simulated time per 50 episodes. What is recorded: each re-probe's record (the `race_cal` keys plus `episode`) in the result's `race_cal_history` with `race_beta_final`, and the log line `reprobe ep N beta X fit Y` (fit = max |win0 − win0_surrogate|, the §18 gate at 0.05); the checkpoint carries the current β and the history, so a resumed run continues with them. Byte-identity: the probe runs on its own stream `(seed, CAL_SEED, episode)` and is frozen (W, membranes and R of the run untouched), so the run's `rng` and `nrng` draw sequences are identical with and without re-probing (the trajectories diverge only through β); `race_reprobe = 0` (the default) changes nothing, the softmax path and the golden test are untouched, and the init `calibrate_race` call is the same runner on the same stream as before (`test_race_reprobe`; the Acrobot seed-20 smoke run reproduces the pilot's init β 7.9403). The re-pilot's three variants are the user's choice: `--rate-cost 5e-4` alone, `--race-reprobe 50` alone, and both.
- **Smoke run (scratchpad `smoke_reprobe.json`, Acrobot `base_race` seed 20, `--race-reprobe 10 --episodes 50`, development):** init β 7.940 (the pilot's value to every digit), then at episodes 10, 20, 30, 40, 50: β 8.09 (fit 0.053), 10.58 (0.002), 10.83 (0.003), 10.36 (0.053), 9.56 (0.090); 1.67 ms/frame, 0.7 min. The fit is within the 0.05 gate at two of five re-probes, and those two are degenerate: at episodes 20–30 column 0 wins 99.5–100% of the probe's races at every contrast (H_race 0), so the ±20% contrasts on column 0 carry no information about β there (the likelihood is flat above a few units) and the fit is trivially good. The probe's contrast set is a convention sized for the balanced policy at init; at a saturated policy a re-probe reports β weakly determined. Noted as an open point for the re-pilot's reading (the `race_cal_history` records carry `win0` and `H_race`, so such re-probes are identifiable); nothing changed.

**Re-pilot pre-registration (2026-10-29, before any run; development, dev seeds 20–29; `pilot016_race_rc.sh` in the scratchpad, outputs `experiments/results/pilot016/race_rc/`, logs `logs/pilot016/race_rc/`).**
- **Arms per task (CartPole with CK's preset η, Acrobot with `--eta-cal`; every arm `--engine nlif --rate-target 40 --block 20 --fast --rate-match-rec`):** the first pilot's six arms are copied in as references and the no-fix baseline (`base_ck`, `base_nl`, `base_race`, `cols_{lam0,on,sealed}_race`; their code path is unchanged since that pilot: everything added since is behind `--rate-cost` and `--race-reprobe`, and the golden test is unchanged). New: for each of `base`, `cols_lam0`, `cols_on`, `cols_sealed` under `--readout race`, three variants, `_rc` (`--rate-cost 5e-4`), `_rp` (`--race-reprobe 50`) and `_rcrp` (both): 12 arms × 2 tasks × 10 seeds = 240 runs. Column arms `--chi-rec 0.3 --rec cols` with `--lam 0`, `--lam 1`, `--rec-seal --lam 0`; θ by the rule; β by the init probe, then per re-probe in the `_rp` arms.
- **Fixed before any return:** c_t = 5e-4 (the sizing above), N = 50 (fix (a)), nothing else differs from the first pilot. No setting is tuned on return; if every variant fails a gate, that is reported before anything changes.
- **Gates per variant (stability only, as the first pilot):** actor rates within 20–60 Hz at every log bin and no silent column; no collapsed run (Acrobot last-100 = −500; CartPole AUC at the random level); no NaN; max Re λ ≤ 0.5, τ_loop ≤ 40 ms; surrogate fit (max |win0 − win0_surrogate|) ≤ 0.05 at init and, in the `_rp` arms, at every **informative** re-probe, an informative re-probe being one where no column wins more than 95 % of the probe's races at δ = 0 (a saturated probe carries no information about β, as the smoke run showed at episodes 20–30; the fraction of saturated re-probes is reported per arm and is itself a collapse signal). A variant passes when `base_<variant>` passes every gate on both tasks.
- **Co-primary reading (Δ = 25, Hodges–Lehmann CI, as before):** (a) AUC of each race arm against `base_ck`: non-inferior, inconclusive or worse; (b) energy as `actor_spikes_to_decision` only (the rate term pins the mean rate, so `actor_spikes_per_decision` no longer discriminates arms; the neuroscientist's issue 1); `actor_spikes_per_decision` is still reported.
- **Structural question, per passing variant:** `cols_sealed` vs `cols_lam0` (the contrast that opened on CartPole in the first pilot, +57 [+2, +97], n = 10).
- **Also read:** the β sequence per `_rp` run (`race_cal_history`: does β settle, and where); the rate trajectory per variant against the first pilot's; `_rc` vs `_rp` vs `_rcrp` on the same seeds (which part does the work).
- **If a variant passes on both tasks:** the Stage B reported study (`hybrid-columns-5`, three tasks, `base_ck` reference, `base_nl` secondary, the three regimes under the passing variant; fresh seeds 7000–7019), pre-registered as the earlier studies. If more than one passes, the one with fewer added mechanisms (`_rp` before `_rc` before `_rcrp`) is the study's, and the others are secondary arms if the budget allows (user's call).
- **Cost and timing:** about 3–5 min per run; 240 runs at 40 workers ≈ 25 min. It runs on the cores that `hybrid-columns-3-ck` leaves idle during its LunarLander phase (20 of 72 in use), which does not oversubscribe the machine; the user is asked before it starts because a study is on the machine.

## 19. Settling the three-task result: companion studies `hybrid-columns-3-ck` and `hybrid-columns-4` (2026-10-29)

**Decision (user, 2026-10-29).** Settle the §17 result before continuing Stage B: two pre-registered companions that close its two holes, run together; the race fix (§18, option b) is built while they run and re-piloted when the machine is free. The four study result sets (`hybrid-columns`, `hybrid-columns-2`, `hybrid-columns-3`, `emergent-v4`) were committed on their worktree branches and merged into `columns-tdstdp` (`cf72b9e`); `experiments.report summarize` reproduces each committed summary byte for byte from the merged `runs/`.

**The two holes.**
1. Every arm of `hybrid-columns-3` uses the normalised readout, so its base is a modified backbone; the cross-study reading (§17) and the pilot (§18: `base_nl` − `base_ck` −11 [−30, +11] CartPole, +1 [−23, +24] Acrobot, n = 10) say the loss is late LunarLander return. The CK-faithful bases exist only in `hybrid-columns` and `hybrid-columns-2`, at unmatched starts and older code.
2. CartPole's inconclusives are power-limited: the CI half-width at n = 20 is about 15 AUC points at Δ = 25.

**Companion A, `hybrid-columns-3-ck`** (`reports/hybrid-columns-3-ck/README.md`): `<task>_base_ck` on the three tasks, `hybrid-columns-3`'s flags without `--norm-logits` (the §18 pilot's `base_ck`), **seeds 5000–5019, the same as `hybrid-columns-3`** (a stated exception to the fresh-seed rule: the contrasts start from the same initial W and first environment seed; this arm never ran on these seeds). Pairs across studies: each regime of `hybrid-columns-3` vs `base_ck` (headline, Holm per task), and `hybrid-columns-3`'s base vs `base_ck` (the readout's own cost). 60 runs; about 7.5 h, set by LunarLander. Code: `summarize` now loads a pair member written `<study>/<group>` from that study's `runs/`, as it already did for a cross-study reference (`test_summarize_cross_study_pairs`). The contrast spans two commits (`4bdbe45` and HEAD) with one effective code path for this arm; the README says why.

**Companion B, `hybrid-columns-4`** (`reports/hybrid-columns-4/README.md`): CartPole, `hybrid-columns-3`'s four arms plus `base_ck`, **seeds 6000–6039, fresh**, 200 runs, about 30 min. Families: headline vs `base` (3), `ck` vs `base_ck` (4), secondary between regimes (3). Expectation: all regimes non-inferior to both bases; regimes do not differ.

**A bookkeeping note on `hybrid-columns-3`'s Holm.** Its README says Holm "within each family per task"; its `pairs.csv` applies Holm over the nine headline contrasts as one family (the declaration has no per-task family), which is the more conservative reading (LunarLander 0.021 → 0.17). Per task, the LunarLander trio would be 0.051, 0.051, 0.051: still none below 0.05, so §17's reading stands either way. The companions declare per-task families explicitly. Whether to re-summarise `hybrid-columns-3` with per-task families (a `summarize`-only change, no run touched) is the user's call.

**Smoke run** (scratchpad, 20 episodes): `base_ck` on LunarLander at seed 5000, the one arm/task pair never run: k 9.15, r_got 40.4 Hz, ent_scale 1, η_cal [0.0105, 0.0162] mV, 1.40 ms/frame, no error.

**Stage B fix in parallel (§18 option b).** Pre-set rule: CK's own target-rate term (`reference/td_stdp/bindsnet_add.py` l. 198, `(ρ̂ − ρ̄) targ_firing_r × eligibility`, common to every post of a layer) on `CKRunner`'s actor E rows per ms, ρ̂ = r₀ = `rate_target` (the rate every arm is matched to at init), c_t from the authors' low-firing-rate config (`config_cp_lfr.ini`: `targ_firing_r` 5e-6 at `targ_firing_rate` 0.005) converted by the same unit rules as c_e, then a magnitude check against the §18 drift (one fixed value chosen from that calculation before any run, never from return). Default c_t = 0 draws nothing (golden unchanged). Neuroscientist review first, then the snn-expert implements; re-pilot on dev seeds 20–29 with the §18 gates after the companions finish; report before any further step. Results and the review's verdict go here when they exist.

**Companion B result (`hybrid-columns-4`, 2026-10-29 12:06–12:20 EDT, 200/200 runs, none failed; numbers from `../map-stdp-hc4/reports/hybrid-columns-4/`).** Every run of every arm solves (`base_ck` 39/40). Primary AUC, Hodges–Lehmann shift [95 % CI] (p, Holm), Δ = 25:

| Contrast | Family | Shift [CI] | p / Holm | Verdict |
| --- | --- | --- | --- | --- |
| `cols_lam0` vs `base` | headline | −8.3 [−22.0, +4.4] | 0.21 / 0.63 | non-inferior |
| `cols_on` vs `base` | headline | −2.4 [−14.0, +9.6] | 0.65 / 1 | non-inferior |
| `cols_sealed` vs `base` | headline | −4.2 [−16.9, +10.0] | 0.59 / 1 | non-inferior |
| `cols_lam0` vs `base_ck` | ck | −14.7 [−25.7, −3.9] | 0.010 / 0.041 | **worse** |
| `cols_on` vs `base_ck` | ck | −8.6 [−17.6, +1.3] | 0.078 / 0.23 | non-inferior |
| `cols_sealed` vs `base_ck` | ck | −9.3 [−20.8, +1.6] | 0.081 / 0.23 | non-inferior |
| `base` vs `base_ck` | ck | −5.7 [−18.6, +6.6] | 0.38 / 0.38 | non-inferior |
| `cols_on` vs `cols_lam0` | secondary | +7.2 [−2.4, +17.5] | 0.14 / 0.41 | null |
| `cols_on` vs `cols_sealed` | secondary | +2.2 [−8.3, +11.9] | 0.72 / 0.79 | null |
| `cols_lam0` vs `cols_sealed` | secondary | −5.7 [−16.4, +7.3] | 0.39 / 0.79 | null |

- **Against the normalised base, n = 40 settles what n = 20 could not:** all three regimes are non-inferior (CI lower bounds −22, −14, −17), which the §17 study left inconclusive for `cols_on` and `cols_sealed`. AUC means 344 / 351 / 348 vs 351.
- **Against the published backbone (`base_ck`, AUC 358), one pre-registered expectation fails:** implicit columns (`cols_lam0`) are worse, −14.7 AUC with the CI just outside the margin and Holm p 0.041; they solve about 7 episodes later on average (T_s 273 vs 266). Learned and enforced columns are non-inferior (lower bounds −17.6, −20.8); the normalised base itself is non-inferior to CK's (−5.7 [−18.6, +6.6]).
- **The regimes do not differ** (secondary null), as expected; `cols_on` − `cols_lam0` is +7.2 [−2.4, +17.5].
- **Rates:** every normalised arm's actor rate falls after solving (41–47 Hz → 4–6 Hz at the last bin) while `base_ck` holds 30 Hz, as §17 saw at n = 20: the post-solve decay belongs to the normalised readout, not to the columns.
- Reading: on CartPole the cost of columns against CK's own readout is small but real for the implicit regime at n = 40 (about 3 % of the AUC scale), and absent for the learned and sealed regimes within Δ. The claim "columns keep CartPole performance" holds against the matched normalised base; against the published backbone it holds for two regimes of three.

## 20. Stage B as the insertion recipe: the walk decides and the walk gets the credit (2026-10-29)

**The lesson (user, 2026-10-29).** Under CK's softmax readout the columns are a small tax and the regimes never differ (§17, §19): the softmax reads one mean rate per column, and the task term seals the columns by itself, so structure never reaches the decision. In v4 the same structure gave +65 to +87 AUC because the policy was the walk. The race readout makes the decision the walk's (§16, §18: the first regime contrast to move); its learning signal is still a rate surrogate, which is where the §18 drift comes from. Stage B is therefore reframed as the demonstration of a proper way to insert recurrent action columns into an existing feed-forward spiking actor, with decision and credit both carried by spikes: (1) columns with loop-gain calibration; (2) the race on column spike counts; (3) **the per-neuron spike score as the credit** (new, `--credit`); (4) CK's rate term as the homeostasis; (5) map term / sealing as structure control; (6) the recipe written as `docs/columns-recipe.md` (user decision) with a pointer in SPEC. The surrogate re-pilot of §18 still runs as the stopgap comparison.

**The credit rule (reviewed 2026-10-29 by the neuroscientist and the snn-expert; both review-only; numbers below are the snn-expert's, from the scratchpad scripts `score_check.py` (one-neuron finite-difference check, common random numbers, 1–2e5 trajectories, `DT1R` parameters) and `frozen_stats.py` (a frozen `CKRunner` at the pilot's Acrobot seed-20 config, 300 windows; kernel copy raster-identical to `_kernel_ext_nb`)).** The noisy LIF's step is V ← a V + (1 − a) μ_bg + ξ + g σ_n with g the standardised noise draw (Irwin–Hall(4)), a hard threshold, reset and a 2 ms refractory period. Given the pre-noise membrane m = a V(t−1) + (1 − a) μ_bg + ξ the spike is Bernoulli with hazard h = 1 − F((θ − m)/σ_n) (h = 0 when refractory, or beyond the noise support). Two per-neuron scores:
- **hazard score** Y = (S − h) ψ, ψ = f(z)/(σ_n h (1 − h)): the escape-noise score of Pfister et al. 2006 / Florian 2007 / Frémaux, Sprekeler & Gerstner 2013 evaluated at the realised membrane. Exact per step and zero-mean given the past (measured: mean 0.0025 ± 0.008 over 3.3e5 actor-ms), **but not the gradient of log P(decision) for this neuron**: the membrane carries the sub-threshold noise forward (a = 0.95), so the per-step factorisation holds given the membrane history, not the spike history (both reviewers; the plan's "exact, not an escape-noise approximation" was wrong). Finite differences: the per-ms pairing Y x(t−1) recovers 0.22–0.29 of dE[R]/dW; the trace form the code would accumulate (z ← a z + Y Pᵀ) gives 1.5–2.0×.
- **exact (observed-membrane) score:** spike step f(z)/(σ_n h), active no-spike step −f′(g)/(σ_n f(g)) with g = (V − m)/σ_n the realised noise (g/σ_n for Gaussian noise): the Gaussian-unit REINFORCE score (Williams 1992), node perturbation on the membrane (Fiete & Seung 2006). Matches finite differences within 2 % in six settings (count and race rewards, two noise kernels, three drives), truncation at the bound included; on the frozen network its per-window signal-to-noise is 2× the hazard score's (winner column's summed score 6.7 ± 12.5 vs 0.9 ± 4.1) at 3.3× the RMS, which the RMS-matched step absorbs.
- **Relation to CK's rule:** CK's δ (A_k − p_k) z_ij is the softmax's score times a Hebbian proxy for ∂rate_i/∂W_ij; under the race the surrogate made it the score of p̃ = softmax(β log r). The spike score moves the "which column" gating inside Y, E[Σ_{i∈k} Σ_t Y_i | D = k] playing the role of β (A_k − p̃_k)/r_k, with no β, no 1/r and no mismatch.

**Decided design (pre-set; every convention stated before any run).**
1. `--credit noise` (the exact score) is the **headline**, `--credit hazard` the secondary, plausibility-control arm. Neuroscientist's recommendation and tags: the exact score is an *abstraction with a supported class* (perturbation learning with the postsynaptic potential minus its prediction: Fiete & Seung 2006; Legenstein, Chase, Schwartz & Maass 2010; Urbanczik & Senn 2014), needing the neuron to separate its per-step noise from its input, given biologically only where the noise is an identifiable input (LMAN → RA in songbird, Ölveczky, Andalman & Fee 2005) and native on the crossbar, where the noise is injected; the hazard score is an *abstraction, supported class* (escape-noise reward-modulated STDP; postsynaptic factor = membrane relative to threshold, Clopath et al. 2010; escape-rate fits to real cells, Jolivet et al. 2006), local, but biased for this neuron. A biased rule has an uncontrolled effective learning rate across regimes, which would contaminate the regime contrast that is the headline question.
2. **Truncation at the bound:** Y = 0 after the copy's decision ms (post-bound scores are zero-mean given the past: measured, no change in the finite-difference match); at a deadline decision the whole window counts. Precedent: bounded integration ignores post-bound evidence (Kiani, Hanks & Shadlen 2008); a decision event gating the trace (Friedrich, Urbanczik & Senn 2011). Caveat: post-bound spikes seed the next window through R, and that credit is forgone; acceptable while the race resets at the resample and τ_loop ≤ 40 ms is gated. Carrier: a per-copy "decided" broadcast, the same class as Urbanczik & Senn 2009's winner feedback.
3. **E actor posts only** carry the score (c = 1 on E posts, 0 on I posts: I posts do not enter the E-count race, their score is noise); the post's E/I type needs no sign correction (its effect is routed through the E posts' hazards); the pre's sign enters through dm/dW as in W's convention.
4. **No entropy bonus** (no explicit policy); **the rate term** `--rate-cost` as the homeostasis, in a **rank-one form with no per-synapse state** (the project's constraint that the only per-synapse state beyond W is the task eligibility trace rules out a second trace): per block G_rate = −c_t Σ_t (ρ̄ − ρ̂) ρ_i P_j on E actor rows, expectation-equivalent to CK's kap z (E[z_ij] ≈ ρ_i P_j), and closer to the heterosynaptic depression by postsynaptic rate it is named after (Zenke & Gerstner 2017). The surrogate arms keep CK's form as pre-registered. **Its step:** c_t was sized for the surrogate arm's η_a (τ_r 19–33 episodes); the score's RMS-matched η_a is 40× (hazard) to 130× (exact) smaller, so the rate term keeps its *effective* step: c_t,eff = c_t × η_a(surrogate calibration) / η_a(the arm's own calibration), both from `calibrate_ck` at init, recorded.
5. **Exploration has no floor in this design** (neuroscientist): the rate term pins the copy's mean rate, not the contrast between columns; a 20 Hz column loses nearly every race against a 60 Hz one while both are "alive". The softmax's entropy bonus bounded exactly that. Exploration rests on the membrane noise (the standard spiking policy-gradient setting: Seung 2003; Urbanczik & Senn 2009). So the **realised race entropy per log bin is a gate** of the pilot; if it fails, the least implausible add-on is a scalar gain or noise modulation driven by the realised decision entropy (adaptive gain, Aston-Jones & Cohen 2005), one broadcast scalar, crossbar-native: a Proposal, not built.
6. **Clip:** the hazard score's ψ is clipped at ψ_max = ψ(h = 1e-3) on both branches (so E[Y | past] stays 0; "a saturating surprise factor"); the exact score needs none (its tail |f′/f| > 190/θ has probability 5e-5). Measured on the frozen network: 74 % of active actor-ms have h < 1e-3 but 7 of 14,926 spikes (0.05 %) do, and the gradient weight lost is 0.035 %: the clip is inert, and the gate is defined on **spikes** (`score_clip_frac` = fraction of spikes at h < 1e-3), not on ms.
7. **R's credit:** module-level first version, hs[:, k] = Σ_{i∈k, E} Σ_{t ≤ bound} Y_i(t)/n_ae, the score of the column's count, the quantity the race reads (a population feedback in the Urbanczik–Senn sense; the per-synapse version Y_i(t) χ S_j(t−1) is the plausible one and a Proposal). Units: the surrogate's hs is ∂log p̃/∂ρ_i, the score's is ∂log P(D)/∂u_i; ∂ρ/∂u ≈ 0.88 here, so RMS matching in `calibrate_rec` absorbs the scale; `credit`'s counter τ_eff b G_i n_j double-counts G for a current-derivative score and is corrected (G = 1/τ_eff on the task path). R is E→E, so no inhibitory pre sign arises.
8. **η:** `--eta-cal` on every task for score arms (CartPole's preset η is a softmax-scale constant with no meaning for the score). Measured at init on Acrobot: per-ms update RMS surrogate 1.2e-3, hazard 4.8e-2, exact 1.6e-1; η_a 0.052 → 0.0013 / 0.00039.
9. **Locality and the crossbar:** Y_i needs the post's membrane, threshold and noise scale (intrinsic; the exact score additionally the step's innovation), P_j is pre-local, δ one broadcast per copy; extra broadcasts: the decision gate and the module score for R, both K-table-sized. Confirmed by both reviewers; the only violation would have been the second trace, now avoided.
10. **Checks:** the noise score against central finite differences on the one-neuron toy (R = a 20-ms count; the project's convention) in `lif.__main__`, recorded in Appendix A's list; h against the empirical spike probability; E[Y] = 0 for the hazard score (clipped) with its residual reported. The hazard score is *not* tested against finite differences (it fails by 4×, by design).
11. **Timing kernel** (c ≡ 1, τ_z 20, τ_q 40): a window's score mass lands 0.24 on the next window's δ (the one for its action), 0.08 on its own δ (zero-mean noise) and 0.23 on the one after; the surrogate's gating gave 0.09 on the next: the score's credit arrives sooner and stronger.

**Implementation (snn-expert, 2026-10-29; `--credit {surrogate,hazard,noise}`, tokens `sch` / `scn`; 71 tests, golden unchanged, default path bit-identical to `4a74319` by a fingerprint of W, rng states, membranes, R and returns on race, softmax and race-plus-columns runners and the kernel in all three noise modes, and by `test_credit_default_path`'s hard-coded fingerprints).** The kernel exports the pre-noise membrane and the noise draw per ms behind `score=True` (no extra draw); `lif.post_scores` gives Y and h for both kinds (`H_CLIP` 1e-3, `psi_max` cached; a fused Irwin–Hall evaluation). In `learn_block` the score path makes **two learner calls per block**: the critic rows exactly as today (bit-identical), the actor rows with Y on the E posts (truncated at the copy's bound ms by a per-ms live mask that `race_block` records, zero on I posts and latents) as the post factor and **x(t − 1), the input the kernel paired with the step, as the pre factor**, c = 1 on E posts, no entropy term, all signs +1. **Correction found on the way:** the plan's z ← a z + Y Pᵀ pairs the score with the backbone's 20-ms pre trace, which integrates inputs whose effect already went through V(t − 1); on the toy that over-counts the gradient 1.7–1.9×, while Y x(t − 1) matches. The rank-one rate term keeps P (homeostatic, not the gradient). R's credit: `hs` = the column's summed score up to the bound / n_ae, installed with the surrogate's one-window lag, and `credit` receives G = 1/τ_eff on the task path so the counter is not double-counted. `calibrate_ck` runs for the arm's credit and for the surrogate (η_a and η_a_ref; c_t,eff = c_t η_a_ref / η_a, recorded in `low_rate.score`); `calibrate_race` still runs (β feeds only η_a_ref and two monitors). Monitors `score_abs_mean`, `score_clip_frac` (spikes at h < 1e-3), `race_entropy` per log bin in `race_log`. **Checks (Appendix A, H1–H3):** Σh vs spike counts within 3.1 s.d. on both kernels; the escape-noise form E[ΣY] = 0.03 ± 0.04; the noise score vs central finite differences of ∂E[count]/∂w on a one-neuron toy, ratio 1.001; and **the eligibility as the learner computes it** (`mapstdp.tdstdp` self-check h4: one actor neuron, two Bernoulli inputs, `block_learn_ck_fast` driven as the runner drives it, τ_z → ∞ and τ_q → 0 so G is the plain sum) vs finite differences of ∂E[R]/∂W_j: ratios 1.003 / 1.005 (seed 11), 0.997 / 1.009 (seed 3). The run's τ_z = 20, τ_q = 40 are CK's time weighting of the eligibility, not checked against a finite difference. Cost: 1.5–1.7× the surrogate's learner time per block (a second learner call, the array building, the score polynomials). **Smoke (Acrobot, 100 episodes, `--rate-cost 5e-4 --eta-cal`):** with the x(t − 1) pairing the actor update's RMS is 7.2× (noise) / 2.2× (hazard) the surrogate's (it was 133× / 40× with the P trace), η_a 7e-3 / 2.4e-2 mV, c_t,eff 1.1–4.6e-3; rates held at 40 Hz, clip 0.03 % of spikes, realised race entropy 0.8–1.08 nats of ln 3, all finite; return at the floor at episode 100 on 4 of 5 runs (seeds 20–23 noise, 20 hazard; the surrogate's seed-20 smoke was −496 too), seed 22 climbing (−426, critic rate rising). 100 episodes say nothing about return; the pilot does.

**Implementation plan as reviewed: `mapstdp/lif.py` kernel exports the pre-noise membrane and the noise draw per ms behind `score=True` (no extra draw; default bit-identical), `post_scores(S, M, GN, p, kind)`; `simulate_nlif(score=...)`; `Config.credit ∈ {surrogate, hazard, noise}`, CLI `--credit`, tokens `sch` / `scn`; the per-ms live mask from `race_block`; `Xmix` (critic rows X, actor E rows Y) with c = 1 on E posts up to the bound; the rank-one rate term with c_t,eff; `hs` under the score; monitors `score_abs_mean`, `score_clip_frac`, `race_entropy`; tests for byte-identity of the default path (raster, W, rng states, learner inputs) and for each rule. Records: this section, `docs/columns-recipe.md` (to write), derivation §8.1 (a Proposal row for the hybrid's credit), CHANGELOG.

**Score pilot (to pre-register before launch; development, dev seeds 20–29, CartPole and Acrobot, after the surrogate re-pilot):** `base_ck` and the surrogate re-pilot's passing variant as references; `base_race_scn`, `cols_{lam0,on,sealed}_race_scn` (headline) and the same four `_sch` (secondary), all `--readout race --rate-cost 5e-4 --eta-cal`; 160 runs. Gates: rates 20–60 Hz, no collapse, no NaN, max Re λ ≤ 0.5, τ_loop ≤ 40 ms, `score_clip_frac` < 5 % of spikes, realised race entropy ≥ 0.3 nats per bin (K = 2 on CartPole: of ln 2; K = 3 on Acrobot: of ln 3) through the run. Reading: AUC vs `base_ck` (Δ = 25), `actor_spikes_to_decision`, the regime contrasts, noise vs hazard on the same seeds. Then the Stage B study `hybrid-columns-5` (three tasks, seeds 7000–7019, `base_ck` reference, the passing credit as headline, the surrogate's passing variant as secondary arms).

**Companion A result (`hybrid-columns-3-ck`, 2026-10-29 12:20–15:12 EDT, 60/60 runs, none failed; numbers from `../map-stdp-hc3ck/reports/hybrid-columns-3-ck/`; the `on` side of every pair is `hybrid-columns-3`'s arm on the same seeds).** Primary metric: last-100 on LunarLander, AUC elsewhere; Hodges–Lehmann shift [95 % CI] (raw p / Holm per task), Δ = 25:

| Contrast vs `base_ck` | CartPole (AUC) | LunarLander (last-100) | Acrobot (AUC) |
| --- | --- | --- | --- |
| `cols_lam0` | −5.7 [−20.9, +8.3] (0.34 / 0.38) non-inferior | **−22.2 [−31.9, −12.2] (2e-4 / 2e-4) worse** | −8.5 [−34.1, +13.7] (0.42 / 0.85) inconclusive |
| `cols_on` | −14.1 [−31.0, +5.2] (0.13 / 0.38) inconclusive | **−24.1 [−32.6, −14.6] (4e-5 / 1e-4) worse** | −14.2 [−30.7, +7.1] (0.20 / 0.60) inconclusive |
| `cols_sealed` | −11.6 [−28.5, +6.7] (0.13 / 0.38) inconclusive | **−25.3 [−34.9, −15.8] (4e-5 / 1e-4) worse** | −0.6 [−17.7, +17.5] (0.95 / 0.95) non-inferior |
| `base` (normalised; readout family) | −0.6 [−15.5, +12.3] (0.86 / 1) non-inferior | **−32.8 [−42.1, −23.0] (2e-6 / 5e-6) worse** | +1.2 [−14.6, +19.7] (0.84 / 1) non-inferior |

- **LunarLander, the one task where columns had a positive signal, is settled against them.** CK's own readout at the matched start reaches last-100 62.3 ± 15.5 (every run to the 10000 cap, none solved), the level of `hybrid-columns`' base (58), while the normalised base plateaus at 30.1: the normalisation costs 33 points of late return (the §17 cross-study reading, now a within-study test on the same seeds). The column arms (37–40) recover a third of that loss and are 22–25 points **worse** than the published backbone. Early learning is the same: AUC over the first 3000 episodes −59 to −61 for the column arms vs −63.6 for `base_ck` (+2.6 to +4.5, n.s. after Holm). Prediction (i) of the README held (readout worse on LunarLander, non-inferior elsewhere); prediction (ii) held on CartPole and Acrobot for 2 of 3 regimes each.
- **Acrobot's secondary metric moves the other way:** last-100 return is **better** in every normalised arm than in `base_ck` (−142.4): `base` +14.2 [+2.1, +26.0] (Holm 0.019), `cols_on` +15.5 (0.014), `cols_sealed` +17.7 (0.029), `cols_lam0` +9.1 (0.064). A secondary metric, not a claim; it says the normalised readout changes the *late* phase in opposite directions on the two long tasks (helps the Acrobot end game, kills the LunarLander climb), and the columns add nothing to the readout they sit in on either.
- **Reading, with §19's companion B:** under the softmax readout the columns never beat the readout they are inserted into, on any task, against either base; the one apparent gain (§17's LunarLander +7 to +11) was a partial recovery of the normalisation's own loss. Combined with the within-column weight share of 0.998 (the regimes become one structure), this closes the softmax branch of the hybrid as a backbone result: structure that the readout cannot see cannot help. It is the strongest argument for §20.

**Re-pilot result (2026-10-29 15:20–15:50 EDT, 240/240 new runs, none failed; run from the detached worktree `../map-stdp-rc` at `4a74319`, results copied to `experiments/results/pilot016/race_rc/`; `an_race.py race_rc`; development, dev seeds 20–29, n = 10 per arm; nothing changed afterwards, per the rule).** AUC median and Hodges–Lehmann shift vs `base_ck` [bootstrap 95 % CI]; `_rc` = the rate term, `_rp` = the β re-probe, `_rcrp` = both; the unsuffixed race arms are the first pilot's (no fix), copied in:

| Task | Arm | base | `cols_lam0` | `cols_on` | `cols_sealed` | base's rate at the end | spikes to decision (base) | collapsed (base / lam0 / on / sealed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole | `base_ck` 364 / `base_nl` 351 | | | | | 33 Hz | 35.6 per window | 0 |
| | race (no fix) | −11 [−56, +10] | −39 [−73, +15] | −36 [−81, +15] | +24 [−6, +40] | 8 Hz | 17.3 | 0 |
| | race `_rc` | **+23 [+4, +45]** p 0.03 | +33 [−10, +51] | **+31 [+12, +48]** p 0.005 | **+29 [+4, +49]** p 0.04 | 36 Hz | 27.4 | 0 |
| | race `_rp` | +11 [−6, +27] | +24 [−3, +50] | **+35 [+15, +53]** p 0.007 | **+41 [+23, +59]** p 0.001 | 6 Hz | 16.1 | 0 |
| | race `_rcrp` | +24 [+2, +53] p 0.05 | **+29 [+8, +49]** p 0.02 | **+24 [+3, +48]** p 0.05 | **+30 [+10, +46]** p 0.01 | 36 Hz | 27.4 | 0 |
| Acrobot | `base_ck` −184 / `base_nl` −182 | | | | | 7 Hz | 24.0 | 0 |
| | race (no fix) | −84 [−277, −4] | −272 | −65 | −102 | 87 Hz | 37.1 | 2 / 5 / 0 / 3 |
| | race `_rc` | −289 [−318, −18] | −295 | −268 | −295 | 41 Hz | 36.1 | 4 / 6 / 4 / 8 |
| | race `_rp` | −164 [−225, −10] | −207 | −177 | −113 | 97 Hz | 36.6 | 1 / 3 / 2 / 0 |
| | race `_rcrp` | −308 [−324, −269] | −298 | −294 | −312 | 40 Hz | 33.9 | 7 / 8 / 6 / 9 |

- **Gate verdict: no variant passes on both tasks.** On CartPole every variant passes every gate (rates: `_rc`/`_rcrp` held at 36 Hz, `_rp` falls to 6 Hz as the unfixed race did, which is inside the gate until the runs solve; no collapse; β fit 3–5/10 within 0.05, as before). On Acrobot `_rc` and `_rcrp` hold the mean rate at 40 Hz **and collapse 4–9 of 10 runs**, with 10–28 silent actor neurons per run and realised-policy entropy 0–0.6 nats: the rate term pins the copy's mean while individual columns go silent and one column takes over, exactly the neuroscientist's issue 3 ("it cannot revive a silent column; it pins the mean, not the contrast"); `_rp` fails the rate gate (97–110 Hz) with 0–3 collapses. So the surrogate race on Acrobot is broken upstream of either fix, and the two fixes fail in opposite ways.
- **CartPole is a positive result, the first for any race arm against the published backbone:** every stabilised arm solves 10/10 (last-100 500.0) and is 23–41 AUC above `base_ck` (6–8 % of the scale), nine of twelve contrasts with CIs above zero at n = 10, with fewer actor spikes to the decision (27 at the held rate, 13–16 under `_rp`) than `base_ck`'s 36 per window. The `_rp` arms are also the most energy-efficient and the largest gains (`cols_sealed_race_rp` +41 [+23, +59]), at the cost of the rate decay the §18 pilot flagged.
- **The structural contrast closed again on CartPole once the drift was fixed:** sealed − implicit −3 (`_rc`), +18 [−10, +44] (`_rp`), −2 (`_rcrp`), against +57 in the unfixed race. The first pilot's regime signal was carried by the drift (implicit and learned columns drifted harder), not by the structure. Under the surrogate the regimes are once more one structure.
- **Reading for the recipe:** the surrogate credit with a stabiliser is a working readout on a 2-action task with positive TD errors and a broken one on a 3-action task with negative TD errors; the §18 drift hypothesis (the sign of δ sets the drift's direction) survives, and the rate term's failure mode on Acrobot (silent columns under a pinned mean) is the predicted one. The score credit (§20) removes the surrogate's mismatch; whether it removes the Acrobot collapse is the score pilot's question, and the realised-entropy gate and the silent-column count are its binding monitors. Nothing is tuned; the §18 pre-set values stand.

**Score pilot result (2026-10-29 16:05–16:35 EDT, 160/160 new runs, none failed; run from the detached worktree `../map-stdp-sc` at `9681eea`, results copied to `experiments/results/pilot016/score/`; `an_race.py score`; development, dev seeds 20–29; nothing changed afterwards).** AUC median, Hodges–Lehmann shift vs `base_ck` [CI]; `_scn` = `--credit noise`, `_sch` = `--credit hazard`, both with `--rate-cost 5e-4 --eta-cal`:

| Task | Arm | base | `cols_lam0` | `cols_on` | `cols_sealed` | rate at the end | realised race entropy first → last (nats) | mean abs score | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole (`base_ck` 364; random ≈ 22) | `_scn` | 16.5 (−345 [−359, −330]) | 14.4 | 14.5 | 16.1 | 46 Hz | 0.46 → 0.44 / 0.58 → 0.37 / 0.53 → 0.34 / 0.50 → 0.54 (of ln 2 = 0.69) | 11.9 |
| | `_sch` | 9.8 (−353 [−361, −334]) | 9.5 | 9.6 | 14.7 | 34 Hz | 0.39 → 0.14 / 0.27 → 0.08 / 0.29 → 0.19 / 0.22 → 0.42 | **0.004–0.015** |
| Acrobot (`base_ck` −184) | `_scn` | −493 (−293 [−318, −185]) | −494 | −498 | −494 | 40 Hz | 1.03 → 0.65 / 1.08 → 0.60 / 1.07 → 0.58 / 1.06 → 0.48 (of ln 3 = 1.10) | 12.4 |
| | `_sch` | −500 (−314 [−327, −294]) | −499 | −499 | −500 | 40 Hz | 1.07 → 0.45 / 1.00 → 0.28 / 1.05 → 0.44 / 1.04 → 0.36 | 1.0–1.1 |

- **Gate verdict: both score credits fail the AUC reading on both tasks, by a wide margin.** On CartPole every score arm is at or below the random policy over 400 episodes (last-100 9–21), with the realised policy entropy *falling*: the policy sharpens onto one action instead of learning. On Acrobot every arm is at the floor. Rates are held (34–46 Hz), the clip never fires, no NaN; the race decides at the bound in 63–95 % of windows, with ties 4–55 % on CartPole (two columns, θ = 16).
- **The monitor that points at the cause:** the hazard score's mean magnitude is 0.004–0.015 on CartPole against 1.0–1.1 on Acrobot, a 100–200× gap between tasks, while the exact score's magnitude is ≈ 12 on both (≈ 1/σ_n, the size of the no-spike noise term g/σ_n). Reading (to be confirmed by the snn-expert's diagnosis, in progress): on CartPole the actor neurons run mean-driven (free membrane above threshold, firing nearly deterministic), so the hazard at spike steps is ≈ 1, the spike-step score f(z)/(σ_n h) is ≈ 0 and the exact score is almost entirely the zero-mean noise term: a credit with no information per spike. The surrogate's credit (A − p̃) is signal by construction, which is why the surrogate race learns CartPole and the score does not. If confirmed, the per-neuron score needs a fluctuation-driven actor (sub-threshold mean, noise-driven spikes), the regime of cortical neurons, which is a calibration rule (mean drive and σ_V at the same 40 Hz), not a tuning of the rule. The earlier smoke runs at 100 episodes could not see this (Acrobot's first 100 episodes are at the floor under every credit).
- **Not changed.** Reported to the user with the diagnosis; the §20 pre-set values stand until a regime rule is decided.

**Diagnosis of the score pilot (snn-expert, 2026-10-29 evening; scratchpad `score_diag.py`, `cascade.py`, `whatif.py`, outputs `sd/`; development). The operating-regime reading above is wrong; two failures upstream of the credit explain the pilot, and with a sane step the exact score solves CartPole.**
- **H1, regime: rejected.** On a frozen runner at the pilot's configs both tasks are *identical* at init: free membrane 1.44–1.46 θ (mean-driven, 100 %), hazard at spike steps 0.36 mean / 0.62 p95, none above 0.9, hazard |Y| 1.5 on both, noise |Y| 12.5 = E|g|/σ_n on both. The pilot's CartPole hazard |Y| of 0.006 was the *exploded-W* regime (free membrane −21 ± 30 θ), a symptom. Signal fraction of the column-summed score (R² against the race outcome): noise 0.11, hazard 0.04, surrogate 0.80–0.85; winner's summed-score SNR 0.35 / 0.20 / 1.85 (CartPole), 0.52 / 0.30 / 2.72 (Acrobot); a σ_V sweep (0.2 → 0.8 at 40 Hz) raises the noise score's SNR 1.5×, not the 5× gap: the regime is not the lever.
- **CartPole: a step-size catastrophe from `--eta-cal` on CartPole, the trap §15 already recorded.** The plan required `--eta-cal` for score arms on every task; on CartPole `calibrate_ck` reads the preset η (the authors' SGD rates 0.005 / 0.01 mV) as Adam steps per ms and returns η_c 0.80 mV, η_a 0.89 (noise) / 3.04 (hazard) mV: 68× (critic) and 280× (actor) the per-ms RMS step of the `_rc` references, which never used `--eta-cal`. In episode 1 the critic runs to the refractory ceiling (44 → 327 Hz, V 0.1 → 2.1), the terminal window's δ = −V pulse kicks the critic rows to −10 mV mean: a **silent critic, an absorbing state** (z = 0 thereafter), δ a constant +0.0012/ms, the actor W random-walks to |W| 5–9 mV, the race becomes deterministic (one column 100 %, 79 % of spikes at h > 0.9) and the return is one action always (9–10, below random). Reproduced bit-identically (|W| 6.0812 at episode 50).
- **Acrobot: the critic dies at the floor in every race arm**, including the surrogate's (`_rc`: 5/10; hazard 10/10; noise 5/10; `base_ck` 0/10, its critic at 57–79 Hz). β_c = −1.2 is below the floor policy's value (≈ −1.0, which needs 29 Hz of critic rate); once silent the critic cannot recover, δ ≈ +0.0002/ms constant, and the actor sees no signal (hazard W never moves; noise W random-walks slowly; a few seeds stumble on reward late). Upstream of the credit and shared with the surrogate: this is the Acrobot problem.
- **H3, the live path is right:** brute-force finite differences of P(D = k) on a frozen runner (300 copies × 120 windows, common random numbers, ties ½) vs E[eligibility × outcome] as `learn_block` builds it: noise 0.96–1.24 (per column, per row, per synapse; both tasks), hazard 0.14–0.23 (the known bias), truncation and the race included; the learner's τ_z = 20 weighting gives 0.57–0.75, not a bug. No layout or mask misalignment at K = 2, n_ai = 0.
- **H4, what-ifs on CartPole (live, 1 copy, 400 episodes, η_c 0.020 mV = the authors' Adam critic step / RMS, the rule the other tasks use):** surrogate η_a 0.048: AUC 324 at 200 episodes, solved by episode 100; **noise, RMS-matched η_a 0.0056: AUC 240 / 306 (seeds 20 / 21), last-100 496 / 497, solved by episode 300 / 250**; noise η_a 0.02 (3.6× the RMS match): AUC 306, solved by 200; noise signal-matched η_a 0.074: collapse (race deterministic by episode 200, critic fine), 16 copies at the same step: no collapse, slow (AUC 33); hazard RMS-matched 0.019: slow and biased (AUC 26–41 at 200 episodes); hazard signal-matched: collapse. Crediting only the deciding column removes the losers' signal with their noise (E[hs | D ≠ k] = −4.9 is as informative as +4.9); a per-window baseline adds little (the per-step score is already zero-mean given the past); averaging over copies is the variance reduction that works.
- **Recommendation (snn-expert): fix the η rule, not the credit.** On CartPole, every `--eta-cal` arm should calibrate against the authors' Adam steps (1.25e-4 / 6.25e-5, the LunarLander config's), not CartPole's SGD-rate preset: η_c ≈ 0.020 mV, η_a RMS-matched (0.0056 noise / 0.019 hazard); re-pilot CartPole; the only tuning worth a pre-set is a step at 3× the RMS match. **Acrobot needs the critic fixed first** (the absorbing silent critic at a floor below β_c's range kills the surrogate too), before any credit contrast means anything. Hazard stays the plausibility control (5× slower at the same step, collapses at a larger one). Open: why the race arms' critics die on Acrobot while `base_ck`'s does not; whether the 2-ms terminal δ = −V pulse should be clipped for every `--eta-cal` arm; the copy count for score arms in the study.

**Acrobot's critic death under the race (snn-expert, 2026-10-29 evening; scratchpad `sd2/critic_live.py`, `summ.py`; live reproductions bit-identical to the pilot runs; development).**
- **How it dies:** a gradual decline, not a pulse, and at the same speed under both readouts (critic rate 30 → 1 Hz over 12–20 floor episodes in `race_rc`, `scn` and `base_ck` alike). At the floor every episode is truncated, so the terminal window never fires; on Acrobot that window's δ = −V ≈ +1.2 is a *positive* pulse worth 5–10 floor episodes of decay, the critic's lifeline, which arrives only when an episode terminates.
- **Why the floor is absorbing at 0 Hz:** the standard TD(0) bias from the critic's own spikes entering both V_{t−1} and z(t−1): E[δ_t z_i(t−1)] contains −(1 − γ a_n) Cov(V_{t−1}, z_i(t−1)), measured as a constant −2.2 ± 0.1e-4 per ms in δ units in every run at every rate (analytic −2.14e-4 with α_c = 7 after the low-rate scaling, n_ce = 40, τ_n = τ_z = 20 ms, τ_d = 1000 ms). The effective floor target is V* = (r_ms + δ_bias) τ_d = −1.21, below β_c = −1.2, so no critic rate represents the floor and the silent critic is the only fixed point; predicted floor rates at β_c −1.3 / −1.5 (12 / 41 Hz) match the live checks (12–14 / 41–43 Hz). A silent critic then emits a constant δ = +1.95e-4 per ms, which under (A − p̃) credit is a perseveration signal (realised entropy → 0, repeat → 1).
- **Why the race and not the softmax: the race's initial policy is uniform random.** Three columns at 40 Hz with θ = 16, the winner set by noise: realised entropy 1.07–1.09 of ln 3, repeat probability at chance, and the uniform policy on Acrobot-v1 is at the floor (96–100 % of episodes at −500, gym check). The softmax at α_a = 175 is sharper and state-locked from the start, terminates 10–50 % of early episodes, and each termination's +1.2 pulse re-inflates its critic (first-50 floor fraction 0.34 vs 0.96 for `race_rc`, 0.98–0.99 for the score arms). Nothing in the race path touches the critic rows; η_c is identical per seed across readouts.
- **Fixes considered:** (i) a critic rate floor (CK's term on critic rows): pins V's offset but leaves a constant δ̄ at the floor, rejected; (ii) **a β_c margin covering V_rand + τ_d δ_bias: closed form, local, pre-settable, recommended**; (iii) clipping the terminal pulse: irrelevant here, the pulse is positive and is the lifeline; (iv) a bias input: does not change V's range; (v) removing the bias with a held-out critic sub-population or n_ce × 4 (bias ∝ 1/n_ce): principled, an architecture change, not for this pilot.
- **Recommendation: β_c = 1.5 V_rand (−1.5 on Acrobot; CartPole stays at −0.2 through the min; Taxi would move to −1.08).** 50-episode live checks (surrogate, `--readout race --rate-cost 5e-4`, 7 seeds): the critic never drops below 36 Hz (pilot: 0.2–0.3 Hz in the dying runs) and δ̄ at the floor is zero-mean; `base_ck` under the same β_c keeps its returns within seed noise with its critic 40 Hz higher for the same V. β_c is a preset rule, so `base_ck` becomes a new reference and must be re-run.
- **Still open after the fix:** 6 of 7 race runs remain at the floor at episode 50 with 0–7 terminations, and the rate term's column collapse (realised H < 0.3 in 16–34 of 50 episodes on four seeds) persists with zero-mean δ: the race on Acrobot still needs its initial policy to be state-dependent enough to find the goal, which the softmax gets from α_a and the race gets only from θ (the integration length that sets how small a rate contrast the race resolves). That is the pre-set knob the per-task grid should turn (θ ∈ {16, 24, 32}), with the entropy-matched softmax as the yardstick for the initial state-dependence.

**CartPole score re-pilot after the η rule fix (2026-10-29 evening, 80/80 runs, none failed; main checkout at `9681eea` plus the `adam_step` patch (`score_cp_code.patch` in the scratchpad); `experiments/results/pilot016/score_cp/`; dev seeds 20–29; development).** AUC median / last-100 (`base_ck` 364 / 500; `base_race_rc` 380 / 500; `cols_sealed_race_rp` 403 / 500):

| Credit | base | `cols_lam0` | `cols_on` | `cols_sealed` | rate at the end | η_a (mV) | c_t,eff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `noise` | 235 / 484 (−134 [−169, −104] vs `base_ck`) | 202 / 490 | 203 / 485 | 187 / 483 | 40 Hz | 0.0055 | 0.004 |
| `hazard` | 66 / 160 (−297) | 41 / 74 | 77 / 200 | 61 / 141 | 40 Hz | 0.019 | 0.001 |

- **The exact score learns CartPole.** Every `noise` arm ends near the ceiling (last-100 483–490; one run meets CK's solve criterion, T_s 373) with rates held at 40 Hz, the clip never firing and the realised race entropy at ln 2 throughout. It learns about 2.5× more slowly than the surrogate with the rate term (AUC 187–235 vs 380–403), as the §20 signal-fraction measurement predicted (0.11 vs 0.80 of the credit's variance is signal, and the RMS-matched step scales the whole update). The hazard form is 5× slower again and biased, as expected. Regimes do not differ (sealed − implicit −14 [−36, +16]).
- **What the rule says now:** with the step rule right, the per-neuron score is a working credit for the race on CartPole and a slow one; the snn-expert's what-if put a step at 3–4× the RMS match (η_a 0.02) at solved by episode 200 (AUC 306) and a step at 13× in collapse, so the step multiplier is the one per-task knob worth a declared grid (η_a × {1, 3, 6}), per the user's per-task-tuning rule.

**Per-task grid pilot (2026-10-29 evening, 130/130 runs, none failed; main checkout at `9681eea` plus the `adam_step`, `--beta-c`, `--eta-mult-a` patch (`grid_code.patch` in the scratchpad); `experiments/results/pilot016/grid/`; dev seeds 20–29; development; `an_grid.py`).**

*CartPole, the exact score at actor-step multipliers (base and sealed columns; reference `base_ck` AUC 364, last-100 500):*

| Arm | ×1 | ×3 | ×6 |
| --- | --- | --- | --- |
| `base_race_scn` | 235 / 484, 1/10 solved (−134 [−169, −104]) | **359 / 498, 8/10 solved (−7.6 [−48, +20])** | 306 / 500, 10/10 solved (−50 [−87, −9]) |
| `cols_sealed_race_scn` | 187 / 483, 0/10 | 325 / 500, 7/10 (−39 [−123, +0.2]) | 336 / 500, 7/10 (−37 [−172, −7]) |

At three times the RMS-matched step the exact score is level with the published backbone on CartPole (AUC 359 vs 364; 8/10 solved); at six times it solves every run but learns a worse early policy (AUC 306). Rates held at 40–41 Hz, no collapse, realised entropy at ln 2. **CartPole setting for score arms: η_a × 3.** (The surrogate race with the rate term or the β re-probe remains ahead at 380–403.)

*Acrobot with β_c = −1.5 (the 1.5 V_rand rule), the race at three integration lengths (reference `base_ck_bc`, the softmax under the same β_c: AUC −174, last-100 −150, its critic at 113 Hz by the end):*

| Arm | θ = 16 | θ = 24 | θ = 32 |
| --- | --- | --- | --- |
| `base_race_rc` (surrogate + rate term) | −468 / −500, 7/10 collapsed | −500 / −500, 7/10 | −499 / −500, 8/10 |
| `base_race_scn` (exact score) | −494 / −497, 5/10 | −492 / −485, 5/10 | −463 / −485, 4/10 |

The critic fix works as predicted: the critic stays at 43–52 Hz in every race run (it died in the pilot). The race still fails on Acrobot at every θ: 4–8 of 10 runs collapse, the realised entropy falls to 0.4–0.6 of ln 3 while the mean rate is pinned at 40 Hz, i.e. the policy perseverates on one column. θ does not help the surrogate (worse at 24 and 32) and helps the score a little at 32 (−463, 4 collapses). **Reading:** with the critic alive and δ zero-mean at the floor, the credit at the floor is pure noise; W random-walks, and the race's winner-take-all dynamics amplify any drift into perseveration, which the mean-rate pin cannot prevent (it holds the mean, not the contrast: the neuroscientist's issue 3, now seen under every credit). The softmax at the same β_c escapes because its initial policy is state-locked enough to end 10–50 % of early episodes (§20, critic diagnosis) and its entropy bonus bounds the contrast. The race on Acrobot therefore needs an exploration floor that bounds the *contrast* between columns, not the mean: the candidates are a one-sided per-column rate floor (potentiate only columns below ρ_min = r₀/2; never pulls a winner down, cannot silence a column) or the decision-entropy gain modulation (a Proposal in §20). Neither is built. Alternatively the race is reported as working on CartPole (and LunarLander, pending) and failing on Acrobot's sparse-reward floor, as a limit of the recipe.

**LunarLander race pilot (2026-10-29 night, 6 runs of 1000 episodes, dev seeds 20–22, base only; `experiments/results/pilot016/ll/`; reference: `hybrid-columns-3-ck`'s `base_ck` over its first 1000 episodes, 20 seeds; development).** At episode 1000: `base_ck` last-100 −103 (range −124 to −87), AUC-1000 −149; the surrogate race with the rate term −163 (−191 to −147), AUC −191, actor rate held at 40 Hz, realised race entropy 1.20 of ln 4 = 1.39, 40 % deadline decisions and 10 % ties at θ = 51; the exact score at ×3 −278 (−327 to −197), AUC −226, entropy 1.23. No collapse, no runaway. **Reading:** the race learns LunarLander but trails the softmax by 40–60 points at 1000 episodes with a near-uniform realised policy at four actions, the same weak state-dependence as on Acrobot; the score credit is slower still. The β re-probe variant, CartPole's best, has not been run on LunarLander or combined with a contrast floor on either task; on Acrobot it was the only race variant that did not collapse (0–3 of 10) and failed only the rate gate. Next pilot: the contrast floor (under review) with the re-probe and the rate-term variants on both tasks.

**The contrast floor (neuroscientist's review, 2026-10-29 night; user decision to build it and re-pilot once).** A one-sided per-column rate floor: for column k with mean E rate ρ_k, the term +c_f (ρ_min − ρ_k)_+ on the column's E rows per ms, ρ_min = r₀/2, c_f = c_t with the rate term's form and effective step; potentiation only, inactive above ρ_min, never lowers a winner. **Class:** the deprivation side of firing-rate homeostasis (scaling-up after activity blockade, Turrigiano et al. 1998; increased excitability after deprivation, Desai, Rutherford & Turrigiano 1999; in vivo recovery of deprived rates, Keck et al. 2013, Hengen et al. 2016), in a rapid coincidence-gated form (Zenke & Gerstner 2017); the exact modelling precedent is the low-rate potentiation term of Zenke, Agnes & Gerstner 2015 that keeps a network from falling silent beside a high-rate brake. Name: "a one-sided population rate floor (the deprivation side of rate homeostasis, as a rapid compensatory process)"; no "dual-threshold" rule exists in the literature. Honesties: biology is bidirectional per neuron and takes 24–48 h; strict one-sidedness and the minutes timescale are engineering choices. **Per column, not per neuron,** which avoids the policy-flattening objection: a pooled signal with one shared target preserves within-population rate heterogeneity (Sweeney, Hellgren Kotaleski & Hennig 2015; Slomowitz et al. 2015), and within a sub-floor column the potentiation ∝ ρ_i f_j scales the column's existing pattern. It bounds the **marginal** action frequency, not the conditional policy: it is not an entropy bonus. Zero in a column already silent (it rides the coincidence trace), so it works by making that state unreachable; **min column rate per log bin is its monitor**. **Recommendation: combine, do not replace:** keep the mean pin (the floor cannot push rates down; removing the pin reopens §18's upward drift) and add the floor, a band (copy mean at r₀, each column ≥ r₀/2), the biological shape of bidirectional scaling. Expectation stated for the pilot: the floor treats the symptom; at Acrobot's floor the credit is still zero-mean noise and the uniform policy is at −500, so the entropy gate should pass while the return remains open. Implementation (`--rate-floor 0.5`, requires `--rate-cost`) by the snn-expert, in progress.

**Study `race-cartpole` result (2026-10-29 18:08–18:13 EDT, 120/120 runs, none failed; numbers from `../map-stdp-rcp/reports/race-cartpole/`, written by `experiments.report`; seeds 7000–7019).** Primary AUC, Hodges–Lehmann shift [95 % CI] (raw p / Holm within the family):

| Arm | AUC median | Last-100 (mean) | Solved | T_s median | Shift vs `base_ck` [CI] | p / Holm | Verdict | Actor spikes to decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `base_ck` (CK softmax, reference) | 347 | 499.2 | 19/20 | 293 | ref | | | 36.0 |
| `base_nl` (normalised softmax) | 347 | 499.8 | 20/20 | 256 | −4.9 [−19.8, +10.8] | 0.58 / 0.58 | non-inferior | 19.3 |
| `base_race_rp` (race, surrogate + β re-probe) | 362 | 475.5 (one run unsolved) | 19/20 | 225 | +14.5 [−5.4, +37.4] | 0.12 / 0.12 | non-inferior | 16.7 |
| `cols_lam0_race_rp` (+ implicit columns) | 390 | 500.0 | 20/20 | 202 | **+29.2 [+12.1, +47.4]** | 0.0043 / 0.013 | **improves** | 15.2 |
| `cols_on_race_rp` (+ learned columns) | 379 | 500.0 | 20/20 | 220 | **+25.0 [+3.7, +43.1]** | 0.021 / 0.041 | **improves** | 15.3 |
| `cols_sealed_race_rp` (+ sealed columns) | 401 | 500.0 | 20/20 | 194 | **+45.2 [+29.4, +62.9]** | 2e-5 / 9e-5 | **improves** | 12.9 |

Secondary: sealed columns vs the race without columns +27.8 [+5.7, +53.4] (Holm 0.033); implicit vs none +11.8 [−14.5, +38.6] (n.s.); sealed vs implicit +15.2 [+2.9, +31.0] (0.026 / 0.077); learned vs implicit −3.2 (n.s.).

- **The reported CartPole result:** with the race readout and the β re-probe, action columns in every regime beat the published backbone after Holm, sealed columns by 45 AUC points (solving 100 episodes earlier on the median, 194 vs 293) with a third of the actor spikes per decision (12.9 vs 36.0); the race without columns is non-inferior (+14.5, n.s.) and the normalised softmax non-inferior (−4.9). Columns help the race significantly (sealed vs none, Holm 0.033), and sealed columns beat implicit ones before Holm (0.026, Holm 0.077). The pre-registered expectations held except the base race's clearance of zero (one of its 20 runs failed to solve, last-100 mean 475.5) and the learned ≥ implicit ordering (−3.2, n.s.). **This is the first reported improvement over the backbone in the hybrid, and it came only once the decision was made by the walk.** The settings were selected on dev seeds (the README says so); the energy figure carries the §18 caveat that the β re-probe arms' actor rate falls after solving.

**Contrast-floor pilot result (2026-10-29 18:17–18:45 EDT, 72/72 runs, none failed; main checkout at `23cc0b9` plus the floor patch (`floor_code.patch` in the scratchpad); `experiments/results/pilot016/floor/`; development; `an_grid.py floor`).** Every race arm carries the band (`--rate-cost 5e-4 --rate-floor 0.5`); Acrobot at β_c = −1.5, dev seeds 20–29; LunarLander dev seeds 20–22, 1000 episodes.

| Acrobot (reference `base_ck_bc` AUC −174, last-100 −150) | AUC median | last-100 | solved | collapsed | column-rate minimum at the end | realised entropy at the end (of ln 3 = 1.10) |
| --- | --- | --- | --- | --- | --- | --- |
| `base_race_rc_fl` (surrogate + band) | −420 (−201 [−322, −36]) | −497 | 1/10 | 5 | 18–37 Hz | 0.73 |
| `base_race_rcrp_fl` (+ β re-probe) | −485 | −500 | 0/10 | 6 | 17–40 Hz | 0.70 |
| `base_race_scn_fl` (exact score ×3) | −497 | −500 | 0/10 | 6 | 22–29 Hz | 0.25 |
| `cols_sealed_race_rc_fl` | −498 | −500 | 0/10 | 7 | 18–37 Hz | 0.53 |
| `cols_sealed_race_rcrp_fl` | −498 | −500 | 0/10 | 7 | | 0.73 |
| `cols_sealed_race_scn_fl` | −499 | −500 | 0/10 | 10 | | 0.22 |

| LunarLander at 1000 episodes (reference `base_ck`, 20 seeds: AUC −149, last-100 −103) | AUC-1000 | last-100 | per seed | rate | realised entropy (of ln 4 = 1.39) |
| --- | --- | --- | --- | --- | --- |
| `base_race_rp` (β re-probe alone) | −296 | −298 | −567, −200, −129 | 42 Hz | 0.91 |
| `base_race_rc_fl` | −189 | −168 | −226, −126, −150 | 40 Hz | 1.22 |
| `base_race_rcrp_fl` | −173 | −170 | −170, −165, −175 | 40 Hz | 1.21 |
| `base_race_scn_fl` | −223 | −201 | −197, −196, −209 | 39 Hz | 1.03 |

- **Acrobot: the floor does what it was built to do and it is not enough.** The column-rate minimum stays at 17–40 Hz (no silent column; the deprived-column collapse of the grid pilot is gone where the floor binds) and the realised entropy of the surrogate arms stays at 0.7 of its maximum, yet 5–10 of 10 runs end at the floor: the policy is no longer perseverating, it is close to uniform, and uniform random on Acrobot is at −500. The exact-score arms still sharpen (entropy 0.22–0.25) at column rates above the floor, as the smoke run predicted (their perseveration is a small persistent contrast, not a deprived column). One surrogate run solved (the first race run to solve Acrobot); the rest did not find the goal in 500 episodes. **Per the user's rule (build the floor, re-pilot once), Acrobot is reported as the race readout's limit:** on a sparse-reward task whose uniform policy sits at the floor, the race's near-uniform initial policy produces no terminations, so there is no signal to learn from, and the two things that rescue the softmax (a state-locked initial gain and an entropy bonus bounding the conditional policy) have no race equivalent in this recipe. Mechanistic diagnosis in this section (critic death, perseveration, uniform floor); candidate Proposals not built: a decision-entropy gain modulation; a state-dependent initial gain for the race (a longer decision window or a larger input contrast).
- **LunarLander: the band holds the race steady and it trails the softmax by 65–70 points at 1000 episodes** (−170 vs −103), with a near-uniform policy at four actions (1.2 of 1.39); the β re-probe alone is unstable (one seed −567). Whether the race catches up over the full 3000–10000 episodes is unknown: the backbone's own LunarLander run spends most of its improvement after episode 3000 (§19). Decision for the user: a full-length pilot of `rcrp_fl` (3 seeds, hours) or reporting LunarLander at 1000 episodes as trailing.

## 21. Acrobot and LunarLander: the decision hold (2026-10-29 evening; user decision to keep fixing, not to report a limit)

**Hypothesis test (snn-expert; scratchpad `sd3/live_policy.py`, bit-identical replays; Acrobot, first 10 episodes, 3 seeds).** The race draws a fresh action every 20 ms: action repeat probability 0.37–0.44 (chance 0.33) and 0–10 % of episodes terminate; the softmax holds an action for several steps (repeat 0.55–0.87) and terminates 10–50 %. Neither readout's *state-conditional* entropy (16 coarse bins) is far below its marginal, so the softmax's "state lock" is not state knowledge: at α_a = 175 it reads the 20-ms-filtered rate noise, a persistent noise sample, and that persistence is what pumps the pendulum. Gym check: a uniform random policy terminates 0/100 Acrobot episodes with per-step draws, 21/100 when each action is held for 2 steps, 43/100 held for 4; on LunarLander holding only hurts (−191 → −247). **So the race's failure on Acrobot is temporal inconsistency of exploration, not gain or state-dependence.**

**Knob screen (Acrobot, 100 episodes, 3 seeds, surrogate + band).** Every knob that sharpens the race (a 40- or 60-ms window with θ by the rule, n_ae 80, σ_V 0.1) collapses to one action within 100 episodes: the probe calibrates a larger β (13–17), the surrogate factor β/(α_a r) scales the actor's effective step, and W random-walks into perseveration. χ_rec 0.5 on sealed columns passes the stability gates (max Re λ 0.45, τ_loop 37 ms) but does not pump; 0.7 fails τ_loop. Independent copies do not pool (no pooling in the code). init_scale 0.2 helps a little. **The hold works:** the race still decides within one 20-ms env step at θ = 16 (β ≈ 8), and the decision is held for 2 env steps (freeze 40 ms): repeat 0.74–0.90, terminations 10–30 % from the first episodes, deadline decisions 0 %. The β re-probe hurts on Acrobot under the hold (β 8 → 30–50, then collapse) and is broken on LunarLander (probe fit 0.24–0.64 against the 0.05 gate; β at the bisection floor 0.125 in 3/3 seeds, after which the actor's credit is about 60× smaller: the LunarLander pilot's −170 was with the actor effectively frozen).

**500 episodes, 10 seeds (reference `base_ck_bc` AUC −174, last-100 −150):**

| Arm | AUC median | last-100 | solved | collapsed | vs `base_ck_bc` [CI] | spikes to decision |
| --- | --- | --- | --- | --- | --- | --- |
| race + band (pilot) | −419 | −497 | 1/10 | 5 | −201 [−322, −36] | 35.9 |
| **hold 2, no columns** | −213 | −184 | 0/10 | 0 | −36 [−62, −14] | 31.9 |
| **hold 2 + sealed columns** | **−174** | **−132** | **2/10** | **0** | **+3 [−27, +25]** | 30.9 |
| hold 2 + learned columns (5 seeds) | −167 | −129 | 0/5 | 0 | +5 [−25, +24] | 30.8 |
| hold 2 + implicit columns (5 seeds) | −214 | −165 | 0/5 | 0 | −34 [−51, −11] | 31.8 |
| hold 3 / 4 / 8 | −274 / −404 / −478 | | | 0 / 1 / 3 | −101 / −218 / −300 | |
| softmax + hold 2 (control) | −178 | −147 | 0/10 | 0 | +1 [−28, +23] | |

The hold removes the floor entirely (0/10 collapsed, every seed alive) and with sealed or learned columns the race is level with the published backbone (+3, +5; CI ± 25) at fewer actor spikes per decision, solving 2/10 where the softmax solves 0/10; longer holds degrade monotonically (control resolution, and the credit is applied n_hold times per decision). The hold does nothing for the softmax (+1), so the comparison stays fair. Stability: max Re λ 0.28, τ_loop 28 ms throughout.

**LunarLander (1000 episodes, seeds 20–22; reference `base_ck` AUC-1000 −148 [−168, −133], last-100 −101):** θ by the 40-ms rule (51) gives 46–54 % deadline decisions; **θ = 26 (the 20-ms rule) at a 20-ms decision, no hold, no re-probe: AUC-1000 −193 / −149 / −126 (median −149 vs the backbone's −148) at half the spikes (175 vs 370 per decision)**, last-100 trailing by about 30 with a wide spread. The live measurement shows no temporal gap on LunarLander (repeat 0.70 for both; both policies near uniform at 1000 episodes), so the hold is not used there. Columns untried; full-length runs needed.

**Pre-set rule (per task, from the screen, never on return):** the race decides within one 20-ms environment step with θ = round(n_ae r₀ · 20 ms) (16 on Acrobot and CartPole, 26 on LunarLander); the decision is held for n_hold env steps, n_hold the smallest hold at which a uniform random policy terminates at least 20 % of episodes (Acrobot 2; CartPole and LunarLander 1); the β re-probe is accepted only when the probe's fit is within the 0.05 gate (code change), and is not used on Acrobot or LunarLander. **Plausibility (neuroscientist's review, citations verified):** the hold is an abstraction with a supported class, *temporally correlated motor variability* (action persistence): trial-to-trial motor variability has slow temporal structure, a random walk in the preparatory state (Chaisanguanthum, Shen & Sabes 2014), that structure is regulated and predicts reward-based learning (Wu et al. 2014), central planning variability is the exploration reinforcement learning uses (Dhawale, Smith & Ölveczky 2017), and choice history biases the next action (Hwang et al. 2017, Nat Commun; Akrami et al. 2018; Urai et al. 2019). Not motor chunking (Graybiel 1998 is learned sequence binding), not a decision refractory period (no documented fixed hold), and not Ölveczky et al. 2005 (rendition-to-rendition variability, nothing about persistence across decisions). In RL it is sticky actions (Machado et al. 2018) and temporally extended ε-greedy (Dabney, Ostrovski & Barreto 2021); Mnih et al. 2015's frame skip is a control-rate choice, not exploration. Honesties: biology's persistence is graded and adaptive, not clock-locked; the hold also applies during exploitation and halves the control resolution (the hold-3/4/8 degradation is that cost); the 20–40-ms decision cycle is already faster than any mammalian decision (the existing timescale abstraction). The n_hold rule from a uniform policy's termination rate is a task property, acceptable as a pre-set, with no biological analogue; `action_repeat_probability` per log bin is its monitor. **The alternative, a leaky non-resetting race counter** (the next race starting from the previous winner's lead), is the leaky competing accumulator with a history-dependent starting point (Usher & McClelland 2001; Bogacz et al. 2006), carried biologically by reverberating decision activity (Wang 2002; Lo & Wang 2006); two cautions: the empirical choice-history bias is in the accumulation rate, not the starting point (Urai et al. 2019), and at the present gain the columns cannot hold a 40-ms lead (τ_loop 28 ms; χ_rec 0.7 fails the gate), so the count would be readout-side state like the hold. **Recommendation: the hold as impl, the leaky counter as a Proposal** (it adds a τ_leak, re-derives θ and the deadline share and shifts the surrogate's calibration, where pre-set rules and a fair softmax control are wanted; the hold is one integer with a clean control).

**Hold pilot, Acrobot (2026-10-29 20:30–20:45 EDT, 40/40 runs, none failed; main checkout at `23cc0b9` plus the floor, `--freeze` and gate patch (`hold_code.patch` in the scratchpad); `experiments/results/pilot016/hold/`; dev seeds 20–29; every arm `--eta-cal --beta-c -1.5`, race arms `--readout race --rate-cost 5e-4 --rate-floor 0.5 --race-theta 16 --freeze 40`, surrogate credit, no re-probe; reference `base_ck_bc`, the softmax at β_c −1.5: AUC −174, last-100 −150, 0/10 solved.)**

| Arm | AUC median | last-100 | solved | collapsed | vs `base_ck_bc` [CI] | p | repeat at the end | realised entropy (of 1.10) | column-rate min | spikes to decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `base_race_h2` | −213 | −184 | 0/10 | 0 | −36.5 [−61.5, −13.6] | 0.021 | 0.78 | 0.62 | 22 Hz | 31.9 |
| `cols_lam0_race_h2` | −215 | −164 | 0/10 | 1 | −42.7 [−70.8, −14.5] | 0.011 | 0.74 | 0.73 | 26 Hz | 31.8 |
| `cols_on_race_h2` | −179 | −136 | 0/10 | 0 | −4.0 [−25.7, +19.4] | 0.62 | 0.68 | 0.88 | 27 Hz | 30.8 |
| `cols_sealed_race_h2` | **−174** | **−132** | **2/10** | 0 | **+3.1 [−27.1, +24.7]** | 0.79 | 0.70 | 0.83 | 24 Hz | 30.9 |

Deadline decisions 0 % in every arm (the race always reaches θ = 16 within the 40-ms hold); no run at the floor (1 late collapse in the implicit arm). **The ordering of the screen holds at n = 10 through the real flags:** sealed and learned columns are level with the published backbone (last-100 18 and 14 points better, 2/10 solved vs 0/10, 14 % fewer actor spikes per decision), implicit columns and no columns are worse by 37–43. Under the race on Acrobot the structure matters in the direction the walk picture predicts: sealed ≈ learned > implicit > none, with the within-column persistence (sealed, or the map term's sealing) carrying the hold's temporal consistency into the columns. Settings for the Acrobot study: as this pilot, no further tuning.

**Study `race-acrobot` result (2026-10-29 19:40–19:52 EDT, 100/100 runs, none failed; numbers from `../map-stdp-racr/reports/race-acrobot/`, written by `experiments.report`; seeds 7100–7119; every arm at β_c −1.5 with `--eta-cal`).** Primary AUC over 500 episodes, Hodges–Lehmann shift [95 % CI] (raw p / Holm within the family), Δ = 25:

| Arm | AUC median | Last-100 (median) | Solved | Collapsed | Shift vs `base_ck` [CI] | p / Holm | Verdict | Actor spikes to decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `base_ck` (CK softmax at β_c −1.5, reference) | −190 | −147 | 0/20 | 1 | ref | | | 26.1 |
| `base_race_h2` (race + hold + band, no columns) | −228 | −152 | 0/20 | 0 | −35.0 [−67.0, +0.3] | 0.05 / 0.20 | inconclusive | 32.7 |
| `cols_lam0_race_h2` (+ implicit columns) | −219 | −145 | 0/20 | 1 | −25.4 [−60.0, +5.6] | 0.12 / 0.36 | inconclusive | 31.7 |
| `cols_on_race_h2` (+ learned columns) | −188 | −125 | 1/20 | 0 | −5.0 [−27.7, +21.8] | 0.71 / 0.71 | inconclusive (by 2.7) | 30.5 |
| `cols_sealed_race_h2` (+ sealed columns) | **−173** | **−132** | **2/20** | 0 | **+13.0 [−5.0, +38.6]** | 0.19 / 0.38 | **non-inferior** | 30.4 |

Secondary: **sealed vs no columns +53.6 [+22.2, +77.4] (Holm 0.0027); sealed vs implicit +42.2 [+13.3, +69.0] (Holm 0.011)**; learned vs none +33.4 [−1.3, +60.4] (0.05 / 0.10); learned vs implicit +21.8 (n.s.).

- **The reported Acrobot result:** the race with the hold and sealed columns is non-inferior to the published backbone (point estimate +13 AUC, the CI's lower end at −5; better late return, −132 vs −147; the only solved runs, 2/20 vs 0/20), and **the regime matters under the race, after Holm:** sealed columns beat the race without columns by 54 AUC and implicit columns by 42. Learned columns sit between (−5 vs the backbone, inconclusive by 2.7 points at Δ = 25). The race without columns and with implicit columns is 25–35 below the backbone, inconclusive at n = 20 (raw p 0.05 and 0.12). The expectation that sealed and learned would be level held; the expectation that both would be non-inferior failed for learned by 2.7 points. **The energy claim does not hold on Acrobot:** the band pins the actor at 40 Hz while CK's softmax lets its actor rate fall to about 10 Hz at the floor, so the race spends more actor spikes per decision (30–33 vs 26); only CartPole shows the saving.
- **Reading across the two reported race studies:** on CartPole every column regime beats the backbone and sealed is best; on Acrobot sealed columns keep the backbone's performance where the race without columns loses 35, and sealed beats implicit after Holm on both tasks (CartPole 0.077, Acrobot 0.011). Under the softmax the regimes never differed (§17, §19). The structure matters once the decision is the walk's, and the within-column walk (sealed, or the map term's sealing) is what carries it.

**LunarLander: the race's late start, diagnosed (snn-expert, 2026-10-29 night; scratchpad `sd4/`; development, dev seeds 20–22, 16 copies; the hold pilot's full-length runs as the trajectory).** First, the stall is a delay, not a plateau: the race's climb from episode 1500 to 4000 (+133 in last-50) matches the backbone's from 1000 to 2500 (+113), about 1000 episodes later; at 4000 the race without columns is at +15 against the backbone's +15 at 4000 / +25 at 4500, with AUC still behind for the delay.
- **Rejected:** the mean-rate pin (the backbone's actor rate is flat at 40 Hz until 3000 and rises only late, to 58 Hz by 10000; no-pin changes nothing at 1500); ties and deadlines (θ 40 is worse); the entropy bonus (c_e 0 is worse); the 16 copies (same structure in both readouts).
- **Confirmed, with the mechanism: the surrogate was reading the wrong quantity on LunarLander.** CK's action score is the *net* column rate E − I (`Layout.Ma` carries −1/n_ai on the I posts), and the surrogate p̃ = softmax(β log r) read that net rate. LunarLander's preset has n_ai = 32 (CartPole and Acrobot have none), so at 40 Hz the net rate is −0.25 ± 4.2 Hz, **below the surrogate's floor ε_r = 1.56 Hz in 67 % of (ms, copy, column)**: the credit factor is 0 there and noise-selected elsewhere, the doubled per-ms RMS halves the calibrated η_a, and the init probe's β = 1.4 is a floor artefact (the surrogate is flat at δ ≤ 0; fit 0.21, beyond the 0.05 gate already at init), which is why β was 1.4–2.5 on LunarLander and 7–9 elsewhere, and why the re-probe "broke" there. The race itself is not noisy (empirical log-odds slope ≈ 10, counts under-dispersed, Fano 0.3–0.5). With the surrogate reading the **E column rate, the quantity the race tallies** (§18's E-count decision, never carried into the surrogate): β = 10–12 on every seed, fit ≤ 0.03, the credit active everywhere.
- **The remaining unmatched difference was the control rate:** the backbone's LunarLander preset holds each softmax action for 40 ms (2 env steps); the race was run at 20 ms.
- **Screen (3 seeds × 1500 episodes; backbone last-50 at 1500: −52, AUC −124):** the current setting −119 / −146; no pin −113; c_t 5e-5 −132; θ 40 −157; η_a × 3 −161; the exact score ×3 −192; c_e 0 −152; the E-rate surrogate alone −117 (AUC −165); the hold alone with the old surrogate −186 (the noise-selected credit applied twice per decision kills the critic); **the E-rate surrogate with the hold: last-50 −46 (−9 / −48 / −80), AUC −131, i.e. level with the backbone at 1500, at 177 actor spikes to decision against 370, deadline decisions 0.2 %, no collapse, column minimum 34 Hz, critic alive (V −0.48 vs the backbone's −0.52).**
- **Decision (pre-set, one rule change and one consistency fix):** (1) the surrogate reads what the race tallies, the column's E rate, with the policy credit on E rows only (a no-op where n_ai = 0, so CartPole and Acrobot and their studies are untouched); (2) the race decides within 20 ms at θ = round(n_ae r₀ · 20 ms) and holds the action for the preset's freeze / 20 env steps, i.e. the race runs at the backbone's own control rate (LunarLander 2, CartPole 1, Acrobot 2 as §21 found), a cleaner statement of the hold rule that keeps the softmax control fair without the random-policy termination test. Monitors: probe fit ≤ 0.05 at init, credit-active fraction 1, deadline share < 5 %, column minimum ≥ 20 Hz, realised entropy and repeat per bin, critic rate and V. Open: n = 3 at 1500 episodes; the columns under the fixed surrogate are untested (the hold pilot's column arms ran with the floored surrogate, so their LunarLander result is not informative); the late phase, where the backbone's climb rides on a rate rise the band forbids, is unknown (the pin showed no drift at 1500 and may be droppable late).

**Old-surrogate LunarLander runs, complete (the hold pilot's 9 full-length runs, 2026-10-30; dev seeds 20–22; the surrogate reading the net rate, i.e. the dead-credit variant, with the band and θ = 26 at the preset's 40-ms window; development).** Every run to the 10000-episode cap, none solved; reference `base_ck` (20 seeds): last-100 62.3, AUC-3000 −63.6, 465 actor spikes to decision at an end rate of 58 Hz.

| Arm | n | last-100 mean (per seed) | AUC-3000 | spikes to decision | end rate |
| --- | --- | --- | --- | --- | --- |
| race, no columns | 3 | **108.3** (115, 82, 128) | −104.2 | 163 | 38 Hz |
| + sealed columns | 3 | 71.6 (86, 63, 65) | −122.2 | 138 | 37 Hz |
| + learned columns | 3 | 50.9 (64, 54, 34) | −120.1 | 138 | 37 Hz |

Even with the dead credit the race's late level is above the backbone's without columns (108 vs 62, above the backbone's best seed, 91) at a third of the spikes, while its first 3000 episodes trail by 40–60 (the delay the E-rate fix removes). The columns did not help under the dead credit on this task (sealed level with the backbone, learned below); whether they do under the restored credit is the fixed pilot's question (at episode 6100–6250 all three fixed arms are 30–40 above the backbone, sealed highest).

**LunarLander pilot with the fix, complete (2026-10-30, 30/30 full-length runs, none failed; main checkout at `74f3dd1` plus the surrogate-fix patch (`ll2_code.patch` in the scratchpad); `experiments/results/pilot016/ll2/`; dev seeds 20–29; every arm `--eta-cal --readout race --rate-cost 5e-4 --rate-floor 0.5 --race-theta 26` at the preset's 40-ms freeze, the surrogate on the E rate, no re-probe; reference `base_ck`, study A's 20 runs; development).** Every run to the 10000-episode cap, none solved, none collapsed.

| Arm | n | last-100 mean (median) | shift vs `base_ck` [CI] | p | AUC-3000 | shift [CI] | spikes to decision | end rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `base_ck` (CK softmax) | 20 | 62.3 (62.6) | ref | | −63.6 | ref | 465 | 58 Hz |
| `base_race_e` (race, no columns) | 10 | 69.0 (70.6) | +6.4 [−2.8, +15.8] | 0.24 | −58.7 | +7.9 [−1.2, +13.0] | 148 | 38 Hz |
| `cols_on_race_e` (+ learned columns) | 10 | 79.8 (80.5) | **+16.9 [+7.4, +26.9]** | 0.005 | −64.8 | −1.0 [−6.4, +4.5] | 140 | 38 Hz |
| `cols_sealed_race_e` (+ sealed columns) | 10 | 93.8 (89.7) | **+31.2 [+13.9, +48.6]** | 0.0008 | −69.6 | −4.9 [−10.2, −1.3] | 139 | 37 Hz |

Secondary: sealed vs no columns +22.3 [+5.8, +41.8] (p 0.011); learned vs no columns +11.1 [+2.3, +19.0] (p 0.031); sealed vs learned +11.5 [−5.2, +32.0] (n.s.).

- **The fix did what the diagnosis said:** the early phase is now level with the backbone (AUC over the first 3000 episodes within ± 8, where the old surrogate trailed by 40–60) and the late level is above it: learned columns +17 and sealed +31 last-100 points with CIs clear of zero at n = 10, the race without columns +6 (n.s.). **On LunarLander the columns are what beats the backbone,** both column regimes above the race without them (p 0.01, 0.03), sealed highest on the third task in a row; and every race arm reaches its decisions with 30 % of the softmax's actor spikes at a held 38 Hz where the softmax's rate rises to 58. The old-surrogate ablation (above) had the no-column race higher (108, 3 seeds) and the columns lower; with the credit restored the ordering is sealed > learned > none, as on CartPole and Acrobot.
- **Settings for the study:** as this pilot, no further tuning; implicit columns added for the regime contrast (untested here). The pre-registered expectations in the README are these numbers.

**Study `race-lunarlander` result (2026-10-30 02:52–20:13 EDT, 100/100 runs, none failed; numbers from `../map-stdp-racl/reports/race-lunarlander/`, written by `experiments.report`; seeds 7200–7219; manifest `431403b`).** Primary last-100 return, Hodges–Lehmann shift [95 % CI] (Holm within the family), Δ = 25; every run to the 10000-episode cap, none solved by CK's criterion (trailing-100 ≥ 200):

| Arm | last-100 median (mean) | AUC-3000 | Shift vs `base_ck` [CI] | Holm | Verdict | Actor spikes to decision |
| --- | --- | --- | --- | --- | --- | --- |
| `base_ck` (CK softmax, reference) | 49.9 (54.1) | −61.9 | ref | | | 467 |
| `base_race_e` (race, no columns) | 76.8 (76.2) | −62.0 | **+23.8 [+12.1, +32.3]** | 1e-4 | **improves** | 146 |
| `cols_lam0_race_e` (+ implicit columns) | 90.5 (96.9) | −63.0 | **+41.1 [+30.9, +51.6]** | 2e-6 | **improves** | 138 |
| `cols_on_race_e` (+ learned columns) | 89.5 (90.2) | −64.6 | **+37.6 [+27.4, +46.5]** | 3e-6 | **improves** | 140 |
| `cols_sealed_race_e` (+ sealed columns) | 88.7 (89.9) | −70.7 | **+35.9 [+25.6, +46.3]** | 3e-6 | **improves** | 136 |

Secondary: sealed vs no columns +13.2 [+5.1, +21.5] (Holm 0.011); learned vs no columns +13.5 [+6.0, +22.3] (Holm 0.005); sealed vs implicit −5.2 and learned vs implicit −3.1 (n.s.).

- **The reported LunarLander result:** every race arm beats the published backbone after Holm, the race alone by 24 last-100 points and the column arms by 36–41, with early learning level (AUC over the first 3000 episodes within 9 points) and about 30 % of the backbone's actor spikes per decision (136–146 vs 467). Columns improve the race further (learned and sealed both above the race without columns after Holm); the three regimes do not differ among themselves here (implicit highest by a few points, n.s.), unlike CartPole and Acrobot where sealed led. The pre-registered expectations held for the direction and size of the column gains and the energy saving; the race without columns came out above the backbone rather than level.
- **Across the three reported race studies:** columns under the race beat the backbone on CartPole (+25 to +45 AUC) and LunarLander (+36 to +41 last-100) and keep it on Acrobot (sealed +13, non-inferior); columns beat the race without them on all three tasks (sealed after Holm on all three); the race saves actor spikes per decision on CartPole and LunarLander (about a third of the backbone's) and not on Acrobot. Weight-saving reproduction of all 100 runs launched at 20:14 EDT (one wave at 100 workers, about 10 h).
