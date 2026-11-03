# Iteration 018: how much cross-column wiring each task wants

## 1. Problem

The race studies order the column regimes differently by task:

- **CartPole** (`reports/race-cartpole`): sealed beats implicit by +15.2 AUC [+2.9, +31.0].
- **Acrobot** (`reports/race-acrobot`): sealed beats implicit by +42.2 AUC [+13.3, +69.0].
- **LunarLander** (`reports/race-lunarlander`): implicit is at least as good as sealed. Last-100: sealed − implicit −5.2 [−14.3, +6.6]. AUC: −5.8 [−12.2, −0.9].

The user's question (2026-11-01): view the column graph as an SBM whose cross-module connection probability runs from 0 (sealed) to unconstrained. Is there a level in between that is best for each task?

## 2. The axis

`HybridRec` draws R's mask as an SBM over the actor E posts, one module per action.

- **Within a column:** p_same = 0.5.
- **Across columns:** p_cross = 0.05.
- **`--rec-seal`:** keeps the same draws and drops the cross entries. Sealed is therefore exactly p_cross = 0, with the same R.
- **Implicit (`cols_lam0`):** p_cross = 0.05, a topology that is already nearly sealed. Its large learned crossing shares of the recurrent drive (Acrobot 31 %, LunarLander 58 %; the replay's `rec.cross`) are weight that learning put on a few cross synapses.
- **Unstructured:** at the same density, p_cross = p_same = 0.5 (Erdős–Rényi).

`--p-cross` (`Config.p_cross`, token `px<p>`; default 0.05 keeps every name and draw) sets p_cross. Only the comparison `rng.random(P.shape) < P` depends on it. So for one seed the masks are nested in p_cross, and the within-column draws are identical (`tests/test_hybrid.py::test_rec_p_cross_sweep`). Each point adds cross synapses to the same graph.

The initial cross share of a neuron's outflow is s0 = (K − 1) p_cross / (p_same + (K − 1) p_cross) in expectation, with K actions. The grid in s0, and in cross synapses per neuron, is:

| p_cross | 0 | 0.0125 | 0.025 | 0.05 | 0.1 | 0.2 | 0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole (K = 2): s0 | 0 | 2.4 % | 4.8 % | 9.1 % | 16.7 % | 28.6 % | 50 % |
| cross synapses / neuron | 0 | 0.25 | 0.5 | 1 | 2 | 4 | 10 |
| Acrobot (K = 3): s0 | 0 | 4.8 % | 9.1 % | 16.7 % | 28.6 % | 44.4 % | 66.7 % |
| cross synapses / neuron | 0 | 0.5 | 1 | 2 | 4 | 8 | 20 |
| LunarLander (K = 4): s0 | 0 | 7.0 % | 13.0 % | 23.1 % | 37.5 % | 54.5 % | 75 % |
| cross synapses / neuron | 0 | 0.75 | 1.5 | 3 | 6 | 12 | 30 |

**What p_cross also sets.** R's task credit is one-hop, h_m(i) − h̄_j. A neuron whose targets all lie in its own column gets exactly zero credit, so sealed R never learns (017 §2). p_cross therefore sets both the topology and how much of R can learn at all. The sweep measures the two together.

**Rates per arm.**

- **Shared per seed:**
  - η_rec and λ_rec (`calibrate_rec`) are measured on the default draws (p_cross 0.05), as sealed already is.
  - s_in and `--eta-cal` are calibrated without recurrence.
- **Per arm:**
  - `--rate-match-rec` re-fits s_in to each point's own R, so every point starts at 40 Hz. This is the same per-arm matching sealed and implicit get in the race studies.
  - On CartPole (no `--eta-cal`) the RMS-matched η_a, and with it η_rec, follow that per-arm fit, as they already do between sealed and implicit.

## 3. Dev stage (pre-registered 2026-11-01, before any run)

- **Arms:** `cols_lam0` (map term off, `--chi-rec 0.3 --rec cols --lam 0`) with each task's race-study flags, plus `--p-cross p` at p ∈ {0, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.5}. p = 0 runs as `--p-cross 0`, the same R as `--rec-seal` (smoke-checked: identical Acrobot returns, seed 20).
  - **CartPole** (as `race-cartpole`): `--rate-target 40 --block 20 --fast --rate-match-rec --readout race --race-reprobe 50`.
  - **Acrobot** (as `race-acrobot`): `--rate-target 40 --block 20 --fast --rate-match-rec --eta-cal --beta-c -1.5 --readout race --rate-cost 5e-4 --rate-floor 0.5 --race-theta 16 --freeze 40`.
  - **LunarLander** (as `race-lunarlander`): `--rate-target 40 --block 20 --fast --rate-match-rec --eta-cal --readout race --rate-cost 5e-4 --rate-floor 0.5 --race-theta 26`.
  - Each run uses `--engine nlif` and the task's default episode count.
- **Seeds:** dev seeds 20–29. 3 tasks × 7 points × 10 seeds = 210 runs, into `experiments/results/pilot018/` (intermediates).
- **Primary metric:** as the race studies, AUC on CartPole and Acrobot and last-100 return on LunarLander.
- **Selection rule:** p*(task) = the grid point with the highest dev-seed mean of the primary metric. If p* is 0 or 0.05 (an existing arm), the task gets no confirmation arm, and its dev curve is reported as "no better interior level on dev".
- **Monitors per point:**
  - s0 and the realised initial cross share of R;
  - the final crossing share of the recurrent drive;
  - the column E rates;
  - the walk statistics HybridRec logs (max Re λ, τ_loop, `within`).
- **Figure:** the primary metric against p_cross (0 drawn as its own tick) and against s0, per task, mean ± SE over seeds.
- **Cost:** CartPole and Acrobot take about 0.1 h per run. LunarLander takes 6–12 h per run (sealed 6.2 h, implicit 12.4 h in `race-lunarlander`), likely more at high p_cross. That is one wave of about 12–20 h on 70 workers, with nothing else heavy alongside it.
- **Stop:** the dev curves and p* are recorded here and reviewed with the user before the confirmation stage.

## 4. Confirmation stage (declared after the dev stage, before its launch)

**Not run** (user decision 2026-11-02, §6): the dev sweep is reported as the result instead.

Study `cross-sweep` in `experiments/report.py`:

- **Seeds:** fresh, 8000–8019 (CartPole), 8100–8119 (Acrobot), 8200–8219 (LunarLander).
- **Arms per task:** `base_ck` (reference), `cols_sealed`, `cols_lam0` (p 0.05) and `cols_px<p*>`.
- **Pairs:**
  - family `primary`: p* vs sealed and p* vs implicit;
  - family `secondary`: each arm vs `base_ck`.
- **Statistics:** Holm per family, as the race studies.
- **Size:** at most 240 runs. Final numbers only from `reports/cross-sweep/`.

## 5. Dev stage results (2026-11-02; development, dev seeds 20–29)

**Runs.** 210 runs from the `cross-sweep` worktree at `e1c27bc`, launched 2026-11-01 19:25 by `experiments/results/pilot018/launcher.py --workers 70` (results and logs in `experiments/results/pilot018/<task>_px<p>/`). All 210 finished, with no failure. LunarLander took 12.5 h per run on average.

**Primary metric**, mean ± SE over 10 seeds. Paired = mean difference from sealed on the same seed, with the number of seeds where the point beats sealed. Within = `rec.within_final`, the share of the final walk's recurrent steps that stay in their column (chance is 1/K).

| p_cross | CartPole AUC | paired | within | Acrobot AUC | paired | within | LunarLander last-100 | paired | within | LunarLander AUC (first 3000) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (sealed) | 405.4 ± 4.8 | – | 1.000 | −185.5 ± 12.0 | – | 1.000 | 93.8 ± 7.4 | – | 1.000 | −69.6 ± 1.8 |
| 0.0125 | 341.2 ± 39.0 | −64.2 (1/10) | 1.000 | −196.3 ± 11.6 | −10.7 (4/10) | 0.865 | 84.8 ± 6.6 | −9.0 (4/10) | 0.653 | −66.9 ± 2.1 |
| 0.025 | 387.3 ± 8.5 | −18.1 (3/10) | 1.000 | −238.4 ± 11.9 | −52.8 (1/10) | 0.804 | 95.1 ± 5.8 | +1.4 (7/10) | 0.479 | −62.8 ± 3.2 |
| 0.05 (implicit) | 379.9 ± 9.4 | −25.4 (2/10) | 0.998 | −244.4 ± 29.6 | −58.9 (2/10) | 0.636 | 88.5 ± 3.3 | −5.2 (4/10) | 0.318 | −59.5 ± 3.3 |
| 0.1 | 378.3 ± 16.2 | −27.0 (4/10) | 0.998 | −236.6 ± 33.1 | −51.0 (2/10) | 0.471 | 93.8 ± 6.5 | −0.0 (6/10) | 0.265 | −63.8 ± 1.6 |
| 0.2 | 366.4 ± 22.5 | −39.0 (3/10) | 0.996 | −271.5 ± 32.8 | −85.9 (1/10) | 0.350 | 88.0 ± 4.9 | −5.8 (4/10) | 0.258 | −58.4 ± 3.1 |
| 0.5 | 376.0 ± 12.5 | −29.3 (2/10) | 0.990 | −273.9 ± 42.0 | −88.3 (3/10) | 0.255 | 89.4 ± 5.4 | −4.4 (6/10) | 0.250 | −58.5 ± 2.5 |

**Selection (the pre-registered rule).**

- **CartPole:** p* = 0, so no confirmation arm.
- **Acrobot:** p* = 0, so no confirmation arm.
- **LunarLander:** p* = 0.025, an interior point, so it gets the confirmation arm. Its margin over sealed is +1.4 (7/10 seeds), well inside the noise. Its learned walk is half-crossing (within 0.48).

**Reading (development numbers).**

- **CartPole.** Any cross wiring costs 20–40 AUC, with no trend beyond the first synapses. Yet the learned walk stays within its column at every point (within ≥ 0.99), so learning seals the columns. The cost is paid while learning, before the cross weights decay.
- **Acrobot.** The cost grows with p_cross. Learning does not seal Acrobot: the walk's within share falls with p_cross to chance (0.26 at 0.5, K = 3). Runs that never learn (AUC near −500) appear from p_cross 0.05 up.
- **LunarLander.** The last-100 return is flat across the grid (84.8–95.1). The learned walk is at chance mixing (within ≈ 0.25, K = 4) from p_cross 0.1 up, without loss of final return. AUC (the report's definition: mean return over the first 3000 episodes, the run's minimum) is higher with any cross wiring (−66.9 to −58.4 vs −69.6 sealed), highest from implicit up: cross wiring speeds early learning, as `race-lunarlander`'s AUC contrast showed. The pre-registered primary metric does not reward that.

## 6. Reported as study `cross-sweep` (user decision 2026-11-02)

The user judged the dev curves conclusive and asked for them to be reported as the final result of this experiment, in place of the fresh-seed confirmation (§4).

- **How the runs were moved.** The 210 runs were copied unchanged into `reports/cross-sweep/runs/`. The study is declared in `report.py` (`_cross_sweep`, seeds 20–29) and summarised by the runner (`reports/cross-sweep/summary.md`).
- **Where the deviations are recorded.** The README states the three deviations from the other final studies: n = 10, dev seeds that also chose the race settings, and no confirmation.
- **The Holm-adjusted tests.** With 6 pairs per family and n = 10, they are weak:
  - Acrobot: sealed − implicit +41.7 AUC [+8.0, +73.0], p 0.031 (Holm 0.19). Sealed − 0.025: +52.4 [+20.3, +95.3], p 0.011 (Holm 0.068). Only sealed solves any run (2/10).
  - CartPole: implicit − sealed −22.7 AUC [−54.6, +1.8], p 0.076. Sealed reaches $T_s$ fastest (192.8 vs 204–233 episodes).
  - LunarLander: last-100 return is flat (every shift against sealed within ±8, CIs about ±25). AUC against sealed rises with cross wiring, +2 to +12, with 0.5 at p 0.0046 (Holm 0.028).

**Conclusion.** Sealed is the best wiring on CartPole and Acrobot. LunarLander's final return does not depend on the wiring, while its learning speed (AUC) rises with cross wiring. No task has an interior optimum on its primary metric.

## 7. Map term on: the same sweep (pre-registered 2026-11-02, before any run)

**Question.** Can learned sealing (the map term) replace enforced sealing? Section 6 varied the wiring with the map term off. With it off, learning sealed CartPole's walk by itself (within ≥ 0.99 at every p_cross) but not Acrobot's (within fell to chance), and Acrobot's cost grew with p_cross. The map term pushes R's cross weights toward the given partition. In the race models it nearly seals the walk on every task (crossing share of the recurrent drive 0.08 %, 0.19 % and 0.43 %, against 0.09 %, 31 % and 58 % with it off).

**Arms.** The groups of `cross-sweep` with `--lam 1` in place of `--lam 0`; everything else is identical. That is the race studies' `cols_on` at each p_cross ∈ {0, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.5}.

- λ_rec comes from `calibrate_rec` on the default draws (p_cross 0.05), so it is shared per seed across the grid.
- At p_cross 0 every target lies in its own column. The map term then has little or nothing to act on, and the point is expected to be close to sealed with the map term off. It is run, not assumed.

**Seeds.** Dev seeds 20–29, the same as `cross-sweep`, so every (task, p_cross, seed) pairs with its map-term-off run. 3 tasks × 7 points × 10 seeds = 210 runs, into `experiments/results/pilot018on/<task>_px<p>/`.

**Code.** Branch `cross-sweep` at `e1c27bc`, with `experiments/tdstdp.py` and `mapstdp/` unchanged since the off sweep. The only uncommitted change is `report.py`'s `cross-sweep` declaration.

**Metrics.** As `cross-sweep`: AUC on CartPole and Acrobot, last-100 return and AUC (first 3000 episodes) on LunarLander.

**Headline contrast.** The paired difference on − off at each p_cross, per task, as a mean over seeds with a Wilcoxon signed-rank p, Holm over the 7 points per task and metric.

**Trend.** Spearman correlation of the paired difference with the p_cross rank, per task.

**Predictions** (stated before the runs):

- Acrobot: on − off grows with p_cross.
- CartPole: on ≈ off.
- LunarLander: on loses some of the AUC that cross wiring gives with the map term off; last-100 flat.

**Monitors.** As section 5: `rec.within_final` per point, rates, and failed runs (AUC at the floor).

**Reporting.** The results go into section 8. Whether they become a reported study, and how, is the user's decision after the runs.

## 8. Map term on: results (2026-11-02; development, dev seeds 20–29)

**Runs.** 210 runs from the `cross-sweep` worktree at `e1c27bc` (plus the uncommitted `report.py` declaration), launched 2026-11-02 10:15 by `experiments/results/pilot018on/launcher.py --workers 70` (results and logs in `experiments/results/pilot018on/<task>_px<p>/`). All 210 finished, with no failure, by 21:34. LunarLander took 10.2 h per run on average (12.5 h with the map term off). The map-term-off runs are those of `reports/cross-sweep/runs/`.

**Paired contrast** (pre-registered headline, §7). On − off on the same seed: mean difference, the number of seeds where on beats off, Wilcoxon signed-rank p, and Holm over the 7 points per task and metric. Within = `rec.within_final`, on / off. Failed = AUC below the floor (Acrobot −450, CartPole 50), on / off.

**CartPole AUC.**

| p_cross | on | off | on − off | on > off | p | Holm | within | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (sealed) | 391.0 ± 4.8 | 405.4 ± 4.8 | −14.4 | 3/10 | 0.049 | 0.34 | 1.000 / 1.000 | 0 / 0 |
| 0.0125 | 391.6 ± 4.8 | 341.2 ± 39.0 | +50.4 | 3/10 | 0.92 | 1 | 1.000 / 1.000 | 0 / 1 |
| 0.025 | 380.6 ± 6.1 | 387.3 ± 8.5 | −6.7 | 3/10 | 0.32 | 1 | 1.000 / 1.000 | 0 / 0 |
| 0.05 (implicit) | 381.6 ± 15.6 | 379.9 ± 9.4 | +1.7 | 5/10 | 0.63 | 1 | 0.999 / 0.998 | 0 / 0 |
| 0.1 | 396.9 ± 4.5 | 378.3 ± 16.2 | +18.6 | 5/10 | 0.43 | 1 | 0.998 / 0.998 | 0 / 0 |
| 0.2 | 357.9 ± 18.0 | 366.4 ± 22.5 | −8.5 | 3/10 | 0.32 | 1 | 0.996 / 0.996 | 0 / 0 |
| 0.5 | 371.7 ± 20.7 | 376.0 ± 12.5 | −4.3 | 5/10 | 1.00 | 1 | 0.990 / 0.990 | 0 / 0 |

Spearman of the per-seed difference with the p_cross rank: ρ −0.007, p 0.95 (n = 70).

**Acrobot AUC.**

| p_cross | on | off | on − off | on > off | p | Holm | within | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (sealed) | −184.0 ± 9.2 | −185.5 ± 12.0 | +1.6 | 5/10 | 1.00 | 1 | 1.000 / 1.000 | 0 / 0 |
| 0.0125 | −226.9 ± 23.3 | −196.3 ± 11.6 | −30.6 | 5/10 | 0.38 | 1 | 0.999 / 0.865 | 0 / 0 |
| 0.025 | −223.4 ± 19.4 | −238.4 ± 11.9 | +15.0 | 6/10 | 0.28 | 1 | 0.999 / 0.804 | 0 / 0 |
| 0.05 (implicit) | −184.1 ± 7.0 | −244.4 ± 29.6 | +60.4 | 8/10 | 0.037 | 0.26 | 0.998 / 0.636 | 0 / 1 |
| 0.1 | −240.6 ± 25.8 | −236.6 ± 33.1 | −4.0 | 4/10 | 0.49 | 1 | 0.996 / 0.471 | 0 / 1 |
| 0.2 | −267.5 ± 28.1 | −271.5 ± 32.8 | +4.0 | 7/10 | 0.77 | 1 | 0.992 / 0.350 | 0 / 1 |
| 0.5 | −214.5 ± 25.8 | −273.9 ± 42.0 | +59.4 | 5/10 | 0.38 | 1 | 0.980 / 0.255 | 0 / 2 |

Spearman: ρ +0.080, p 0.51 (n = 70).

**LunarLander last-100 return.**

| p_cross | on | off | on − off | on > off | p | Holm | within |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (sealed) | 88.4 ± 3.1 | 93.8 ± 7.4 | −5.4 | 5/10 | 0.56 | 1 | 1.000 / 1.000 |
| 0.0125 | 80.3 ± 7.5 | 84.8 ± 6.6 | −4.5 | 4/10 | 0.77 | 1 | 0.999 / 0.653 |
| 0.025 | 81.9 ± 5.0 | 95.1 ± 5.8 | −13.3 | 3/10 | 0.11 | 0.42 | 0.998 / 0.479 |
| 0.05 (implicit) | 79.8 ± 3.4 | 88.5 ± 3.3 | −8.8 | 1/10 | 0.037 | 0.26 | 0.996 / 0.318 |
| 0.1 | 78.7 ± 5.7 | 93.8 ± 6.5 | −15.1 | 4/10 | 0.16 | 0.48 | 0.992 / 0.265 |
| 0.2 | 75.8 ± 4.1 | 88.0 ± 4.9 | −12.2 | 3/10 | 0.084 | 0.42 | 0.984 / 0.258 |
| 0.5 | 72.1 ± 3.3 | 89.4 ± 5.4 | −17.3 | 2/10 | 0.037 | 0.26 | 0.958 / 0.250 |

Spearman: ρ −0.170, p 0.16 (n = 70).

**LunarLander AUC (first 3000 episodes).**

| p_cross | on | off | on − off | on > off | p | Holm |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (sealed) | −69.5 ± 2.5 | −69.6 ± 1.8 | +0.1 | 7/10 | 0.92 | 1 |
| 0.0125 | −66.2 ± 1.8 | −66.9 ± 2.1 | +0.7 | 5/10 | 0.92 | 1 |
| 0.025 | −63.5 ± 1.8 | −62.8 ± 3.2 | −0.7 | 4/10 | 0.63 | 1 |
| 0.05 (implicit) | −64.8 ± 2.3 | −59.5 ± 3.3 | −5.3 | 2/10 | 0.027 | 0.19 |
| 0.1 | −60.6 ± 2.3 | −63.8 ± 1.6 | +3.3 | 7/10 | 0.28 | 1 |
| 0.2 | −61.5 ± 2.0 | −58.4 ± 3.1 | −3.2 | 2/10 | 0.38 | 1 |
| 0.5 | −63.4 ± 3.0 | −58.5 ± 2.5 | −4.9 | 2/10 | 0.16 | 0.96 |

Spearman: ρ −0.130, p 0.28 (n = 70).

**Pre-registered tests.** No point survives Holm on any task or metric (smallest adjusted p 0.19), and no task has a trend in p_cross. At n = 10 per point the test is weak; the readings below are development numbers.

**Predictions.**

- **Acrobot, on − off grows with p_cross: not supported.** ρ +0.08. The gain is large at 0.05 (+60.4, 8/10, unadjusted p 0.037) and 0.5 (+59.4, 5/10), and near zero at 0.1 and 0.2.
- **CartPole, on ≈ off: supported.** No difference at any point; the one large mean difference (+50.4 at 0.0125) is the single seed that fails with the map term off.
- **LunarLander, on loses AUC and keeps last-100: reversed.** AUC moves by at most 5 either way. Last-100 is lower with the map term on at every p_cross > 0 (−4.5 to −17.3).

**Reading (development numbers).**

- **The map term seals the walk on every task.** Within stays at 0.96 or above at every point with it on, against chance mixing with it off on Acrobot and LunarLander (0.26 and 0.25 at 0.5). Learned sealing replaces enforced sealing structurally.
- **Acrobot: it removes the failures, not the cost.** No Acrobot run fails with the map term on (5 with it off). Every failing off seed is rescued, by +312 to +329 (one at each of 0.05, 0.1 and 0.2, two at 0.5). But other seeds lose up to 222 (0.1: four lose 77–149; 0.2: three lose 95–222; 0.5: one loses 111), so from 0.1 up the mean stays 30–85 below sealed. Only implicit plus the map term matches sealed (−184.1 vs −184.0).
- **LunarLander: forced sealing costs final return.** With the map term off LunarLander's walk mixes at chance without loss; with it on, last-100 falls with p_cross (88.4 at 0 to 72.1 at 0.5), while the off arm stays flat. Pooled over the six p_cross > 0 points (exploratory, not pre-registered): on − off −11.8, on > off in 17/60 seeds, Wilcoxon p 0.0004.
- **At p_cross 0 the map term has no cross synapses to act on**, and on ≈ off on Acrobot and LunarLander. CartPole's −14.4 (3/10, p 0.049, Holm 0.34) is unexplained and not significant after correction.

**Best point per task, both sweeps.** CartPole: sealed with the map term off (405.4). Acrobot: sealed (−185.5 off, −184.0 on), tied with implicit plus the map term (−184.1). LunarLander last-100: off at 0.025 (95.1), with the off curve flat. The map term does not produce a better operating point than the best map-term-off wiring on any task.

**Reporting.** Not reported; the decision is the user's (§7).
