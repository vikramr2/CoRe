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

Study `cross-sweep` in `experiments/report.py`:

- **Seeds:** fresh, 8000–8019 (CartPole), 8100–8119 (Acrobot), 8200–8219 (LunarLander).
- **Arms per task:** `base_ck` (reference), `cols_sealed`, `cols_lam0` (p 0.05) and `cols_px<p*>`.
- **Pairs:**
  - family `primary`: p* vs sealed and p* vs implicit;
  - family `secondary`: each arm vs `base_ck`.
- **Statistics:** Holm per family, as the race studies.
- **Size:** at most 240 runs. Final numbers only from `reports/cross-sweep/`.

## 5. Runs

(Filled in when the dev stage runs.)
