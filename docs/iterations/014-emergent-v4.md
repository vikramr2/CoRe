# Iteration 014: the emergent-walk rule on v4, four engines (pilot)

**Date:** 2026-10-27.

- **Model:** v4 flags with `--rule emergent` (`docs/derivation.md`: the walk observed from spikes, Eqs. 4, 4b, 4d, 8–11). `docs/model.md` is bumped once the reported study confirms the settings.
- **Plan:** `jiggly-chasing-sparkle` (user-approved 2026-10-26). Excitatory-only; QIF's fixed inhibitory pools stay outside W and outside the walk. LIF is a noisy LIF, τ_m 20 ms, 60 ms window, 100 ms frames, Siegert gain.
- **Compute:** the local workstation (72 cores, 40 workers, `OMP_NUM_THREADS=1`), no SLURM.
- **Agents:** `neuroscientist` (accepted with changes; CHANGELOG 2026-10-27), then `snn-expert` and a derivation agent for the code and the derivation rewrite.
- **Status:** the pilot ran (56/56 runs). Its settings are frozen into the pre-registered study `reports/emergent-v4/` (§6). Every number here is a development result: 2 seeds, 300 episodes.

## 1. What was built

The code map is in the CHANGELOG (2026-10-27, "Added") and in CLAUDE.md under `rule="emergent"`. In short:

- **Rule:** `core.observed_walk` (Eq. 8), `structural_update_emergent` (Eq. 4), `homeostasis_update_emergent` (4d), `eligibility_step_emergent` (4b). Flow under `rule="emergent"` is bit-identical to v4 (tested).
- **Counters:**
  - branching: `spiking.branching_counters`, exact (Eq. 11a);
  - QIF: Eq. 11b, gain regressed on the injected noise current;
  - LIF: `mapstdp/lif.py`, Siegert gain (11c).
- **Step sizes:** `--match-scales` sets λ and η per spiking engine so the RMS updates at initial W match flow's (50 frames, seed 5521).
- **Saving:** `--save-model` writes W at episode 0, every 250 episodes and at the end. `experiments/structure.py` compares saved models.
- **Reporting:** `report.py` runs m3 studies (`runner="m3"`, on/off `pairs`).

## 2. Pilot design

- **Arms:** task {CartPole, LunarLander} × seeds {0, 1} × λ {0.05 = on, 0 = off}, 300 episodes each.
- **Common flags:** `--rule emergent --td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5 --episodes 300`.
- **Spiking flags:** `--backend spiking --heads 16 --readout diffbound --match-scales`.
- **Groups:**

| Group | Extra flags | Question |
| --- | --- | --- |
| `flow` | `--backend flow` | reference |
| `branching` | `--engine numpy` | |
| `qif` | `--engine qif` | |
| `lif` | `--engine lif` | v4 readout as is |
| `lifbm1.6` | `--engine lif --beta-max 1.6` | readout inside BOUND_CAL_LIF's range (β_eff saturates at 2.35) |
| `lifrace` | `--engine lif --readout race --beta-kappa 0` | race readout instead |
| `branchingEbmod` (on only) | `--ebar module` | ablation: module-level baseline |
| `branchingStk` (on only) | `--step-stat kappa` | ablation: pair counts as the step |

- **Files:** commands in `experiments/results/pilot014/commands.txt`, results in `experiments/results/pilot014/runs/<task>_<group>_<on|off>/s<seed>.json`, models in `.../models/`, logs in `logs/pilot014/`.
- **Run:** 2026-10-27, 00:55 to 04:21, LunarLander queued first.

## 3. Results

AUC is the mean return over all 300 episodes; "last 100" is the mean over episodes 201–300. Both are means over the 2 seeds. D is the held-out map equation of the flow walk of W. L_obs is the held-out map equation of the engine's own observed walk. Both are in bits, at episode 0 → 300.

### CartPole

| Engine | AUC on | AUC off | Last 100 on | Last 100 off | D on | D off | L_obs on | L_obs off |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flow | 190 | 160 | 172 | 234 | 6.54 → 4.53 | 6.54 → 6.10 | = D | = D |
| branching | 218 | 111 | 321 | 71 | 6.54 → 4.41 | 6.54 → 6.12 | 7.55 → 6.75 | 7.55 → 7.12 |
| QIF | 187 | 62 | 266 | 101 | 6.54 → 4.10 | 6.54 → 5.78 | 7.40 → 6.54 | 7.40 → 6.99 |
| LIF (β ≤ 16) | 172 | 156 | 242 | 201 | 6.54 → 3.99 | 6.54 → 6.46 | 7.78 → 7.49 | 7.78 → 7.74 |
| LIF (β ≤ 1.6) | 216 | 151 | 297 | 182 | 6.54 → 3.98 | 6.54 → 6.38 | 7.78 → 7.48 | 7.78 → 7.76 |
| LIF, race | 30 | 31 | 36 | 36 | 6.54 → 4.23 | 6.54 → 6.14 | 7.78 → 7.59 | 7.78 → 7.75 |

### LunarLander

| Engine | AUC on | AUC off | Last 100 on | Last 100 off | D on | D off | L_obs on | L_obs off |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flow | −88 | −83 | −64 | −43 | 7.35 → 4.49 | 7.35 → 6.63 | = D | = D |
| branching | −72 | −56 | −73 | −29 | 7.35 → 4.42 | 7.35 → 6.86 | 8.21 → 7.02 | 8.21 → 7.50 |
| QIF | −35 | −142 | −4 | −93 | 7.35 → 3.84 | 7.35 → 7.04 | 8.23 → 6.91 | 8.23 → 8.05 |
| LIF (β ≤ 16) | −89 | −104 | −59 | −85 | 7.35 → 3.93 | 7.35 → 7.52 | 8.52 → 8.17 | 8.52 → 8.41 |
| LIF (β ≤ 1.6) | −87 | −99 | −64 | −84 | 7.35 → 3.93 | 7.35 → 7.55 | 8.52 → 8.18 | 8.52 → 8.44 |
| LIF, race | −178 | −197 | −165 | −185 | 7.35 → 4.05 | 7.35 → 6.45 | 8.52 → 8.18 | 8.52 → 8.35 |

### Ablations (branching, map term on)

| Arm | CartPole AUC | LunarLander AUC | L_obs end (CP / LL) |
| --- | --- | --- | --- |
| default (ē^obs, step c) | 218 | −72 | 6.75 / 7.02 |
| `--ebar module` | 171 | −67 | 6.77 / 7.03 |
| `--step-stat kappa` | 165 | −53 | 6.48 / 6.74 |

### Monitors (mean over the last 50 episodes, map term on)

| Engine | Causal share of κ | ρ(Λ̂^K) | M* | Silent-module frames | Clip rate | Deadline frames |
| --- | --- | --- | --- | --- | --- | --- |
| branching | 0.12 | 0.79 | 3.9 (CP), 4.8 (LL) | 0 | 0.13–0.14 | 0 |
| QIF | 0.06–0.07 | **0.98 (CP), 1.02 (LL)** | 3.8, 4.2 | 0 (CP), **0.05 (LL)** | 0.13–0.19 | 0.23–0.26 |
| LIF (β ≤ 16) | 0.01 | 0.64 | 3.5, 4.0 | 0 | 0.19–0.25 | **0.36–0.61** |
| LIF (β ≤ 1.6) | 0.01 | 0.64 | 3.5, 4.0 | 0 | 0.22–0.28 | 0.01–0.02 |

### Runtime (40 concurrent runs)

| Engine | ms/frame CP | ms/frame LL | Hours per 300-episode run, CP / LL |
| --- | --- | --- | --- |
| flow | 3 | 5 | 0.04 / 0.2 |
| branching | 53 | 56–68 | 0.5–1.2 / 2.4–3.4 |
| QIF | 87–112 | 65–102 | 0.6–1.4 / 1.8–2.6 |
| LIF | 66–69 | 85–89 | 0.8–1.4 / 2.1–2.3 |

### Structure (`python -m experiments.structure`, final W)

- **What the map term does on every engine and both tasks:** it removes nearly all non-routing cross-module weight (A↔A, A↔L, L↔L and →C together fall from 15–37% of total weight to 0.3–0.6% on the spiking engines and 1.0–1.3% on flow). Within-module weight rises from 0.22–0.49 to 0.59–0.64. The rest is the exempt controller routing.
- **Engine-free L of W's own walk:**
  - CartPole: 3.8–4.4 bits on, 5.8–6.4 off.
  - LunarLander: 3.9–4.5 on, 7.0–7.6 off.
- **Output:** the per-module tables and on − off maps are in `experiments/results/pilot014/structure.md` and `structure_*.png`.

## 4. Reading

- **Description length:** the map term lowers it on every engine and task, both D and L_obs, with no overlap between the 2 seeds of each arm.
  - On LIF the drop in L_obs is small in bits (about 0.3), because about 99% of LIF's pair counts are chance (causal share 0.01).
  - The engine-free D falls as much on LIF as on the other engines.
- **Performance, mixed at 300 episodes and 2 seeds:**
  - The map term is ahead in AUC for branching, QIF and LIF on CartPole, and for QIF and LIF on LunarLander.
  - For flow (both tasks) and branching on LunarLander it is behind, or ahead in AUC but behind over the last 100 episodes.
  - The pilot is too short and too small to call any cell. The reported study tests every cell, two-sided.
- **QIF sits at the stability edge with the map term on** (ρ ≈ 1.0, 5% silent-module frames on LunarLander). The column bound of §7 does not cover QIF's fixed pools. The study reports ρ and does not gate on it.
- **Ablations:** neither beats the derivation's default on both tasks (2 seeds), so the defaults stay. `--step-stat kappa` lowers L_obs further on branching, but loses CartPole AUC.

## 5. Decisions (frozen for the study)

1. **LIF readout: diffbound with `--beta-max 1.6`.** With v4's cap of 16, 36–61% of LIF frames hit the deadline: β runs past the 2.35 where LIF's calibrated bound saturates (BOUND_CAL_LIF), so decisions fall back to the deadline. At 1.6, 1–3% do. This decision rests on that mechanism, not on the returns. Race fails on LIF: CartPole about 30, near random.
2. **Other engines keep v4's readout** (diffbound, β ≤ 16). QIF's deadline rate (0.13–0.26) was not tuned. Tuning it now would be tuning on returns.
3. **Rule defaults stay:** `--ebar obs`, `--step-stat c`, `--obs-tau 0` (per-frame statistics).
4. **Episodes:**
   - CartPole 1500 and LunarLander 1000, as planned.
   - **Projected cost:** about 8–9 h per spiking CartPole run and 10–12 h per spiking LunarLander run under 40-way load, so about 2.5 days for 240 spiking runs on 40 workers. Flow is about 2 h in total.
5. **Seeds:** 2000–2019, fresh. No earlier run used seeds at or above 2000.

## 6. Reported study

`reports/emergent-v4/README.md` (pre-registered draft), declared as `STUDIES["emergent-v4"]` in `experiments/report.py`. It is on hold: the user asked for the issues below to be resolved first. The results go there, not here.

## 7. After the pilot: the action collapse (diagnosis)

**Observation:** episodes of 14 steps or fewer (the agent pushing one way) in the last 100 CartPole episodes:

| Group | Seed 0 | Seed 1 |
| --- | --- | --- |
| branching off | 43% | 100% |
| QIF off | 44% | 56% |
| branching on | 2% | 0% |
| QIF on | 0% | 0% |
| flow, on or off | 0–1% | 0–1% |

**What the collapse is** (scripts in the session scratchpad, not in the repo):

1. **It is in the weights, not the readout.** On the final W of the collapsed branching runs, the flow policy also picks action 0 on 92–100% of the 100 held-out stimuli (spiking readout: 97–98%).
   - Controller→A1 weight is 2.7–9× controller→A2 (12.1 vs 4.5, and 8.5 vs 0.9).
   - In the healthy on-arm run checked (branching on, seed 0), the two are close: 7.1 vs 6.9.
2. **It is transient.** Collapses come in blocks of 20–60 episodes and then recover, for example episodes 218–280 in QIF off seed 0.
   - The training return of a collapsed W can still look fine. Branching off seed 0 averaged 295 over its last 30 training episodes, but its final W (saved during a fresh collapse at episode 300) scores 10 with learning frozen.
   - **Frozen-W returns** (30 episodes, learning off) are therefore reported alongside training returns from now on.
3. **The recurrent loops do not carry it.** On the collapsed W, the flow policy stays collapsed (P(a0) 0.79–1.00) if any of these is removed:
   - the feedback into the controller (→C);
   - the cross-module L/A loops;
   - everything except C→A and the within-module weights.
   Equalising the C→A totals leaves it collapsed too. The bias lives in the action modules' own weights and in how their routing developed.
4. **With the map term, every action module ends with the same self-retention:** within-module offspring 0.80 against 0.36–0.66 without it. All of an action module's output then stays inside it, so action modules compete only through their controller input. Without the map term, one module's retention can drift above the other's, and the task rule amplifies the difference.

**Consequence for the claims:** on spiking engines, part of the map term's benefit is that it seals the modules, and sealing stabilises the action competition. Sealing is structure: it is the minimum-description structure for the given partition, which the map term is built to produce. The open question is whether the benefit comes from that structure itself or from the process that learns it, since a designer who knows the partition could wire it in directly.

**The control:** the same agent with λ = 0 on a network built sealed, `--p-cross 0`. Cross-module synapses outside the routing pairs are absent from the start; the SBM draws are unchanged, so only the mask differs. The two arms reach the same target structure by different routes, learned from spikes or prescribed at construction. Readings:
- **Sealed matches on:** the benefit is the structure, and the map term is a local rule that finds it. This supports the structural hypothesis, but not that learning the structure beats prescribing it when the partition is known.
- **On beats sealed:** the map term contributes beyond removing cross-module links. Candidates: within-module reshaping (within-module synapses grow with the neuron's exit share), M* weighting, and cross-module links that fade during training rather than being absent from episode 0.
- **Sealed beats on:** having the structure from the first episode matters more than learning it.
- Routing pairs are exempt in both arms, so if sealing alone removes the collapse, the cause is leakage between action modules, not the controller→action weights.
- P2 (description length) is unaffected: the sealed arm starts at low L by construction.

## 8. Runtime

**Profile:** single process, CartPole, 8 episodes, `--match-scales`. ms/frame includes evaluation and profiler overhead.

| Engine | Before | After | Change |
| --- | --- | --- | --- |
| branching | 22.6 | 5.8 | `branching_counters` loops over synapses j → i only, compiled with numba. Equal to the numpy version up to summation order (6 × 10⁻¹⁵ relative); 8.4 → 1.3 ms (CartPole) and 12.2 → 2.0 ms (LunarLander) per frame |
| LIF | 20.4 | 6.3 | numba loop with in-loop noise (`LIFParams.kernel = "nbf"`) and numba `pair_counts` (identical integers); frame 19 → 2.6 ms |
| QIF | 15.5 (72 numba threads) | 20.3 (1 thread) | none yet: the RK2 kernel is the cost (about 10 ms per frame per thread) |

- **Why `nbf`:** numba's Gaussian draws alone cost 10 ms per LIF frame, so `nbf` draws variance-matched Irwin–Hall(4) noise from a xorshift64* generator. The membrane averages the noise over about 40 steps (τ_m / dt), so only its variance matters.
  - Checked against the numpy reference on the v4 CartPole network (200 frames, H = 16): module rates within 0.4%; correlations of per-neuron rates 0.999, κ 0.9985 and c 0.9999, the same as two Gaussian streams.
  - `python -m mapstdp.lif` passes on it: Siegert rate and gain within 1% of simulation.
  - The random streams differ, so LIF runs are not bit-identical to the pilot's.
- **Numba threads:** QIF's kernel is `parallel=True`. Under the pilot's 40 processes each one could start 72 numba threads (`OMP_NUM_THREADS` does not cap numba), which is part of the pilot's 3× slowdown under load. `report.py` now sets `NUMBA_NUM_THREADS=1`.

**Worker count.** The benchmark ran concurrent LunarLander jobs (4 episodes each, engines cycled, `NUMBA_NUM_THREADS=1`). Throughput is the sum of 1000 / ms-per-frame over the jobs.

| Workers | Frames/s | ms/frame branching | ms/frame LIF | ms/frame QIF |
| --- | --- | --- | --- | --- |
| 24 | 1929 | 10.8 | 8.4 | 34.0 |
| 36 | 2767 | 11.2 | 8.8 | 35.8 |
| 48 | 3405 | 11.8 | 10.3 | 36.1 |
| 60 | 3726 | 13.0 | 12.7 | 36.8 |
| 72 | 3309 | 18.0 | 18.2 | 37.7 |

- Throughput peaks at about 60 workers. 48 gives 91% of the peak and leaves room for other work.
- Under load, frames are 5–8× cheaper than in the pilot (branching 53–68, LIF 85, QIF 65–112 ms/frame there).
- QIF is now the bottleneck: it is about 3× the other spiking engines.

## 9. Second pilot (development, `experiments/results/pilot014b/`)

**Launched** 2026-10-27, 116 runs on 48 workers (`NUMBA_NUM_THREADS=1`), longest first. Commands are in `commands.txt`, logs in `logs/pilot014b/`.

**Base settings:**
- Seeds 3–5, 500 episodes per task.
- Common flags: `--rule emergent --td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --eta 5`.
- Spiking engines: `--readout diffbound --match-scales`, 16 heads; LIF β ≤ 1.6, the others ≤ 16.
- Every run records `frozen_returns` (20 episodes with W frozen, new in `m2.run`).

| Groups | Question |
| --- | --- |
| `<task>_<engine>_{on, off, sealed}` (all four engines) | sealed control (§7): λ = 0 on `--p-cross 0` |
| `<task>_lifw20_{on, off}` | LIF pairing window 20 ms instead of 60 ms (causal share of κ) |
| `<task>_qifh2_{on, off}` | QIF homeostasis ×2 (`--q-eps-h-scale 2`): ρ below 1? |
| `<task>_{branching, lif}h8_on` | 8 heads instead of 16 |
| `LunarLander_{flow, branching}3000_{on, off}` (seeds 3–4) | LunarLander at 3000 episodes: cost and learning past 1000 |

**Calibration for 8 heads:** `BOUND_CAL[8]` and `BOUND_CAL_LIF[8]` come from `experiments.bound_cal` on the pilot014 flow CartPole models (seeds 0–1).
- The same models at 16 heads reproduce `BOUND_CAL[16]` within 7%.
- The LunarLander models were left out: their policies are nearly deterministic, so the maximum-likelihood β_eff diverges (26–58).
- The new LIF table at 16 heads (0.37–3.17) differs from `BOUND_CAL_LIF[16]` (0.47–2.35, fitted on Acrobot, MountainCar and CartPole). The 16-head table is kept, so the pilot stays comparable.


## 10. Second pilot: results and decisions (2026-10-27)

Development numbers: 3 seeds (3–5), 500 episodes, means over seeds. AUC is the mean return over all episodes. "Frozen" is the mean of the 20 frozen-W episodes. L_obs is the held-out map equation of the engine's observed walk at the last evaluation. The summary script was a scratch script; the raw runs are in `experiments/results/pilot014b/runs/`.

**CartPole**

| Engine | Arm | AUC | Last 100 | Frozen | L_obs |
| --- | --- | --- | --- | --- | --- |
| flow | off | 208 | 359 | 497 | 6.24 |
| flow | on | 205 | 319 | 373 | 4.54 |
| flow | sealed | 196 | 310 | 316 | 4.34 |
| branching | off | 94 | 64 | 51 | 7.33 |
| branching | on | 204 | 218 | 279 | 6.71 |
| branching | sealed | 199 | 272 | 202 | 6.77 |
| QIF | off | 79 | 63 | 44 | 7.05 |
| QIF | on | 182 | 193 | 206 | 6.56 |
| QIF | sealed | 174 | 209 | 155 | 6.81 |
| LIF | off | 174 | 162 | 253 | 7.78 |
| LIF | on | 235 | 283 | 229 | 7.50 |
| LIF | sealed | 214 | 283 | 273 | 7.70 |

**Reading (CartPole):**
- The map term raises AUC on every spiking engine (branching 94 → 204, QIF 79 → 182, LIF 174 → 235). On flow it changes neither AUC nor last-100.
- Off-arm branching and QIF runs still collapse onto one action (frozen returns of 9–24 on some seeds).
- Sealed matches on for AUC on every engine. In the terms of §7: the task benefit is the structure, and the map term is a local rule that finds it.
- On the spiking engines, on reaches lower L_obs than sealed (branching 6.71 vs 6.77, QIF 6.56 vs 6.81, LIF 7.50 vs 7.70). The map term shapes the walk beyond removing cross-module links.
- No engine reaches a trailing-100 mean of 475 in 500 episodes. v4 flow needs a median of about 500 episodes to get there (40 earlier seeds, range 297–1319).

**Settings:**
- **8 heads:** same returns as 16 at about half the cost per frame. CartPole branching AUC 212 vs 204 at 4.7 vs 10.0 ms; LIF 231 vs 235 at 5.3 vs 7.6 ms. **Adopted.**
- **LIF, 20 ms window:** causal share 1.1% → 3.0%; returns unchanged within the seed spread (AUC 242 vs 235). Harmless; not adopted for the study, which keeps 60 ms.
- **QIF, homeostasis ×2:** ρ 1.02 → 0.94, no gain in return, and one collapsed seed. **Dropped.**

**LunarLander (completed groups only):**
- After 500 episodes every engine is still near or below 0. The map term lowers L_obs on all of them.
- On return, the map term helps flow (AUC −64 → −37) and QIF (−136 → −36), and is mixed on branching and LIF.
- Flow at 3000 episodes (seeds 3–4): last-100 return 148 with the map term on, against 29 off.

**LunarLander cancelled (user, 2026-10-27).** Seven runs were stopped with checkpoints in place:
- both 3000-episode branching pairs;
- `qif_sealed` s3;
- `qifh2_on` s3–s4.

The user's reasoning: this iteration asks whether, in a minimal v4 backbone, the map term improves performance and lowers description length, and CartPole answers that on all four engines. Performance at the level of the TD-STDP baseline is the question for the TD-STDP backbone study (iteration 015), which takes LunarLander.

**Next:**
- The reported study `emergent-v4` becomes CartPole only: four engines × on, off and sealed, 8 heads, 20 fresh seeds, 1500 episodes.
- It runs from a frozen worktree, so the code can keep changing for iteration 015.

## 11. Reported study `emergent-v4`: results (2026-10-27)

All 240 runs completed (0 failed) in the worktree `../map-stdp-ev4`, branch `emergent-v4-study`, git `086996f` with an empty patch. Final numbers are in `reports/emergent-v4/summary.md`, `pairs.csv` and `structure.md` on that branch; this section only reads them.

**P1, performance (AUC, on vs off, Holm over 4 engines):** supported on 3 of 4 engines.

| Engine | AUC on vs off | Holm p | Verdict |
| --- | --- | --- | --- |
| flow | 357 vs 336 (+21) | 0.17 | not supported |
| branching | 204 vs 117 (+87) | 2.7e-7 | supported |
| QIF | 168 vs 103 (+65) | 2.7e-7 | supported |
| LIF | 276 vs 194 (+82) | 2.7e-7 | supported |

**P2, description length (L_obs at episode 1500, on vs off):** supported on all 4 engines (Holm p 2.7e-7 each). The drops are −1.47 bits (flow), −0.37 (branching), −0.63 (QIF) and −0.30 (LIF).

**S, learned vs prescribed structure (on vs sealed):**

| Engine | AUC on vs sealed (Holm p) | Reading | L_obs on − sealed |
| --- | --- | --- | --- |
| flow | 357 vs 388 (1.0) | on ≈ sealed | +0.20 bits (sealed lower) |
| branching | 204 vs 201 (1.0) | on ≈ sealed | −0.06 |
| QIF | 168 vs 196 (0.0023) | **sealed > on** | −0.32 |
| LIF | 276 vs 271 (0.6) | on ≈ sealed | −0.22 |

- On three engines, the task benefit is the structure, and the map term is a local rule that finds it.
- On QIF, having the structure from the first episode does better than learning it.
- On the spiking engines the map term reaches a lower L_obs than sealing (Holm p ≤ 6e-5). It shapes the walk beyond cutting links.

**Secondary:**
- **Solving:** only flow reaches a trailing-100 mean of 475 (on 15/20, off 20/20, sealed 20/20), plus branching sealed 3/20.
- **Frozen-W return, on vs off:** higher on branching (+72, Holm 0.02) and LIF (+111, Holm 0.0018); not significant on flow (+51, Holm 0.063) and QIF (Holm 0.19).
- **Structure, final W:**
  - The map term brings the within-module share to 0.63–0.64 on every engine (sealed 0.64; off 0.40–0.48).
  - It removes A↔L, L↔L and →C weight (off: 6–14% into the controller).
  - The engine-free L_W of W's own walk with the map term on is 3.57 (QIF) and 3.80 (LIF), below sealed (4.18–4.20). On these engines the map term also reshapes weights *within* modules.
- **The task term alone also structures W.** With λ = 0, latent→action weight goes to 0 and A↔A stays at ≤ 0.6%. Off is implicit modularity, not none.

The three regimes (implicit, learned, enforced) are compared again on LunarLander in iteration 015.

## 12. Exploratory, not pre-registered: description length vs an SOP energy proxy (2026-10-27)

**Question (user):** does description length track an energy proxy based on synaptic operations?

**Data:** the saved `emergent-v4` models of the three spiking engines, W frozen, at 7 snapshots per run (episodes 0–1500 every 250), 50 held-out stimuli per snapshot (`m2.held_out_obs`). Scratch scripts, development only.

**Measures:**
- L_obs as `evaluate_emergent` computes it (it reproduces the logged value: 7.433 vs 7.430 bits on LIF on, seed 2000);
- spikes per head and frame;
- SOPs = Σ_j n_j × (synapses of j);
- effective SOPs, counting only synapses above twice the weight floor;
- the cross-module share of effective SOPs, routing excluded.

**Spearman ρ of L_obs with each measure:**

| Engine | Effective SOPs, across 60 final networks | Effective SOPs, over training (median per-run ρ; share of runs with ρ > 0) | Spikes, across final networks | Effective SOPs, map on vs off |
| --- | --- | --- | --- | --- |
| branching | +0.61 (p 3e-7) | +0.54; 85% | −0.48 | −19% |
| QIF | +0.95 (p 2e-31) | +0.68; 90% | +0.96 | −64% |
| LIF | +0.91 (p 2e-24) | +0.89; 87% | +0.10 | −53% |

**Reading:**
- **Description length tracks connectivity-weighted activity, not spike count.** Effective SOPs correlate with L_obs on every engine, both across networks and along training.
- **Raw spikes do not, except on QIF.** On LIF, spikes rise during training while L_obs falls (median per-run ρ −0.89).
- **SOPs over all existing synapses correlate weakly** (ρ +0.33 to +0.53), because the mask is fixed.
- **Within an arm the correlation weakens,** except on QIF (on +0.77, sealed +0.72). Much of the cross-network correlation is the arm effect: the map term drives cross-module weights to the floor, which both lowers L and removes effective synapses.
