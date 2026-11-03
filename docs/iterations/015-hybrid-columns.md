# Iteration 015: Map-STDP columns in the TD-STDP backbone (no Adam)

**Question (user, 2026-10-27):** inside the TD-STDP actor-critic of Chung & Kozma 2020 (iteration 013, `reports/tdstdp-baseline`), does modular organisation keep or improve performance without Adam? Ideally it replaces Adam. Beating the no-Adam baseline is already a strong result.

**Headline:** LunarLander without Adam, where the BindsNET baseline solved 0 of 10 runs (last-100 return 48.3).

**Design sources:** `docs/derivation.md` §8.1 and the hybrid diagram in `docs/derivation-guide-emergent.pdf` (p.12). The plan is `~/.claude/plans/jiggly-chasing-sparkle.md`.

All numbers here are development results (seeds 0–5, `experiments/results/pilot015/`).

## 1. Design (user decisions, 2026-10-27)

- **Controller C:** the Fourier input population. Routing C→A and C→L is CK's input→actor W, learned by TD alone.
- **Engine:** noisy LIF (`--engine nlif`, `lif.simulate_ext`, dt 1 ms). Every arm, including the no-recurrence baseline, runs on it. BindsNET's deterministic LIF is a descriptive reference only.
- **Modules:**
  - **Action columns.** Each action's actor E population is one module, with recurrent E→E synapses R between all actor E posts. The map term acts on R only.
  - **One extra arm adds 2 latent modules.** Input→latent W is frozen at init (a random expansion), and latent→action R learns.
- **Readout:** CK's softmax over column rates. Read as a first-spike exponential race of escape-noise decision units with constant hazards, it is exactly softmax. This is an abstraction [Yellott 1977].
  - Its local score is I[k = a] − Λ_k(t*). Its expectation given the winner is A_k − p_k, which the rule uses.
  - Monte Carlo check (400k races, K = 2 and 4): choice frequency within 0.0007 of softmax; E[score | a] within 0.0013 of A − p. The local score has 1.05–2.3× the variance of A − p.
- **Scope:** no Adam, CartPole and LunarLander. The step rule matches the baseline: preset η on CartPole, `--eta-cal` on LunarLander.
- **Neuroscientist review, accepted:**
  - loop gain τ_m / (1 − max Re λ) ≤ 2 frames, and χ_rec·G ≤ 0.5;
  - action autocorrelation is logged;
  - h̄_j follows from output normalisation;
  - latents frozen at the input;
  - the column tag is modulatory only.

## 2. Step 1: the nlif engine

- **Kernel:** `lif.simulate_ext` takes an external current from the input spikes, plus recurrent kicks on an index set.
- **Self-check (h1):** without noise or background, and with CK's τ 100 ms and 13 mV threshold, the kernel reproduces `simulate_bindsnet` exactly.
- **Rate matching:** s_in is bisected so the initial post rate equals BindsNET's: CartPole 140 Hz, LunarLander 365 Hz. This needs τ_ref = 0, as in BindsNET.
- **CK's rate filter** is τ_n = 20 ms. The 100 ms is BindsNET's membrane τ.

**Pilot A** (`pilot015/A/`, CK-matched rates, no recurrence). CartPole, 5 seeds × 400 episodes:

| Engine | Solved | T_s | AUC |
| --- | --- | --- | --- |
| nlif | 5/5 | 154–191 | 393–453 |
| BindsNET lif | 5/5 | 150–250 | 381–446 |

LunarLander, 3 seeds × 3000 pooled episodes, `--eta-cal`. Neither engine solves on any seed.

| Engine | Last-100 return | AUC | Best 100-episode mean |
| --- | --- | --- | --- |
| nlif | −6, −29, −8 | −36, −45, −54 | 69, 30, 26 |
| BindsNET lif | 29, 31, 0 | −28, −35, −29 | 40, 34, 58 |

In the CK-matched regime, nlif is a little behind BindsNET on LunarLander, within the large seed spread. The study baseline at 10000 episodes reached 48.3.

## 3. Step 2: recurrent columns and the map term

- **Recurrent weights:** R is SBM-initialised. Kicks are b = χ_rec R/d_j (output-normalised).
- **Walk statistics:** pair counts over a 20 ms window. The counter c is τ b G n, per head and per window. Statistics frames are 5 windows, smoothed by an EMA.
- **Updates:**
  - the structural term (Eq. 4) and homeostasis, once per statistics frame;
  - the task term on R, once per 20 ms block. The counter is paired with the next window's CK score; the eligibility is kept per head.
- **Calibration:** η_rec and λ come from `calibrate_rec`, a pre-set RMS match.
- **Gain:** Siegert fails in the mean-driven CK regime, so the gain comes from the kernel's exact Markov transfer curve (`lif.gain_markov`).
  - Counter vs replay truth, synapse correlation: 0.974 on CartPole, 0.935 on LunarLander.
  - Credit Monte Carlo, E[cψ] vs R ⊙ ∂J/∂R: cosine 0.976.
- **Smoke (CartPole, seed 0):** the map term raises the within-module weight share from 0.92 to 0.98 (0.92 with λ = 0). Max Re λ ≈ χG ≈ 0.13–0.25.
- **Problem:** on LunarLander most R posts lock at 500 Hz (τ_ref 0, dt 1 ms), where the gain is about 0. R barely learns there.

## 4. Step 3: the low-rate regime (user decision, 2026-10-27)

`--rate-target 40` sets:
- τ_ref 2 ms (`lif.DT1R`);
- s_in so the initial rate is 40 Hz;
- α_a and α_c multiplied by k = r_CK / 40;
- η on CartPole multiplied by RMS(g_CK) / RMS(g_low) (`calibrate_low`, pre-set).

40 Hz was chosen before any return was seen: it is near the gain peak, 12% of the refractory ceiling, and 0.8 spikes per neuron per 20 ms window.

- **LunarLander at init:** every R post has a gain of about 37 Hz/θ and none sits at the ceiling. The counter correlates 0.887 with replay truth (noise ceiling about 0.88).
- **CartPole baseline, low rate** (`pilot015/A2/`, 5 seeds × 400 episodes): every run reaches a last-100 return of 500, but it is slower than the CK-matched regime.

| Seed | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| T_s | 252 | – (last-100 499.5) | 260 | 325 | 370 |
| AUC | 399 | 358 | 342 | 347 | 322 |

The likely cause is column-rate noise: CV 0.093 at 40 Hz against 0.029 at 143 Hz, scaled by k.

**Decision (user):** keep the pre-set 40 Hz for both tasks and report the slower floor. All arms share the regime, so the comparison stays matched. The BindsNET numbers are reported alongside.

## 5. Pilot B

`pilot015/B/commands.txt` holds 27 runs:
- **CartPole cols:** χ_rec ∈ {0.15, 0.3} × (λ = 0, or λ on with r ∈ {0.3, 1, 3}), seeds 0–2, 400 episodes.
- **LunarLander low-rate baseline:** seeds 0–2, 3000 episodes, `--eta-cal`.

χ_rec and r are frozen on stability and structure only, never on return:
- max Re λ, τ_loop ≤ 40 ms, χG ≤ 0.5;
- within-module share with λ on above λ = 0;
- cost.

**Results, CartPole** (3 seeds × 400 episodes, 40 Hz):

| χ_rec | Arm | Solved | T_s | AUC | Max Re λ | τ_loop | Causal share | Within at 10 / 50 / 200 / 400 episodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.15 | λ = 0 | 3/3 | 220, 316, 304 | 358 | 0.13 | 23 ms | 0.6% | 0.92 / 0.92 / 1.00 / 1.00 |
| 0.15 | r 0.3, 1, 3 | 9/9 | 207–272 | 357–364 | 0.13 | 23 ms | 0.6% | 0.98 / 1.00 / 1.00 / 1.00 (r 3) |
| 0.3 | λ = 0 | 3/3 | 235, 233, 259 | 354 | 0.27 | 28 ms | 1.1% | 0.92 / 0.92 / 1.00 / 1.00 |
| 0.3 | r 0.3, 1, 3 | 9/9 | 224–305 | 336–367 | 0.27 | 28 ms | 1.0–1.3% | 0.95 / 1.00 / 1.00 / 1.00 (r 1) |

- **Gates:** every setting passes the stability gates. Action repeats are below chance (repeat − chance ≈ −0.2), so there is no perseveration.
- **The task term seals the columns on its own.** With λ = 0 the within-module share rises from 0.92 to 0.999 by about episode 200, as learning takes off. Cross-column excitation lowers the chosen action's logit margin, so the task gradient removes it.
- **The map term reaches the same structure earlier:** about episode 50 at r = 1, and 25 at r = 3.
- **Returns:** CartPole returns are the same with and without columns, and with and without the map term. They also match the low-rate baseline (AUC 322–399).

**Frozen (structure and stability only):** χ_rec = 0.3 and r = 1.
- χ_rec 0.3 doubles the causal share, so the walk signal is stronger, and stays inside every gate.
- r = 1 is the plan's pre-set default. Every r passes.

**LunarLander low-rate baseline** (3 seeds × 3000 episodes, `--eta-cal`):

| Seed | Last-100 return | AUC | Best 100-episode mean |
| --- | --- | --- | --- |
| 0 | 38 | −48 | 50 |
| 1 | 24 | −53 | 66 |
| 2 | 17 | −57 | 31 |

- No seed solves.
- This floor is at or above CK-matched nlif and level with BindsNET, at 0.4 h per run.

## 6. Pilot C: LunarLander columns and latents

`pilot015/C/`: χ_rec 0.3, seeds 0–2, 3000 episodes, `--eta-cal --rate-target 40`. The base rows are pilot B's low-rate baseline (same seeds, same s_in and α).

| Arm | Last-100 return | Mean | AUC | Best 100-episode mean | Within share at 10 / 200 / 1000 / 3000 episodes |
| --- | --- | --- | --- | --- | --- |
| base (R = 0) | 38, 24, 17 | 26 | −48, −53, −57 | 50, 66, 31 | – |
| cols λ = 0 | 48, 55, 54 | 52 | −40, −45, −55 | 52, 59, 55 | 0.76 / 0.76 / 0.72 / 0.46 |
| cols r = 1 | 15, 21, 35 | 24 | −43, −43, −47 | 35, 34, 62 | 0.81 / 0.99 / 1.00 / 1.00 |
| cols r = 3 | 5, 37, 36 | 26 | −58, −39, −52 | 29, 67, 45 | 0.87 / 1.00 / 1.00 / 1.00 |
| lat r = 1 | 31, 49, 13 | 31 | −46, −42, −49 | 42, 67, 31 | 0.72 / 0.98 / 0.99 / 0.99 |

No run solves.

**Stability gates:** every run passes.
- Max Re λ ≤ 0.25, τ_loop ≤ 27 ms, χG ≤ 0.25.
- Mean rate 51–56 Hz; no silent modules.
- Repeat − chance ≈ −0.08: no perseveration.
- Cost: 0.6–0.9 h per run, 1.7–2.2× the baseline's 0.4 h (3.2–4.4 ms per frame against 2.1).

**The task term does not seal the columns on LunarLander; it builds a hub.**
- With λ = 0 the within-module share falls from 0.77 to 0.46.
- Final R module table (mean of 3 seeds, share of total R): the main-engine column receives 0.73 of all R weight: 0.25 from itself and 0.12–0.19 from each other column. Do-nothing keeps 0.13 within and receives 0.04 from the side engines. The side-engine columns keep 0.04–0.05 within.
- So the task gradient uses R as a pathway from any column's activity to the main engine, a learned bias toward firing the main engine. On CartPole (2 actions) cross-column excitation only narrows the margin, so the same rule seals there (§5).
- The map term forbids exactly this pathway: with r = 1 or 3 the columns seal by about episode 200 (cross-column R weight below 0.1%), and returns fall back to the baseline.

**Reading (3 seeds, development):** recurrent columns without the map term beat the baseline on all three seeds (smallest columns run 48 vs largest base run 38). The map term on top keeps baseline performance but removes the gain. Three seeds give a Mann–Whitney p of 0.1 at best, so this is a hypothesis for the study, not a result.

**L_obs is not informative here.** The causal share is 0.1–0.2%, so the observed walk is almost all chance pairs, and L_obs moves by less than 0.04 bits in every arm (8.23 → 8.20 with λ = 0, → 8.22 with the map term).

**Latent arm (pre-set go criteria, plan Phase 4 C):**

| Criterion | Result | Pass |
| --- | --- | --- |
| runs complete | 3/3 | yes |
| max Re λ < 0.9 (tightened: ≤ 0.5) | 0.25 | yes |
| each latent module > 1 Hz in > 50% of frames | 100% of frames, 48–50 Hz | yes |
| latent→action synapses at w_min in < 50% by episode 200 | 0% at 200, 14–22% at 3000 | yes |

The arm passes on the letter of the criteria, so it stays in the study as pre-set. **But the latents are close to disconnected:** latent→action carries 0.25 of the R weight at init, 0.02 by episode 200 and 0.004 at the end. The map term removes latent→action as cross-module flow, and the barrier (`la_beta` 0.3) does not hold it. Returns are level with cols r = 1.

## 7. Reported study `hybrid-columns` (launched 2026-10-27 20:02 EDT)

- **Pre-registration:** `reports/hybrid-columns/README.md`, on branch `hybrid-columns-study`.
- **Code:** git `0d2d851` with an empty patch, run from the worktree `../map-stdp-hc`.
- **Design:** 2 tasks × 5 arms × seeds 3000–3019 = 200 runs, 64 workers.
- **Command:** `python -m experiments.report run hybrid-columns --workers 64` (runner PID 3216629, log `../map-stdp-hc/logs/hybrid-columns.log`).
- **Before launch:**
  - `cols_sealed` was added: `--rec-seal`, plus the `calibrate_rec` fix for sealed R.
  - Smoke runs on seed 0: CartPole solved (T_s 234); LunarLander stable over 300 episodes; R unchanged in both.
  - `pytest -q tests`: 42 passed in the worktree.

## 8. Study results (completed 2026-10-28 10:00 EDT)

- 200 of 200 runs, 0 failed, on branch `hybrid-columns-study` (git `0d2d851`, empty patch).
- `summary.md` and `pairs.csv` were written by `experiments.report` in the worktree. They reach `main` only by merging that branch.
- No LunarLander run solved (0/100; all ran to the 10000-episode cap).

**Headline verdicts** (Δ = 25; Hodges–Lehmann shift vs `base` [95% CI], Holm-adjusted p):

| Contrast | CartPole AUC | Verdict | LunarLander last-100 | Verdict |
| --- | --- | --- | --- | --- |
| `cols_lam0` vs base | −5.2 [−27.6, +10.2], p 1 | inconclusive (CI crosses −Δ) | −6.9 [−16.5, +0.8], p 0.43 | non-inferior |
| `cols_on` vs base | −15.6 [−32.0, +10.1], p 1 | inconclusive (CI crosses −Δ) | −2.0 [−14.4, +11.2], p 1 | non-inferior |
| `cols_sealed` vs base | +0.4 [−18.1, +18.8], p 1 | non-inferior | −0.5 [−9.6, +6.9], p 1 | non-inferior |

- No regime improves the primary metric, and none is worse.
- **Secondary contrasts:** no difference between regimes on either task (all Holm p ≥ 0.23).
  - `lat_on` vs `cols_on` on CartPole: AUC +28 [+1.7, +45], p 0.039 unadjusted, Holm 0.23.

**Secondary metrics (pre-registered, not tested for the claims):**
- **LunarLander AUC over the first 3000 episodes:** every column arm learns faster than base.
  - `cols_lam0` +10.1 [+6.4, +13.4] (Holm 8e-5);
  - `cols_on` +10.7 [+6.9, +14.1] (Holm 2.4e-5);
  - `cols_sealed` +8.8 [+4.7, +12.9] (Holm 0.0014).
  
  T_f: 1100–1290 with columns vs 1546 for base. The early lead seen in pilot C and at mid-run is real, but base catches up by 10000 episodes.
  - **Qualified later (iteration 016 §10–11):** most of this early lead is a starting-gain effect. The column arms started at a higher rate and a sharper policy; with the starts matched (`--rate-match-rec --ent-match-rec`, dev seeds 10–15) the gain fell to +3 to +6 and was not significant.
- **CartPole solved:** 20/20 for every column arm (`lat_on` 19/20) vs 16/20 for base. Final last-100: 499.7–500.0 vs 491.4.
- **Module structure of R** (`rec.RK`, share of R's weight within columns, from first to last snapshot, mean over 20 seeds):

  | Arm | CartPole | LunarLander |
  | --- | --- | --- |
  | `cols_lam0` | 0.909 → 0.999 | 0.769 → **0.399** |
  | `cols_on` | 0.909 → 0.999 | 0.769 → 0.996 |
  | `cols_sealed` | 1 → 1 | 1 → 1 |

  - **CartPole:** the task term seals the columns on its own, as the map term does.
  - **LunarLander:** the task term builds a hub. 60% of R's weight crosses columns, and 97% of that goes into the main-engine column, about equally from the no-op, left and right columns (sd of the within share 0.02 over seeds). The map term prunes this hub (within share 0.996).
  - **Performance converges although the structure diverges:** a hub (`cols_lam0`), modular (`cols_on`) and sealed (`cols_sealed`) reach the same final return and the same early speed-up (secondary contrasts all null).
  - **`lat_on`:** the within share over the whole table, latents included, rises from 0.44 to 0.64 (LunarLander) and from 0.38 to 0.52 (CartPole).

**What the sealed and map-term arms test (expert review, 2026-10-28; iteration 016 §13):** in the sealed arm, and in `cols_on` once sealed, the one-hop credit on R is zero and the map term barely acts within modules, so R is a fixed random within-column amplifier: these arms test recurrence, not learned column structure. The map term shapes only the cross-column weights.
