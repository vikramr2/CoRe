# Inserting recurrent action columns into a feed-forward spiking actor: the recipe

**Scope.** The current procedure for converting an existing feed-forward spiking actor-critic, here the TD-STDP backbone of Chung & Kozma 2020 (`models/tdstdp.py`, `mapstdp/tdstdp.py`), into one whose actions are recurrent action columns ("action assemblies" in the paper; never "cortical columns"), so that the decision and the credit are carried by spikes. It states the current form only; history is in `docs/CHANGELOG.md` and `docs/iterations/015-hybrid-columns.md`, `016-columns-more-tasks.md`. It documents the hybrid of derivation §8.1, not Map-STDP's own network (`docs/model.md`).

**The lesson that shaped it (016 §19–20).** With the backbone's softmax readout, columns are a small tax (1–3 % of AUC on CartPole, about 5 % for partial and dynamic columns on Acrobot) and the three regimes of modularity never differ: the softmax reads one mean rate per column, and the task term seals the columns by itself (within-column weight share 0.91 → 0.998 on CartPole). In model v4, where the policy was the walk (flow share, diffbound race over module spikes), the same structure gave +65 to +87 AUC. So the insertion works only when the walk, modelled by spikes, makes the decision and gets the credit.

Status tags as in `model.md`: **impl** (implemented and tested), **spec** (decided, being implemented), **Proposal** (undecided option, behind a flag), **abstraction** (a plausibility note from the neuroscientist's review).

## 1. Columns

| Item | Current form | Status | What showed it |
| --- | --- | --- | --- |
| Column | Each action's actor E population (n_ae = 20) gets recurrent E→E synapses R, within and across columns (`--rec cols`, `HybridRec`); the feed-forward input→actor weights W stay the backbone's. Latent modules (`--rec lat`) are strategy 3 and not part of the recipe. | impl | 015 §2–3 |
| Loop gain | Total recurrent kick per spike χ_rec = 0.3 (threshold units), r = 1; frozen in 015's pilots B and C. Stability monitors per window: max Re λ ≤ 0.5 of the offspring table, τ_loop ≤ 40 ms, χG. | impl | 015 §5–6; 016 §10 |
| Matched start | `--rate-match-rec`: s_in calibrated with the arm's own R so every arm starts at the target rate (40 Hz, `--rate-target`); `--ent-match-rec`: α_a scaled so the initial policy entropy equals the base run's (softmax arms only). Without matching, column arms start hotter and sharper, which collapsed Acrobot and produced most of 015's early LunarLander speed-up. | impl | 016 §10–11 |
| Step on R | η_rec and λ_rec by `calibrate_rec`: RMS matching of R's task step to W's at init, and λ_rec = λ_ratio × RMS(task step)/RMS(structural step). | impl | 015 §4 |
| Three regimes | partial (`--lam 0`: task term only), dynamic (`--lam 1`: plus the map term, r = 1), full (`--rec-full --lam 0`: no cross-column synapses). | impl | 015 §1; 016 §1 |

## 2. Readout: the race

| Item | Current form | Status | What showed it |
| --- | --- | --- | --- |
| Why not the softmax | The backbone's softmax over column rates discards which neuron fired after which: structure enters only as a gain on a mean, which the feed-forward drive already provides. Under it columns cost 1–3 % and the regimes are indistinguishable. | impl (as the reference arm) | 016 §17, §19 |
| The race | `--readout race`: per copy, each column's E spike count since the last resample accumulates at 1 ms; the first column to reach θ wins at that ms; the leader at the deadline (the freeze window: 20 ms on CartPole and Acrobot, 40 ms on LunarLander) wins otherwise; an exact tie costs one uniform draw. The E posts only are tallied (I posts shape the dynamics). | impl | 016 §14.1, §16, §18 |
| θ | round(n_ae r₀ T_freeze): the mean count of one column at the target rate over a window (16 on CartPole and Acrobot, 51 on LunarLander). | impl | 016 §16 |
| What it reads | The count that races to the bound is the walk circulating inside a column (a spike recruits further spikes through R); a cross-column step is a spike that leaves one tally for another. The quantity the map equation measures, how much of the walk stays inside its module, is the quantity the race decides on. | impl | 016 §20 |
| Monitors | `race_decisions` (bound / deadline / ties), `actor_spikes_per_decision`, `actor_spikes_to_decision` (the energy a loop stopped at the bound would spend). | impl | 016 §16, §18 |
| The hold | `--freeze 20·n_hold --race-theta` by the 20-ms rule: the race decides within one 20-ms env step; the winner is held for n_hold steps, the credit applied each step. n_hold is pre-set per task, the smallest hold at which a uniform random policy terminates ≥ 20 % of episodes (Acrobot 2; CartPole, LunarLander 1). Class: action persistence, temporally correlated exploration variability (Chaisanguanthum, Shen & Sabes 2014; Wu et al. 2014; Dhawale, Smith & Ölveczky 2017; history bias: Hwang et al. 2017; Urai et al. 2019); in RL, sticky actions (Machado et al. 2018) and temporally extended ε-greedy (Dabney et al. 2021). The clock-locked hold is the engineering form; biology's persistence is graded and adaptive; the hold also halves control resolution. The softmax already persists through its filtered rate (hold +1 on the softmax: no effect). Monitor: `action_repeat_probability`. | impl / abstraction (supported class) | 016 §21 |
| Leaky counter | Proposal: column counts decay with τ_leak instead of resetting, so the next race starts from the previous winner's lead. Class: leaky competing accumulator with a history-dependent starting point (Usher & McClelland 2001); biological carrier would be reverberating column activity (Wang 2002; Lo & Wang 2006), which the columns cannot sustain at the present gain (τ_loop 28 ms), so the count stays readout-side. Needs a τ_leak rule and a re-derived θ; the empirical history bias is in the drift, not the starting point (Urai et al. 2019). Not built. | Proposal | 016 §21 |
| θ per task | θ = round(n_ae r₀ · 20 ms) at a 20-ms decision on every task (16 on CartPole and Acrobot, 26 on LunarLander); LunarLander's 40-ms rule (51) put half the decisions at the deadline. | impl (`--race-theta`) | 016 §21 |

## 3. Credit: the spike score

| Item | Current form | Status | What showed it |
| --- | --- | --- | --- |
| The signal | δ, the backbone's TD error (unchanged critic). | impl | CK Eq. 69 |
| The score (headline) | `--credit noise`: for each actor E neuron and ms, the exact score of the neuron's own noise model given its realised membrane: on a spike step f(z)/(σ_n h), on an active no-spike step −f′(g)/(σ_n f(g)) with g = (V − m)/σ_n the step's noise innovation (g/σ_n for Gaussian noise); 0 when refractory. The eligibility on an input synapse is z ← a z + Y_i x_j(t−1)/τ_z (the input the step was paired with; the backbone's 20-ms pre trace would over-count), q ← b q + z/τ_q, ΔW ∝ δ q, the backbone's trace structure with Y in place of the post spike and the column gating inside Y. Derivation Eq. 13; the eligibility as the learner computes it matches finite differences of ∂E[R]/∂W within 1 % (Appendix A, H3). | impl | 016 §20 |
| The score (secondary) | `--credit hazard`: Y = (S − h) ψ, the escape-noise score at the realised membrane, h = 1 − F((θ − m)/σ_n), ψ = f/(σ_n h(1 − h)) clipped at ψ(h = 1e-3). Zero-mean per step; biased as a gradient for this neuron (0.2–0.3× per ms) because the membrane carries the noise forward. Kept as the plausibility-control arm. | impl | 016 §20 |
| Truncation | Y = 0 after the copy's bound ms (post-bound scores are zero-mean); at a deadline decision the whole window counts. Carrier: a per-copy "decided" broadcast. | impl | 016 §20 |
| Which posts | E actor posts only (c = 1); I posts do not enter the race and get no task credit; latent posts none. | impl | 016 §20 |
| Step | `--eta-cal` on every task: η = the authors' step / RMS of the per-ms update at init, with the arm's own credit. | impl | 013; 016 §20 |
| The surrogate (stopgap) | `--credit surrogate` (default): p̃ = softmax(β log max(r, ε_r)), eligibility β/(α_a r_j)(A_j − p̃_j), β by a contrast probe at init (`calibrate_race`) and optionally re-probed every N episodes (`--race-reprobe`). Its mismatch with the race drives a common-mode rate drift of task-dependent sign. | impl | 016 §16, §18 |
| Credit on R | Module level: hs_k = the column's summed score up to the bound / n_ae, in `HybridRec`'s existing per-window credit (the score of the column's count, the quantity the race reads). The per-synapse version Y_i(t) χ S_j(t−1) is the plausible one. | impl / Proposal | 016 §20 |
| Plausibility | Exact score: abstraction with a supported class (perturbation learning with the postsynaptic potential minus its prediction: Fiete & Seung 2006; Legenstein et al. 2010; Urbanczik & Senn 2014); needs the neuron to separate its per-step noise from its input, given biologically where the noise is an identifiable input (LMAN → RA, Ölveczky et al. 2005), native on the crossbar where the noise is injected. Hazard score: abstraction, supported class (escape-noise reward-modulated STDP: Pfister et al. 2006; Florian 2007; Frémaux et al. 2013; postsynaptic factor = membrane relative to threshold, Clopath et al. 2010). | abstraction | 016 §20 |
| Locality | Per synapse: a post-local factor (membrane, threshold, noise scale) × a pre-local trace × one broadcast δ per copy; extra broadcasts: the decision gate and the module score for R, both K-table-sized. Crossbar-native. | abstraction (confirmed) | 016 §20 |

## 4. Homeostasis

| Item | Current form | Status | What showed it |
| --- | --- | --- | --- |
| Rate term | `--rate-cost c_t`: CK's own target-rate term (their Eq. 31; `targ_firing_r`), per copy on the actor E rows, target ρ̂ = r₀ = `rate_target`. In surrogate arms CK's form −c_t (ρ̄ − ρ̂) z per ms; in score arms the rank-one form −c_t (ρ̄ − ρ̂) ρ_i P_j per ms (no per-synapse state beyond the eligibility trace; expectation-equivalent). | impl | 016 §18, §20 |
| Its size | c_t = 5e-4: the smallest power-of-ten multiple of the authors' 5e-6 at which the restoring term exceeds the pilot's observed drift at the gate edge on both tasks (τ_r 19–33 episodes). In score arms the effective step is held: c_t,eff = c_t × η_a(surrogate)/η_a(arm). | impl | 016 §18, §20 |
| What it is | Coincidence-gated (or rate-driven) rate homeostasis on the population mean: a rapid compensatory process (Zenke, Hennequin & Gerstner 2013; Zenke & Gerstner 2017), not synaptic scaling. It holds the common mode only; it cannot revive a silent column, and it makes spikes per window a constant across arms (only spikes to the bound remain an energy measure). | abstraction | 016 §18 |
| Entropy bonus | Off under the score (no explicit policy); on in surrogate and softmax arms where the preset has it (LunarLander, Acrobot). | impl | 016 §20 |
| Rate floor | `--rate-floor 0.5`, ρ_min = r₀/2, pre-set: a one-sided per-column term c_t (ρ_min − ρ_k)_+ on column k's actor-E rows per ms, the rate term's form and effective step (surrogate: on z; score: rank-one ρ_i P_j). Potentiation only, inactive above ρ_min, never lowers a winner; with the mean term it makes a band (copy mean at r₀, each column ≥ r₀/2) that bounds the log-rate contrast between columns from below and keeps every action alive in the race's marginal policy. Not an entropy bonus: it bounds the marginal, not the conditional, policy; zero in a column already silent, so min column rate per log bin is its monitor. | impl | 016 §20 |
| What the floor is | The deprivation side of firing-rate homeostasis on a population mean (Turrigiano et al. 1998; Desai, Rutherford & Turrigiano 1999; Keck et al. 2013; Hengen et al. 2016), in a rapid coincidence-gated form (Zenke & Gerstner 2017); modelling precedent: the low-rate potentiation term of Zenke, Agnes & Gerstner 2015 that keeps a network from falling silent. Per column, not per neuron: a pooled signal with one shared target preserves within-population rate heterogeneity (Sweeney et al. 2015; Slomowitz et al. 2015). Strict one-sidedness and the minutes timescale are engineering choices; biology is bidirectional per neuron and slower. | abstraction | 016 §20 |
| Exploration | Rests on the membrane noise, with the rate floor as the race's exploration floor where the softmax had an entropy bonus. The realised race entropy per log bin is a gate. If it fails: a scalar gain or noise modulation driven by the realised decision entropy (adaptive gain, Aston-Jones & Cohen 2005). | impl (monitor) / Proposal | 016 §20 |

## 5. Structure control

| Item | Current form | Status | What showed it |
| --- | --- | --- | --- |
| Map term on R | λ = 1, the observed-walk rule of the derivation on R's pair counts, per statistics frame; exempt routing. | impl | 015 §3 |
| Full modularity | `--rec-full` (formerly `--rec-seal`): R without cross-column synapses, the same draws. | impl | 015 §3 |
| The test | Whether the regime matters once the readout reads the structure. Under the softmax: never (016 §17, §19). Under the race with the surrogate: full over partial +57 AUC on CartPole (n = 10, development), the first regime contrast to move (016 §18). Under the race with the score: the Stage B study. | impl / spec | 016 §18–20 |
| Walk monitors | `rec.RK` (R's module weight table per snapshot), `within`, `L_obs`, `rec.RKc` (the causal counter per executed action). | impl | 016 §15 |

## 6. What the recipe does not cover

- Latent state modules between input and columns (strategy 3, `--rec lat`): expected to matter only under noise or partial observability (016 §14.2).
- Plastic inhibition and a shared inhibitory pool (Stage A, Stage C of 016 §14.1): not built; the rate term does the common-mode job for now.
- The per-synapse score on R (Proposal above); a decision-entropy gain (Proposal above); the exact per-synapse credit of post-bound spikes that seed the next window through R.

## 7. Studies

| Study | What | Result |
| --- | --- | --- |
| `hybrid-columns` (015) | CartPole, LunarLander; softmax; unmatched start | no regime improves; full non-inferior on both |
| `hybrid-columns-2` (016) | Acrobot; softmax; matched start | partial and dynamic worse (−35, −31); the entropy bonus's silencing drift |
| `hybrid-columns-3` (016 §17) | three tasks; softmax, normalised scores, matched start | never worse, 5/9 non-inferior, LunarLander +7 to +11 before Holm; regimes never differ |
| `hybrid-columns-4` (016 §19) | CartPole, n = 40, both bases | all regimes non-inferior to the normalised base; partial worse than CK's base (−14.7); regimes equal |
| `hybrid-columns-3-ck` (016 §19) | the CK-faithful base on three tasks, same seeds as `-3` | LunarLander: every regime worse than CK's base (−22 to −25), the normalised base −33; CartPole, Acrobot: level; columns never beat the readout they sit in |
| Stage B pilots (016 §18, §20) | the race with the surrogate, its fixes, the score | surrogate: rate drift; with a stabiliser +23 to +41 AUC over CK's base on CartPole (10/10 solved, fewer spikes to decision) but collapse or rate runaway on Acrobot; regimes equal; score: the pilot failed upstream of the credit (CartPole: a 280× step from `--eta-cal` on CartPole's SGD preset, the §15 trap; Acrobot: the critic dies at the floor in every race arm); with the η rule fixed the exact score learns CartPole on 10/10 seeds (last-100 483–490; AUC 187–235 vs the surrogate race's 380–403: 2.5× slower), the hazard form 5× slower again; live-path finite differences match |
| `race-cartpole` (016 §20) | CartPole, race + β re-probe on no / partial / dynamic / full columns vs CK's softmax; settings selected on dev seeds | **columns beat the backbone after Holm in every regime**: full +45 AUC [+29, +63], partial +29, dynamic +25; race alone +14.5 (n.s.); a third of the actor spikes per decision |
| Acrobot and LunarLander race pilots (016 §20–21) | the race with the band and three credits; then the decision hold | without the hold: near-uniform policy at the sparse-reward floor (5–10/10 at −500). **With the hold (2 steps) and the band:** full and dynamic columns level with CK's softmax (AUC −174 vs −174; 2/10 solved vs 0/10; fewer spikes), partial and no columns worse by 37–43; LunarLander: θ 26 at 20 ms matches the backbone's AUC-1000 at half the spikes; full-length pilot running |
| `race-acrobot` (016 §21) | Acrobot, race + hold 2 + band on no / partial / dynamic / full columns vs CK's softmax at β_c −1.5; settings selected on dev seeds | full columns non-inferior to the backbone (+13 [−5, +39]; better last-100; the only solved runs); **full beats no columns (+54, Holm 0.003) and partial (+42, Holm 0.011)**; dynamic −5, partial −25, none −35; more actor spikes than the softmax here (the band pins the rate) |
| LunarLander pilot with the fix (016 §21) | race with the E-rate surrogate at the backbone's control rate, dev seeds, full length | full columns +31 last-100 over CK's softmax (CI clear of zero), dynamic +17, no columns +6 (n.s.); early learning level; 30 % of the spikes; study `race-lunarlander` pre-registered |
