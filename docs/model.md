# Map-STDP model: current form

**Version:** v4 (2026-10-21; reinstated after iteration 008). v4 is a configuration of flags (`--p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5`, with `--td-norm 0.04` and `--la-beta 0.3`). v5 (v4 plus `--beta-eps 0.005 --beta-decay 1000`, iteration 007) was withdrawn: at 40 seeds it gave no benefit (iteration 008); the code defaults still reproduce v3, so earlier results and `tests/golden.json` are unchanged. The specification is in [`derivation.md`](derivation.md) and [`SPEC.md`](SPEC.md). Iterations are logged in [`iterations/`](iterations/), and every change to this file is listed in the version history at the end.

Status tags:

- **impl**: implemented and tested.
- **spec**: decided, not yet implemented.
- **Proposal**: an undecided option, behind a flag.
- **abstraction**: a biological-plausibility note from the `neuroscientist` review.

## 1. Network

| Item | Current form | Status |
| --- | --- | --- |
| Neurons | Modules C, L, L, A, A, all excitatory. M1 used 16 neurons per module (80 total). For CartPole the controller has 36 neurons (§1, Stimulus). | impl (M1) |
| Weights | $W_{ij} \ge 0$, pre $j$ → post $i$, bounded in $[10^{-3}, 1]$. $T = W/d$. In SuperNeuroMAT, `W_snm = (1-α)·(W/d).T`, so the output gain is $1/d_j$. | impl (M1) |
| Initialisation | SBM: $p_{same} = 0.5$, controller recurrence $p_{cc} = 0$ in v4 (`--p-cc`; 0.1 in v1–v3), controller→latent/action routing $p_{route} = 0.3$, other cross links $p_{cross} = 0.05$. Initial $d_j = 1$. With $p_{cc} = 0$ the controller is a feed-forward encoder: the module-level task term cannot credit controller→controller synapses (derivation §2.5), and removing them raised flow return on every M3 task (iteration 006). Abstraction: a thalamic-relay or cerebellar-granule-like input layer, not cortex. | impl (v4) |
| Stimulus | **Conjunctive** Gaussian receptive fields. For CartPole: a 6×6 grid over $(\theta/0.21, \dot{\theta}/3.5)$, clipped to $[-1, 1]$, $\sigma = 0.4$, $v$ normalised. Per-dimension fields cap the policy at about 84 steps. Abstraction: choosing $(\theta, \dot{\theta})$ is designer feature selection, and a full grid scales exponentially; random sparse conjunctive fields are the scaling path. | impl (M2) |
| Stimulus (M3 tasks) | `models/basic.py` (`TASKS`). MountainCar: 6×6 grid over (position, velocity) normalised by the bounds, $\sigma = 0.4$. Acrobot, LunarLander: 64 conjunctive RFs at centers drawn from 20000 random-policy states, observation z-scored by that pool, $\sigma$ the median nearest-center distance (the random sparse conjunctive fields above). One action module of 16 per action; two latents of 16. | Proposal |
| Stimulus, iteration 012 encoders | `--encoder g<n>s<σ>` (an $n \times n$ Gaussian grid) or `f<order>c<conj>` (Fourier basis $\cos(\pi c \cdot x)$ over observations scaled to $[0, 1]$ by hand-set bounds, conjunction order ≤ `conj`, nonnegative ON/OFF rates $[\cos]_+$, $[-\cos]_+$, one feature per controller neuron, $v$ sums to 1) [Chung & Kozma 2020]. Best so far: MountainCar `g12s0.15` (CEM ceiling −101 against −114 for 6 × 6), CartPole `f2c2` (64 neurons). The Fourier encoders lowered LunarLander's ceiling and did not help Acrobot (iteration 012). | Proposal |
| Excitation/inhibition | Excitatory only. **No plausibility claim** until per-module inhibitory (PV-like) pools with inhibitory STDP replace the software normalisation (derivation §7). | abstraction |

## 2. Per-frame flow (one walk per frame)

| Item | Current form | Status |
| --- | --- | --- |
| Flow | $\pi_f = (1-\alpha) T \pi_f + \alpha v_f$, by $n = 20$ hops of power iteration (Eq. 1a), with $\alpha = 0.2$. Within the bound on both initial and trained $W$. | impl (M1) |
| Carry-over | Eq. 1b, $\rho = 0$. | Proposal |
| Spiking mode | **Branching:** `leak=inf`, uniform threshold noise $U(-\theta, \theta)$ with $\theta = 0.5$, so $P(\text{spike}) = u/(2\theta)$ is linear. Teleportation is Bernoulli input at rate $1.0 \cdot v_i$. Mean rate ≈0.057 per neuron per step. Abstraction: one step ≈ 3–10 ms, valid at ≤0.05 spikes per neuron per step. The branching ratio $1-\alpha = 0.8$ is much more subcritical than cortex (≈0.98). | impl (M1) |
| Frame window | $k = 100$ steps. Module error ≈ $1.3 / \sqrt{S_{mod}}$, with $S_{mod}$ the spikes per module per frame; about 170 spikes per module give 0.1. Finite leak has a 0.09–0.15 error floor and is not used. | impl (M1) |
| Parallel heads | $H$ independent spiking replicas on the shared W (`--heads`). Spikes are pooled: $\hat{\pi}$ from summed counts, $\kappa$ the sum of per-head covariance counts over total spikes, one race over all heads (still Eq. 1d). One update per frame. $\hat{\pi}$ error falls as $1/\sqrt{H}$ (iteration 005); effect on return not yet measured. | Proposal |
| Spiking engine | SuperNeuroMAT (default, M2), or `sp.run_heads`: the branching-mode step batched over heads, raster-identical to SuperNeuroMAT (self-check), 7–13× faster. M3 default. | impl |
| Pairings $\kappa$ | **Covariance count** $C_{ij} - c_i c_j (k-1)/k^2$: lag-1 causal pairings minus the product of spike counts. It correlates 0.91 with $J$, and its average update has cosine 0.992 with $-W \odot \nabla D$. Causal-only is an ablation; balanced is dropped (cosine 0.847). Biological variant (Proposal): subtract slow pre/post traces online. | impl (M1) |

## 3. Readout and state model

| Item | Current form | Status |
| --- | --- | --- |
| Module chain | $T^K_{ba}(\mathbf{o}) = J_{ba} / p_a$ (Eq. 1c). | impl (M1) |
| Policy | $P(a \mid \mathbf{o}) = p_a / \sum_{a'} p_{a'}$ (Eq. 1d). In spiking, a race readout that starts after a **burn-in of $b = 10$ steps**; KL to Eq. 1d is 0.011, at the sampling floor of 0.010. | impl (M2) |
| Sharpened policy | $P_\beta \propto p_a^\beta$ with matched score (Eq. 1e). Fixed $\beta = 4$ is unstable (M2). | Proposal |
| Adaptive $\beta$ | Eq. 1g: after each episode $\log \beta \mathrel{+}= \kappa_\beta z$, $z$ the clipped trend of return (20- against 100-episode averages, over the running sd), $\beta \in [1, 16]$, $\kappa_\beta = 0.01$, no update over the first 20 episodes; one global scalar, all tasks (`--beta-kappa 0.01 --beta-max 16`). Flow; in spiking only with the bound readout below (a race to $k$ spikes reaches $\beta_{eff} \approx 1.5$). Through the matched score it is also a global step-size gain on the task term. Abstraction: a utility-trend gain like the adaptive gain theory of LC-NE, with no term for the utility level. v5 added a set-point rule (iteration 007; withdrawn), an upward-only pull ($\epsilon_\beta = 0.005$) toward an experience-dependent set point: the running maximum of a 100-episode average of $\log \beta$, leaking back over $\tau_D = 1000$ episodes (`--beta-eps 0.005 --beta-decay 1000`). It equals 1g while $\beta$ is at or above the set point, and makes $\beta$ recover after a fall (iteration 007, runs pooled over $\kappa_\beta$ 0.01 and 0.0099: 5 of 5 CartPole drops recover, against 1 of 3 in v4; **not confirmed** at $\kappa_\beta$ 0.01 on seeds 0–19: 0 of 3, against 2 of 3 for v4, two of the three drops within 200 episodes of the end; iteration 008). It does not change return, and it cannot repair a collapse in the weights (iteration 007). Abstraction: a history-dependent homeostatic set point (metaplasticity, tonic dopamine as average reward), not a known LC-NE mechanism. Known issue: needs episode boundaries. The time-based ramp (`--beta-end`) is an oracle reference ablation, not part of the model. | Proposal (v4) |
| Spiking bound readout | Eq. 1h: the action is the module whose capped spike count (at most one spike per module, head and step) first leads every other by $\theta$, after the burn-in; the leader at the deadline. Gambler's ruin makes it Eq. 1e with $\beta = \theta$ for two actions; spike bursts ($F \approx 15$–20) make $\beta_{eff} \approx \theta / F$ raw. $\theta$ comes from $\beta$ through the measured table `BOUND_CAL` (16 heads: $\beta_{eff}$ 3.1–11.5 for $\theta$ 1–64). `--readout diffbound` (spiking, `--heads 16`). Pilot, 5 seeds: MountainCar −132.2 against −177.0 for the race (flow parity); CartPole 153.9 against 298.3 (worse; $\beta$ floor 3.1 and top 11.5). In flow it adds nothing (same ceilings as Eq. 1e). Abstraction: a decision threshold set by cortico-striatal/subthalamic circuits; a saturating readout synapse for the cap (iteration 010). | Proposal (not adopted) |

## 4. Plasticity

The full rule is $\Delta W_{ij} = \eta_e \delta_f e_{ij} - \lambda (g_{ij} - \bar{g}_j) \kappa_{ij} - \epsilon_h (d_j - 1) \kappa_{ij}$, with $\eta_e = \eta / (1 + e / \tau_\eta)$ in episode $e$.

| Item | Current form | Status |
| --- | --- | --- |
| Structural modulator | Map equation, $g_{ba} = M^{\ast}_a \mathbb{I}[b \ne a]$, with $M^{\ast}$ clipped at 20. $\lambda = 0.05$, constant from frame 0. In spiking, $\lambda$ is divided by the spikes per frame, keeping the per-count $\lambda \lesssim 5 \times 10^{-4}$. It seals latent and action modules (persistence 0.98–0.996) at no cost to CartPole return. In v4, $\lambda = 0$ costs CartPole (500 → 396), MountainCar (−132 → −187) and LunarLander (102 → 11), not Acrobot (iteration 006). Latents become flow sinks: the task term cuts latent→action flow and the map term seals latents against the controller (iteration 003). | impl (M2) |
| Latent output | Barrier (Eq. 4e), $-\beta_{LA} p_a / J_{\mathcal{A} a}$ subtracted from $g$ on latent→action pairs; `--la-beta`, off by default. On CartPole at $\beta_{LA} = 0.3$ it raises the latent→action share from 0.002 to 0.029, with flow return 279 ± 22 and spiking 155 ± 9. Candidates for M4 are 0.15 and 0.3. It replaces the exit-floor Proposal, which was rejected: its dual does not converge and its exit is not directed to the actions. | Proposal |
| Routing protection | **Exempt** (default): $g = 0$ on controller→latent/action pairs, so the controller acts as an input layer. Ablations: none (collapses learning in the closed loop) and dual (Eq. 4a, $q^{\ast} = 0.75 J_{route}(W_0)$, $\eta_\mu = 2$; $\mu$ diverges, see iteration 002). | impl (M2) |
| $d_j$ homeostasis | **Its own term** (Eq. 4d), $-\epsilon_h (d_j - 1) \kappa_{ij}$ with $\epsilon_h = 5$, every frame, independent of $\lambda$. Its expectation is a uniform column rescaling, so $T$ and $D$ are unchanged. It holds $d_{max} \le 1.04$ in v3 (v1: up to 19). Abstraction: fast heterosynaptic plasticity; the biological evidence is for postsynaptic, not presynaptic, conservation. | impl (M2) |
| Task eligibility | $e_{ij} \leftarrow (1 - 1/\tau_e) e_{ij} + \kappa_{ij} (h_{m(i)} - \bar{h}_j)$ (Eq. 4b), $\tau_e = 3$ frames. | impl (M2) |
| Task modulator | TD error $\delta_f = r_f + \gamma V_{f+1} - V_f$, $\gamma = 0.99$. Critic $V = u \cdot \pi_{\mathcal{C}} / p_{\mathcal{C}}$, a readout of controller flow shares, trained by NLMS with $\eta_V = 0.1$. Reward −1 on termination, 0 otherwise, bootstrapping on truncation. | impl (M2) |
| Separate critic | Eq. 4c'': $V = u \cdot y$ on its own fixed nonnegative sensory layer $y = \phi(\mathbf{o})$ instead of the controller shares, NLMS as in 4c (`--critic-layer g24` / `f2c2b` / …). Best so far: MountainCar 24 × 24 grid (with the 12 × 12 encoder), Acrobot and CartPole `f2c2b`; no help on LunarLander. A ventral-striatal value pathway on shared input; communities remain the actor's states (iteration 012). | Proposal |
| Entropy and usage terms | Eq. 4h, and a usage term (iteration 012): ungated per-frame module-level updates toward higher entropy of the $\beta = 1$ action shares (`--c-ent`, optionally `--ent-target f`) or toward equal long-run usage (`--c-use`, `--use-tau 300`). Destructive at 0.1–0.3 of the task term's RMS; at $c_e = 0.01$ with target $0.8 \log \lvert \mathcal{A} \rvert$ a decaying exploration bonus that helped MountainCar only (iteration 012). Abstraction: usage homeostasis of action channels, not a neuromodulator. | Proposal |
| Task reward (M3) | CartPole −1 on termination; MountainCar and Acrobot $0.1 r$; LunarLander $0.01 r$ (environment reward $r$). | Proposal |
| TD normalisation | `--td-norm s`: the task term uses $\delta s / \mathrm{RMS}(\delta)$, running mean square over 1000 frames; the critic learns from the raw $\delta$. $s = 0.04$ is CartPole's RMS $\delta$. Fixes the Acrobot collapse (−374 ± 156 → −128 ± 7), neutral on the other tasks. Candidate M3 default. | Proposal |
| Task rate | $\eta = 10$ in both backends (v4 flow: $\eta = 5$, `--eta`), **annealed** as $\eta / (1 + e / 300)$ over episodes $e$. A constant rate stalls (spiking 88 at $\eta = 10$, 120 at $\eta = 5$; annealed 184). | impl (M2) |

## 5. Parameters

| Symbol | Meaning | Value |
| --- | --- | --- |
| $\alpha$, $n$, $\rho$ | teleportation share, hops per frame, carry-over | 0.2, 20, 0 |
| $\lambda$, $\epsilon_h$ | structural rate, $d_j$ homeostasis rate | 0.05, 5 |
| $\eta$, $\tau_\eta$, $\tau_e$ | task rate, its annealing time constant (episodes), eligibility time constant (frames) | 10 (v4 flow: 5), 300, 3 |
| $p_{cc}$ | controller recurrence | 0.1 (v4: 0) |
| $\kappa_\beta$, $\beta_{max}$ | adaptive-$\beta$ gain and bound (Eq. 1g, Proposal) | 0 (off; v4: 0.01), 16 |
| $c$, shape | cap on $\beta$'s gain on the task step (`--beta-step-cap`, `--beta-step-shape`; ablation, not adopted, iteration 008) | 0 (off) |
| $\epsilon_\beta$, $\tau_D$ | pull toward the $\beta$ set point and its leak time in episodes (set-point rule of iteration 007, withdrawn) | 0 (off; withdrawn v5: 0.005), 0 (strict maximum; withdrawn v5: 1000) |
| $b$ | race burn-in (steps) | 10 |
| $\beta_{LA}$ | latent-output barrier (Eq. 4e, Proposal) | 0 (off; M3: 0.3) |
| $\gamma$, $\eta_V$ | discount, critic rate (NLMS) | 0.99, 0.1 (`--gamma 0.995` tested, not adopted, iteration 009) |
| $c_e$, $f$, $c_u$, $\tau_u$ | entropy term, its target entropy fraction, usage term, usage time constant (Eq. 4h and the iteration-012 usage term, Proposal) | 0, off, 0, 300 frames (MountainCar best: 0.01, 0.8) |
| $\lambda_V$ | critic eligibility trace, TD($\lambda$) (`--critic-lambda`; tested, not adopted: diverges on Acrobot, iteration 009) | 0 |
| $\theta$, input rate, $k$ | spiking noise, input rate, frame window | 0.5, 1.0, 100 |
| $w_{min}$, $w_{max}$, $M^{\ast}$ clip | bounds | $10^{-3}$, 1, 20 |
| seeds, episodes | M2 runs | 0–4, 1500 |
| $H$, $s$ | spiking heads, TD-normalisation target RMS (Proposals) | 1, 0 (off; M3 candidate 0.04) |

## Version history

| Version | Date | Change | Iteration |
| --- | --- | --- | --- |
| v0 | 2026-10-05 | Initial form from derivation and SPEC; nothing implemented. | — |
| v1 | 2026-10-05 | M1 implemented. Changes: covariance $\kappa$; output gain $1/d_j$ and $d_j$ homeostasis; branching spiking mode with $k = 100$; conjunctive encoding; routing exempt by default; constant $\lambda$; task-term parameters from flow-level CartPole tests. | [001](iterations/001-m1.md) |
| v2 | 2026-10-05 | M2 fixes: $d_j$ homeostasis as its own term (Eq. 4d, $\epsilon_h = 5$); race burn-in $b = 10$; spiking $\eta = 5$. Fixes the ratchet and the race bias, but return falls (spiking 140 → 120). | [002](iterations/002-m2.md) |
| v3 | 2026-10-05 | Task rate $\eta = 10$ annealed as $\eta / (1 + e/300)$ in both backends; the v1 ratchet had been hidden annealing. Spiking return 184 ± 13, flow 287 ± 21. | [002](iterations/002-m2.md) |
| v3 | 2026-10-20 | Defaults unchanged. Added: M3 task encoders and rewards, parallel spiking heads, TD normalisation (Proposals); the batched spiking engine. | [005](iterations/005-m3-tasks-heads.md) |
| v4 | 2026-10-21 | M3 configuration: controller built without recurrence ($p_{cc} = 0$); adaptive $\beta$ (Eq. 1g, Proposal) with $\eta = 5$; TD normalisation 0.04 and $\beta_{LA} = 0.3$. Flow return: CartPole 500, MountainCar −132, Acrobot −100, LunarLander 102 (10 seeds). Code defaults unchanged. | [006](iterations/006-m3-reward.md) |
| v5 | 2026-10-21 | Eq. 1g′ (Proposal): $\beta$ recovers after a collapse through an upward-only pull toward a leaky maximum of its own slow average. Flow return, pooled over $\kappa_\beta$ 0.01 and 0.0099 (n = 20): CartPole 489.8, MountainCar −130.4, Acrobot −106.7, LunarLander 92.3, none significantly different from v4 (466.0, −132.4, −106.9, 100.6). Code defaults unchanged. **Correction (iteration 008):** at $\kappa_\beta$ 0.01 on seeds 0–19, v5 is not better than v4 on any task (CartPole 449.7 against 487.6, MountainCar −129.5 against −131.9, Acrobot −105.5 against −107.6, LunarLander 60.3 against 77.5); seeds 0–9 were a favourable set. At n = 40 (seeds 0–39): CartPole 474.6 against 486.7, MountainCar −130.4 against −131.0, Acrobot −107.1 against −107.8, LunarLander 52.4 against 81.3 (p = .13). | [007](iterations/007-robust-beta.md) |
| v4 (reinstated) | 2026-10-21 | v5 withdrawn: at $\kappa_\beta = 0.01$ and 40 seeds it is not better than v4 on any task, and nominally worse on LunarLander. v4 at 40 seeds: CartPole 486.7, MountainCar −131.0, Acrobot −107.8, LunarLander 81.3. The step-gain caps of iteration 008 were not adopted. | [008](iterations/008-beta-step-gain.md) |
| v4 (kept) | 2026-10-21 | Iteration 009 tested the credit horizon (critic TD($\lambda$), $\tau_e = 10$, $\gamma = 0.995$): nothing replicated; v4 sits 10–25 below the ceiling of its own Eq. 1e readout. | [009](iterations/009-credit-horizon.md) |
| v4 (kept) | 2026-10-21 | Iteration 011 diagnostics (no change): a sharper readout does not help either task; MountainCar is limited by the encoder (6 × 6 ceiling about −114 at 10 CEM runs) and the critic's resolution (12 × 12 encoder + separate 24 × 24 critic: −101.9, 18/20, preliminary); Acrobot by early entrenchment of the zero-torque action. | [011](iterations/011-flow-diagnostics.md) |
| v4 (kept) | 2026-10-21 | Iteration 012, per-task Proposals (40 seeds): MountainCar 12 × 12 encoder + 24 × 24 separate critic + entropy target −100.7 (38/40 at −110; v4 −131.0, 0/40); Acrobot separate Fourier critic −92.8 (35/40; v4 −107.8, 17/40); CartPole Fourier encoder + critic 500 (40/40; median 296 episodes to solve against 542); LunarLander unchanged. No version bump: no single configuration covers all tasks, and iteration 013 may change the backbone. | [012](iterations/012-encoders-critic-entropy.md) |
| v4 (kept) | 2026-10-21 | Iteration 010: a bounded count-difference race (Eq. 1h) as the spiking readout for adaptive $\beta$ (Proposal). Flow ceilings equal Eq. 1e's, so the flow model is unchanged; spiking MountainCar reaches flow parity in a 5-seed pilot, CartPole regresses. | [010](iterations/010-decision-bound.md) |
