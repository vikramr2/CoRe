# Map-STDP Project Spec

Status: implementation. M1 and M2 are done (iterations 001 and 002); the current model is in [`model.md`](model.md), and iterations are logged in [`iterations/`](iterations/). The math lives in [`derivation.md`](derivation.md), and changes are logged in [`CHANGELOG.md`](CHANGELOG.md).

Each consideration below states the user's direction first. Items marked **Proposal** are candidate approaches that have not been decided. **Open** items are unresolved.

## 1. Goal and hypothesis

Give a spiking network a **state-space abstraction in which its communities are the states**, and use it to solve RL tasks.

**Approach.** A random walk on a graph with communities has a minimum description length, given by the map equation. Community detection finds the partition that minimises that length for a given graph. **Map-STDP inverts this.** The partition is given, and a three-factor STDP rule shapes the weights until the given partition is the minimum-description-length one (`derivation.md` §3). The coarse-grained walk between communities is then a Markov model over the network's states, conditioned on the stimulus. It can be extracted, and it chooses the action (`derivation.md` §2.4–2.5).

**Hypotheses.**

1. **Modules form.** The communities emerge and match the intended controller, latent and action roles (NMI).
2. **The state model works.** The extracted chain $T^K(\mathbf o)$ gives a policy that solves basic RL tasks. With carry-over, the latent communities give a predictive model of hidden state.
3. **Side tie: thermodynamics.** A neuron pays more metabolic cost to communicate with cortices it does not normally talk to. If description length tracks that cost, the information dynamics the rule minimises are a model of thermodynamics in the brain, and the rule lowers a biophysical energy proxy. This is secondary: the state abstraction does not depend on it.

Map-STDP ([`derivation.md`](derivation.md), Eq. 4) is the rule: reward-modulated STDP for the task, plus cost-modulated STDP whose neuromodulator reports the marginal description length. Which description length is best is an open question (§6).

## 2. Architecture

```text
stimulus o_f ──► controller C ─────────────► action communities A ──► action
                   ▲      │                        ▲
                   │      ▼                        │
                   └── latent communities L ───────┘
                              │      ▲
                              └──────┘  carry-over ρ: frame f → f+1
```

There are three community types (`derivation.md` §2):

- **Controller $\mathcal C$.** One input community that receives the stimulus: $v(\mathbf o)$ is concentrated here. It is built without internal recurrence ($p_{cc} = 0$, iteration 006), so it is a feed-forward encoder, closer to a thalamic relay or a cerebellar granule layer than to cortex. It routes flow to latent and action communities through plastic projections, as cortex does onto striatum. Recurrence here would blur the policy, and its credit is per neuron, which the module-level task term cannot deliver (`derivation.md` §2.5). Because the stimulus enters here, the extracted chain is stimulus-conditioned (`derivation.md` §2.4).
- **Action communities $\mathcal A$.** One per discrete action, analogous to basal-ganglia action channels. The policy is the flow share of each action community (`derivation.md` Eq. 1d). In spiking, a race readout (first population to spike) gives exactly the same policy, while the column with the most spikes is a different, near-greedy policy.
- **Latent communities $\mathcal L$.** Abstract states that are not tied to an action, analogous to task-state codes in orbitofrontal cortex and hippocampus. With carry-over they persist across frames and form a learned world model.

**Conventions.** $W_{ij}$ is the synapse from pre $j$ to post $i$. The flow $\pi_f$ follows spikes forward and is computed once per frame $f$.

## 3. C1: Simulator

**Direction (revised 2026-10-04): SuperNeuroMAT** ([ORNL SuperNeuro](https://github.com/ORNL/superneuro); reference submodule in `docs/superneuro`). This replaces Brian2.

Why it fits:

- **It already works the way the model does.** The simulator is discrete-time and matrix-based. Each step computes `states += input + W_snm.T @ spikes`, then threshold, reset and refractory period. With delay 1, one step is one matrix–vector product: one hop of the walk, and one crossbar read.
- **The per-frame closed loop is plain Python.** Each frame does `simulate(k)`, reads `ispikes`, computes the plasticity, writes the weights back with `set_weights_from_mat`, then steps the Gymnasium environment. Brian2 would have been restricted to runtime mode for this.
- **The plasticity code is shared with the numpy flow-level reference.** That reference runs the RL ladder first (§5).
- **Backends:** `cpu` (with scipy sparse), `jit` (numba) and `gpu` (numba-cuda).
- **A hardware path.** Networks export to the NeuroCoreX FPGA platform (`to_json`). That platform does inference only, and it is digital, not memristive.

Gaps and how they are handled:

| Gap | Handling |
| --- | --- |
| Built-in STDP is global (`apos`/`aneg` vectors per lag) with no third factor. `aneg` is not acausal STDP: it depresses every non-coincident synapse on every step, which works as a decay (checked). | Turn built-in STDP off for Map-STDP. Each frame, compute covariance pairing counts from `ispikes` (`derivation.md` §2.2), plus the modulator table, the action-gated eligibility and the TD error, in numpy. The built-in STDP is kept only for the "STDP only" ablation. |
| Neurons are deterministic LIF with a subtractive linear leak, and there are no synaptic time constants. | Inject noise as random input spikes, or as random thresholds set from the frame loop or a per-step `callback`. |
| `weight_mat()` is indexed `[pre, post]`, the transpose of our $W_{ij}$ (checked). | `W_snm = W.T` everywhere, as recorded in CLAUDE.md. |
| Delays above 1 are built from hidden chains of neurons, which inflate $N$. | Use delay 1. |
| No E/I populations or Dale's law. | Use signed weights with sign masks that the plasticity code enforces. SuperNeuroABM (LIF and Izhikevich neurons, exponential synapses, GPU through SAGESim) is an option for heterogeneous neuron models later. |
| `add_spike` times are relative to the current step, because queued inputs shift after each `simulate` (checked). | Queue each frame's inputs at times 0 to $k-1$. |

**Proposal: branching-process mode.** With `leak=inf`, a neuron's state resets every step, so spiking at $t+1$ depends only on input at $t$. With stochastic input, the network is then a branching process in which one step is one hop, the closest discrete analogue of the per-frame walk (`derivation.md` §2.1). Test it at M1 against the finite-leak mode.

**Measured cost** (SuperNeuroMAT 3.5.0, smoke test, 20 steps per frame, 20% connectivity, custom plasticity applied every frame):

| Neurons | Per frame, total | Per frame, inside `simulate` |
| --- | --- | --- |
| ~100 | ~1 ms (`jit` compiles once, ~0.1 s) | ~0.5 ms |
| ~500 | 40–50 ms | 12–16 ms |
| ~1000 | ~230 ms (`cpu`) | ~45 ms |

Above a few hundred neurons, most of the time is spent outside `simulate`, reading and writing the weights and queuing input spikes in Python. Optimising that, for example with sparse weight updates or by writing the internal arrays directly, is an implementation task for M1.

Risks:

- Plasticity runs in Python, outside the backend kernels.
- The neuron model is limited.
- The simulator is less established in the literature than Brian2.

**Fallbacks.** Brian2 for continuous-time or conductance-based fidelity checks, if a reviewer asks for them. A PyTorch SNN library with batching if scale demands it.

**Proposal.** Run the RL ladder first on the numpy flow-level reference: implicit walk, expected or sampled pairings, and the same modulators. Use SuperNeuroMAT for spiking fidelity at M1 and M2, then for the ladder once the rules work. At 500–3500 episodes × ~100 frames, a few-hundred-neuron network costs minutes to hours per seed.

## 4. C2: Applications

**Direction:** basic discrete-action RL environments first, several of them. Each action is a community.

**Task ladder (Proposal):**

| Step | Environment | Actions | What it tests |
| --- | --- | --- | --- |
| 1 | CartPole | 2 | closed loop, controller routing |
| 2 | Acrobot, MountainCar | 3 each | more actions; MountainCar's sparse reward |
| 3 | LunarLander | 4 | richer observation |
| 4 | a partially observable task, e.g. a T-maze or MiniGrid memory task | 3–4 | latent communities with carry-over must hold hidden state |

- **Observation encoding (decided, iteration 001).** Use **conjunctive** Gaussian receptive fields. For CartPole, that is a 6×6 grid over $(\theta, \dot\theta)$, $\sigma = 0.4$ in normalised units, 36 controller neurons. Per-dimension fields cap the policy, because Eq. 1d averages their votes: even ideal votes reach only 84 steps, against 491 for conjunctive cells (flow-level). **Open:** how conjunctive grids scale to LunarLander's 8-D observation.
- **Deferred: vision.** Image classification as a contextual bandit (one-step episodes, one community per class), MNIST and Imagenette with a frozen feature encoder. This is out of scope until the RL ladder works.

## 5. C3 + C7: Stimulus, frames and timescale

**Direction:** the stimulus is in the equations, each environment frame is one random walk, and the walk must be implicit or very fast so that timescales do not become a problem.

**Adopted: stimulus as teleportation.** $\pi_f = (1-\alpha)T\pi_f + \alpha v_f$ with $v_f = v(\mathbf o_f)$. This:

- makes the flow well defined, since the chain is irreducible and the fixed point is unique;
- makes $\pi_f$, and hence the modulators and the extracted chain, stimulus-conditioned; and
- gives $\alpha$ a measurable meaning: the external share of total drive.

**Proposal: implicit walk per frame.** $\pi_f = \alpha (I - (1-\alpha)T)^{-1} v_f$ is approximated by $n$ hops of power iteration, with error $\le 2(1-\alpha)^n$ independent of network size (`derivation.md` Eq. 1a). Modular networks approach this bound, so budget for it: $\alpha = 0.2$ and $n = 20$ give $\le 2.3\%$. The three realisations are:

- software reference: $n$ sparse matrix–vector products;
- crossbar: $n$ analogue reads;
- spiking: not discrete hops. The recurrent dynamics relax with time constant about $\tau/\alpha$, so a frame takes about 100–250 ms, and a 50 Hz environment needs frame-skipping (`derivation.md` §2.3).

The software reference is the default. The spiking estimate is the biological and hardware variant, and its agreement with the reference is a milestone (M1).

**Proposal: carry-over.** Latent state persists across frames by mixing the previous frame's latent flow into the teleportation vector with weight $\rho$ (`derivation.md` Eq. 1b). With $\rho = 0$, frames are independent, which suffices for fully observed tasks. Linear carry-over forgets geometrically, by a factor of about 0.3–0.4 per frame in a test, so holding a cue needs:

- a module-level soft winner-take-all over latent communities;
- eligibility traces at least as long as the cue-to-reward delay;
- protection of controller→latent routing.

**Timescale ordering:** walk hops, then frames, then plasticity.

**Proposal (extension): environment as nodes.** Add sensor and actuator nodes, so that flow runs from action communities through the environment and back into the controller. The environment becomes a community in the map equation, and the codelength then accounts for agent–environment coupling. Try this after teleportation works.

**Open:**

- $n$, $\alpha$ and $\rho$, and whether $\alpha$ should be estimated from input and recurrent currents.
- Whether teleportation and carry-over steps should count as module exits (Infomap's "recorded teleportation" choice). The derivation currently does not count them.
- Whether the spiking estimate settles within a frame window short enough for the environment.

## 6. C4 + C5: Description length as a neuromodulator

**Direction:** the description length is the map equation (decided 2026-10-23; alternative description lengths were dropped), and the structural term is a three-factor rule, since neuromodulators gate costly plasticity.

**Decision (2026-09-27): cost-modulated STDP is the implemented form.** See `derivation.md` §4.

**Decision (2026-10-04): the task term is TD-modulated STDP with action-gated eligibility.** It replaces $(R - \bar{R}) e_{ij}$, which gets no action credit when the action is sampled from flow shares, and stalls under sparse reward. See `derivation.md` §4.6. The rule is

$$
\Delta W_{ij} = \eta \delta e_{ij} - \lambda \left( g_{ij} - \bar{g}_j \right) \kappa_{ij}
$$

where:

- $\delta$ is a global TD error from a linear critic on the controller's flow shares (`derivation.md` Eq. 4c);
- $e_{ij}$ is a trace of $\kappa_{ij} (h_{m(i)} - \bar{h}_j)$, where the $K$-vector $h$ marks the chosen action (Eq. 4b). Its average is a weight-scaled policy-gradient score. It needs the chosen action broadcast to the action modules, like an efference copy;
- $\kappa_{ij}$ is the pre→post coincidence, counted in spiking as the covariance count (causal pairings minus the product of spike counts), to cancel chance coincidences (decided in iteration 001);
- $g_{ij} = \partial D / \partial J_{ij}$ is the **marginal** description cost of the transition, a $K \times K$ module-pair table broadcast per module;
- $\bar{g}_j$ is a per-neuron baseline.

On average this is weight-scaled gradient descent on $D$ (`derivation.md` Theorem 3). In an environment, $D$ is the frame-averaged description length, and the theorem holds frame by frame (`derivation.md` §3). The factorized exact gradient is kept only as the analysis reference and as simulation baseline A: the additive rule $\eta \mathrm{STDP} - \eta' G$.

Why this form:

- **Crossbars.** It is a single rule for memristive crossbars (§7).
- **No trace for structure.** Description costs are instantaneous, so the structural term needs no eligibility trace. Only delayed task reward does.
- **Marginal, not pointwise.** The modulator must be the marginal cost. Broadcasting pointwise cost, such as each step's codeword length, is biased (checked in simulation).

**Known conflict: structure starves routing** (`derivation.md` §4.5). Controller→action and controller→latent links are cross-module, so the map term alone drives the controller's exit flow to zero. The action communities starve, $M^{\ast}$ diverges, and the policy collapses.

**Decision (iteration 001): routing exemption is the M2 default.** In the flow-level closed loop, the map term without protection collapsed CartPole learning (≈53 against ≈195 with $\lambda = 0$). With routing exempt, the structural term raised the return to ≈343 at $\lambda = 0.05$. The exemption has no free parameter.

**Proposal: dual routing term** (an ablation for M2). Subtract $\mu$ from $g_{ba}$ on the routing module pairs, with $\mu$ set by dual ascent toward a target routing flow $q^{\ast}$ (Eq. 4a). The simplest variant exempts those pairs. Preliminary flow-level bandit results (4 seeds), P(correct):

| Structural term | P(correct) |
| --- | --- |
| none | 0.657 |
| map term alone | 0.50, with 3 of 4 runs collapsed |
| routing pairs exempt | 0.693 |
| dual term | 0.762 |

**M2 result (iteration 002, model v3, 5 seeds, flow level), mean of the last 100 of 1500 episodes:**

| Routing | Return |
| --- | --- |
| none | 39 ± 11 |
| exempt (default) | 287 ± 21 |
| dual | 327 ± 46 |

$\lambda = 0$ gives 187 ± 15. The dual's $\mu$ never settles (2600–8900), so it acts as routing potentiation rather than a constraint. Its lead over the exemption is about one sd and was reversed in v2. Exemption stays the default; the dual stays a Proposal until $q^{\ast}$ is learned or $\mu$ converges.

**Schedule.** Constant $\lambda = 0.05$ from frame 0 beat $\lambda = 0$ in the flow-level tests, so the ramp is an ablation. The task rate $\eta$ is annealed (model v3).

**Open:**

- **Energy model.** The side hypothesis (description length tracks metabolic cost) needs a concrete energy proxy, e.g. Attwell & Laughlin-style ATP accounting: per-spike cost, plus a per-synaptic-event cost that scales with weight, plus a wiring-distance term.
- **Lateral inhibition.** How $M^{\ast}$ is defined when modules are implicit.
- **Biological carriers.** A neuromodulator can plausibly carry a slow per-region scalar, but not a $K \times K$ table. The map modulator needs only $K$ scalars $M^{\ast}_a$ and an exit mask (`derivation.md` §6).

## 7. C6: Hardware target: memristive crossbars

**Direction:** the rule must be implementable on memristive devices.

Constraints this places on every rule (see `derivation.md` §6):

- **Updates** are pulse-coincidence STDP scaled per row by a broadcast signal: one phase per target-module column mask.
- **No nonlinear function of an individual device's conductance** is allowed, and no access to the transposed element $W_{ji}$. This is why every modulator is defined at module level.
- **Allowed reads.** Per-row sums come from $K + 1$ masked crossbar reads ($d_j$, $e_j$, module flows). The per-frame walk is $n$ ordinary reads. Modulators and baselines are computed in peripheral logic from $K \times K$ statistics.
- **Per-synapse state.** The only per-synapse state beyond $W$ is the task eligibility trace. Volatile, diffusive memristors are one candidate for it.

**Open:**

- device non-idealities (update asymmetry and nonlinearity, limited conductance levels, noise);
- whether to prototype on a crossbar simulator (e.g. with device models) after M3.

## 8. Open questions carried from the derivation

- **Partition $M$.** The controller and action communities are pinned, because they carry the input and the policy. Latent communities: static (SBM initialization) or periodically re-detected (Infomap, Leiden), which makes training an alternating minimisation (`derivation.md` §3).
- **Number of latent communities** $\lvert \mathcal L \rvert$: a hyperparameter, or set by re-detection.
- **World model.** The definition and estimator of the across-frame latent chain (`derivation.md` §2.4) are proposals, not yet checked in simulation.
- **Structural plasticity.** Threshold pruning ($W_{ij} < \epsilon$) vs. top-$k$ per neuron.
- **Initialization.** SBM (head start, bias) vs. Erdős–Rényi (neutral, slow).
- **Locality convention.** The forward walk ties flow to firing and maps onto crossbar row sums. The backward, dendritically normalized walk has postsynaptic sums but loses that tie. See `derivation.md` §6 (Biological locality).
- **Lateral-inhibition variant.** The anti-Hebbian $\Delta I_{ij} = \eta\pi_i\pi_j$ is unbounded and treats co-active neurons as competitors, which conflicts with Hebbian STDP. It also ignores Dale's law. It needs a bounded, correctly signed rule, probably through interneuron populations.
- **Mean-field validity.** Does $\pi \approx$ normalized firing rate, and do causal pairings occur at a rate proportional to $J_{ij}$ (`derivation.md` §2.2)? Theorem 3 depends on both. It needs constant $d_j$, global normalisation by inhibition, and balanced pairing counts. A preliminary Poisson test found 85% chance pairings in a 20 ms window. Measure at M1, and derive $\mathbb{E}[\Delta W]$ including the residual chance term.
- **Controller in $D$.** Provisionally answered: under routing exemption, the controller's column of $g$ is zero, so it acts as an input layer outside the structural term. Neuroscience reading: a low-recurrence controller is relay-like, as in thalamus, not cortical.
- **Routing target.** How to set $q^{\ast}$ for the dual term, and whether it should be learned from reward.
- **Nonlinear carry-over.** Is a nonlinear latent carry-over compatible with the theory, given that the gradients assume a linear resolvent?
- **Device behaviour.** Do memristors with state-dependent updates give the multiplicative ($\propto W$) form natively, or only as an expectation over pairings?
- **Dale's law.** Route lateral inhibition through per-module interneuron pools with inhibitory STDP (Vogels et al. 2011). The same pools supply normalisation and rate homeostasis (`derivation.md` §7).
- **Efference copy.** The action-gated eligibility and the race readout need the chosen action broadcast to the action modules. How plausible is that, and what carries it?

## 9. Metrics

- **Task:** episodic return, compared with a random policy and a standard RL baseline.
- **Structure:** $L(M)$ over training; NMI between the Infomap-detected partition and the intended communities; mean exit fraction $\bar e$.
- **State model:**
  - KL divergence between the policy from $T^K(\mathbf o)$ (`derivation.md` Eq. 1d) and the spiking readout;
  - error of the implicit walk vs. the spiking estimate of $\pi_f$;
  - with carry-over, next-latent-state prediction accuracy.
- **Energy (secondary):** firing rates and the energy proxy.

## 10. Ablations

- STDP only
- task term: action-gated vs. plain eligibility; TD vs. $R - \bar{R}$
- routing: dual term vs. exemption vs. none
- $\lambda$ schedule: constant (default) vs. ramped
- presynaptic normalisation: output gain $1/d_j$ plus $d_j$ homeostasis as its own term (default, Eq. 4d) vs. homeostasis only vs. inside the structural baseline (v1)
- task rate: annealed (default) vs. constant
- policy temperature: Eq. 1d (default) vs. sharpened Eq. 1e (Proposal)
- latent-output barrier, Eq. 4e (Proposal, M4), against none; the exit floor was rejected in iteration 003
- encoding: conjunctive (default) vs. per-dimension receptive fields
- spiking readout: race with burn-in (default) vs. race from step 0 vs. argmax
- pairing count: covariance (default) vs. causal-only vs. balanced
- exact-gradient baseline (additive $G$)
- per-module vs. global modulator
- per-neuron vs. per-module baseline
- per-frame vs. running-average modulator table
- implicit vs. spiking walk
- carry-over on/off ($\rho$)
- latent communities on/off
- static vs. re-detected latent partition

## 11. Milestones

| ID | Milestone | Done when |
| --- | --- | --- |
| M0 | Docs: corrected derivation, spec, changelog; reformulated around communities as states | Done (2026-09-27, reformulated 2026-10-04) |
| M1 | **Done 2026-10-05** ([iteration 001](iterations/001-m1.md)). Numpy flow-level reference, plus SuperNeuroMAT rule prototype on a small graph with controller, latent and action modules (no task) | Implicit per-frame walk matches power iteration; spiking error against window length (in units of $\tau/\alpha$) reported; $L(M)$ decreases under the cost-modulated rule and the exact-gradient baseline; pairing counts regressed on $J_{ij}$ and $r_i r_j$; no silent or runaway runs |
| M2 | **Done 2026-10-05** ([iteration 002](iterations/002-m2.md); model v3: flow 287 ± 21, spiking 184 ± 13, random 22). CartPole, closed loop (numpy reference first, then SuperNeuroMAT) | Beats random policy; dual term vs. exemption vs. none compared; modules persist; policy from $T^K$ agrees with the spiking race readout |
| M3 | **In progress** ([iteration 005](iterations/005-m3-tasks-heads.md), [006](iterations/006-m3-reward.md), [007](iterations/007-robust-beta.md), [008](iterations/008-beta-step-gain.md), [009](iterations/009-credit-horizon.md), [010](iterations/010-decision-bound.md), [011](iterations/011-flow-diagnostics.md), [012](iterations/012-encoders-critic-entropy.md)). Acrobot, MountainCar, LunarLander; parallel spiking heads (Proposal); flow reward raised on all four tasks by $p_{cc} = 0$ and the adaptive $\beta$ (Proposal) | Runs end to end; returns reported for all ablations; wall-clock informs the simulator decision |
| M4 | Partially observable task with latent communities and carry-over | Carry-over beats $\rho = 0$; latent chain predicts the next latent state |
| M6 | Scale-up: hierarchical/multilevel map equation, richer architectures, deferred vision track | Open |

**The hybrid with the TD-STDP backbone** (iterations 013–016; derivation §8.1): the procedure for inserting recurrent action columns into an existing feed-forward spiking actor so that the decision (a race on column spike counts) and the credit (the per-neuron spike score) are carried by spikes is documented as a current-state recipe in `docs/columns-recipe.md`. Current work, not a settled milestone: its study is `hybrid-columns-5` (016 §20).
