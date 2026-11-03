# Map-STDP on four engines: flow, branching, LIF and QIF

A plain-language guide to what Map-STDP assumes, how each simulation engine meets those assumptions, and where LIF and QIF fall short. The canonical math is in [`derivation.md`](derivation.md); this guide explains it and never overrides it. Equation numbers such as (8) or (11b) and section numbers such as §5.3 refer to the derivation. For what changed in this guide and why, see the [CHANGELOG](CHANGELOG.md).

Status: 2026-10-27. Engine numbers come from frozen-weight measurements (no learning) in the derivation's Appendix A and the `mapstdp` self-checks. They are preliminary until reproduced through the repo CLI.

---

## 1. The idea in one page

**What we want.** A spiking network that solves a task by organising itself into *communities* (modules) that act as states: one module receives the stimulus (the *controller*), one module per action, and some *latent* modules. The flow of activity between modules is then a small Markov chain, a readable model of the task.

**The key move: read the spikes as a random walk.** Follow the network's own spikes. Whenever neuron $j$ fires and neuron $i$ fires shortly afterwards, count that as one step of a walker from $j$ to $i$. Over a frame, the share of spikes each neuron fires is its *flow* $\pi_j$, and the rate of steps from $j$ to $i$ is the *link flow* $J_{ij}$. Nothing about the walk is assumed: it is measured.

**Why a walk.** A walk on a graph with communities has a known shortest description, the **map equation** (Rosvall and Bergstrom 2008). If walkers mostly stay inside modules and rarely cross, the walk is cheap to describe. Community detection asks "given this graph, which partition is cheapest?" Map-STDP asks the reverse: "given this partition, change the synapses until it is the cheapest one."

**The learning rule in words.** Per frame, each synapse $j \to i$ changes by three terms:

- **Structure term:** a broadcast signal per module, $M^{\ast}$, says how many bits each unit of leakage out of that module currently costs. A synapse that carries activity *out* of its module is weakened if excitatory (strengthened if inhibitory); one that keeps activity *inside* is strengthened if excitatory (weakened if inhibitory).
- **Task term:** a TD error from a critic, times an eligibility trace that remembers which synapses pushed activity into the chosen action's module.
- **Homeostasis:** keeps each neuron's total output weight near 1 (4d), without changing where its output goes.

**What every engine must supply.** Two statistics per synapse, each with its own job (§3 below):

1. **Pair counts** $\kappa_{ij}$: how often $i$ fires in the window after $j$. They *measure the walk*, and from them the rule's baseline.
2. **The signed counter** $c_{ij}$: how many spikes of $i$ the kicks from $j$ actually *added* (or removed). It *measures the synapse's own effect*, and sets the size and sign of each step.

If both are right, the structural term is gradient descent on description length, on average (Theorem E, (10)). The engines differ only in the window and in how they compute the counter.

---

## 2. The walk as observed spikes

The derivation defines the walk from spikes (§2.1, Eq. 8). After a spike of $j$, look at the next short window $w$ and ask how many spikes $i$ fires in it, on average. That number, $E_{ij}$, has three parts:

$$
E_{ij} = a_i + D_{ij} + X_{ij}
$$

| Part | What it is | Plain meaning |
| --- | --- | --- |
| Chance $a_i = r_i w$ | the spikes $i$ would fire in the window anyway | $i$ is busy whether or not $j$ fired |
| Direct $D_{ij}$ | the effect of $j$'s kick through the synapse $j \to i$ | what this synapse does |
| Indirect $X_{ij}$ | chains $j \to k \to i$, shared input, module-wide swings | everything else that links $j$ and $i$ |

Dividing by $j$'s total gives the step probabilities $T^{obs}_{ij} = E_{ij} / E_j$, the observed walk.

**Chance is teleportation by firing rate.** The chance part does not depend on $j$ at all. It sends $j$'s walker to $i$ in proportion to how fast $i$ fires, wherever $i$ is. That is the spiking version of the map equation's teleportation: a jump that is not explained by any synapse.

**Inhibition is a below-chance transition.** An inhibitory kick from $j$ lowers $i$'s spikes in the window, so $D_{ij} < 0$. But $E_{ij}$ is the average of a count, so it can never go negative: $j$ can only remove spikes $i$ would otherwise have fired. Inhibitory synapses therefore fit inside the walk as transitions that happen *less often than chance*. No negative probabilities, no special case.

**The walk always exists.** Pair counts can be taken in any network: noisy or noiseless, linear or not, with or without inhibition. The walk needs only a frame that is roughly stationary. Stability and noise matter for the *counter*, not for the walk (§7 of the derivation).

**The flow engine is the special case.** With no spikes there is no chance and no indirect term, and every neuron has the same gain, so the observed walk is the weight-normalised walk $T = W / d$, re-injected by the stimulus:

$$
\pi = (1 - \alpha) T \pi + \alpha v(\mathbf{o}) \qquad \text{(1)}
$$

**The window and the chance share.** A longer window catches more of a spike's offspring but also more chance. The causal share of a neuron's transitions is roughly its offspring divided by (population rate × window + offspring). Measured: about 12% causal in the branching engine, about 1% in LIF, about 4–5% in QIF. Chance does not dilute the learning step (that uses the counter), but it pulls the baseline toward a module constant and keeps $M^{\ast}$ finite (§3 of the derivation).

---

## 3. Why two statistics

The rule (4) is

$$
\Delta W_{ij} = -\lambda M^{\ast}_{m(j)} \left( \mathbb{I}[i \notin m(j)] - \bar{e}^{obs}_j \right) c_{ij}
$$

It has two ingredients that look similar but answer different questions.

- **The baseline $\bar{e}^{obs}_j$ asks: where does $j$'s walker actually go?** It is the share of $j$'s observed transitions that leave its module, chance included. So it comes from pair counts. It needs only $j$'s own spikes and two population signals (spikes in its own module, spikes in the whole network), not a per-synapse coincidence.
- **The step $c_{ij}$ asks: how much would the walk change if this synapse were stronger?** Only the direct part $D_{ij}$ depends on $W_{ij}$; chance and indirect terms do not. So the step must count the synapse's own effect, not all coincidences.

**Swapping them breaks descent.**

- Pair counts as the step are mostly chance. Chance is spread evenly across targets, so renormalising each neuron's outputs cancels it, and learning stalls. In the branching E–I test (Appendix A, E4), the exact counter lowered the description length from 4.611 to 4.096 bits and cut excitatory weight leaving modules from 62% to 11%; pair counts as the step reached only 4.556 bits and 53%. On the repo's branching engine with frozen weights, the frame-averaged rule has cosine 0.999995 with the true descent direction using the counter, and 0.907 using pair counts.
- Any baseline other than the observed walk's is off direction whenever neurons differ in sensitivity (E2: cosine 0.978–0.979 instead of 1.000000).

**What the rule then guarantees.** With both statistics right, the rule's average is weight-scaled gradient descent on description length, scaled per neuron by its observed transition count, plus a rescaling of each neuron's outputs that cannot change the walk (Theorem E, (10); checked to cosine 1.0000000 by expectation, F2, and 0.999998–0.999999 with sampled counts, F4). The sign pattern follows from module membership and synapse type alone:

| Synapse | Across modules | Within module |
| --- | --- | --- |
| Excitatory | weakened | strengthened |
| Inhibitory | strengthened (lateral inhibition) | weakened |

**What each engine has to deliver.**

1. **An unbiased counter**: on average, $c_{ij}$ equals the spikes of $i$ that $j$'s kicks add or remove.
2. **A stable operating point**, so that small changes have small, predictable effects (Eq. 12): the largest eigenvalue magnitude below 1 for branching (a one-step map), the largest real part below 1 for LIF and QIF (continuous time).
3. **Enough spikes** for the pair counts and the module statistics to be reliable within a frame.

---

## 4. Flow: the exact reference

**What it is.** No spikes. Each frame, the code computes the walk's flow (1) by repeated matrix–vector products ("hops"); twenty hops get within 2.3% of the exact answer (1a). The counter is replaced by its expectation, the link flow $J_{ij} = \pi_j T_{ij}$, and the baseline by the weight shares $\bar{e}_j$.

**How it meets the rule.** By construction. With no chance and no indirect terms, the observed walk is $T$, the baseline is exact, and the rule reduces term for term to Theorem 3: weight-scaled gradient descent (§5.1). The implemented rule is unchanged and bit-identical in code, so every flow result stands.

**Measured.** Theorem E's sampled-count check: cosine 0.999999 (F4). Task credit from the one-hop eligibility (4b): cosine 0.98–0.998 with the exact score on untrained weights, 0.83–0.99 on trained ones.

**Limitations.**

- **It is not a spiking network.** It shows what the rule does when every assumption holds, not whether neurons deliver it.
- **The flow is held fixed when taking gradients** (the local split). On test networks the resulting direction keeps cosine 0.55–0.92 with the full gradient.
- **Excitatory only.** Flow with inhibition is a Proposal (§5.1), untested.
- **It is cheap** (about 0.5–1.3 ms per frame), which makes it the development tool for every other engine.

**Results so far (development, not a reported study).** Model v4 on flow: CartPole 40/40 solved (median 296 episodes to solve), LunarLander 8/40 solved within 1000 episodes (trailing return 88), without Adam.

---

## 5. Branching: the implemented spiking engine

**What it is.** SuperNeuroMAT's LIF neuron with the leak set to infinity (the membrane forgets everything each step) and uniform threshold noise. Each step a neuron fires with probability proportional to its input, $P(\text{spike}) = u / 2 \theta$, clipped to $[0, 1]$. Outputs are scaled by $1 / d_j$ and by 0.8, so one step is one hop of the walk and each spike has on average 0.8 offspring.

**How it meets the rule.** Exactly. The window is one step. Because the neuron already knows its summed input $u_i$, it can compute what its firing probability would have been *without* $j$'s kick. The difference is the counter (11a):

$$
c_{ij} = \sum_t x_j(t) \left[ p \left( u_i(t + 1) \right) - p \left( u_i(t + 1) - b_{ij} \right) \right]
$$

No linearisation, no gain estimate: one subtraction and one clip per presynaptic spike. For inhibition it removes only the probability $i$ actually had.

**Measured.**

| Check | Value |
| --- | --- |
| Rule average vs true descent direction (frozen weights), counter as the step | cosine 0.999995 |
| Same, pair counts as the step | cosine 0.907 |
| Causal share of transitions | about 12% |
| Stability measure (largest eigenvalue magnitude, map form) | 0.79 |
| Lag-1 offspring on $0.8 T$ (CartPole) | slope 1.04, correlation 0.94 |
| E–I test: negative transitions; inflow imbalance | none; 1.5% mean, 4.2% max |

**Limitations.**

- **Sampling noise.** A frame's estimate of module flow has error about $1.3 / \sqrt{S}$, with $S$ the spikes per module. More spikes cost more time.
- **Escape noise hurts precise control.** In the TD-STDP baseline, branching neurons learn as fast as deterministic LIF but fail CartPole's strict criterion (100 perfect episodes in a row) in 15 of 20 seeds; the noise-free version of the same rule solves 20/20.
- **Saturation.** Linearity breaks where input exceeds $2 \theta$; the exact counter still holds there, but the walk is no longer a simple multiple of $T$.
- **Engineering conveniences.** The $1/d_j$ output gain and network-wide normalisation stand in for inhibitory gain control; they are abstractions.
- **Closed loop.** The 0.999995 is on frozen weights; closed-loop results so far are excitatory-only.

**Results so far.** Model v4 spiking (16 heads) on CartPole: last-100 return 154. TD-STDP on branching: CartPole 5/20 solved (strict), LunarLander 6/10 with Adam, 0/10 without.

---

## 6. LIF: the leaky integrate-and-fire neuron

**What it is.** The textbook neuron: the membrane integrates input and leaks back to rest, and it spikes and resets at a threshold. The repo's LIF engine (`mapstdp/lif.py`, §5.3) uses a 20 ms membrane time constant, membrane noise standing for background input, signed kicks, and a 60 ms window (about three response times).

**How it meets the rule.** The walk comes from pair counts, as always. The counter cannot be computed exactly, because a LIF neuron's response to one kick is spread over time and depends on its whole history. Instead the engine uses the **mean counter** (11b):

$$
c_{ij} = \tau_m \hat{G}_i b_{ij} n_j
$$

presynaptic spike count × kick size × postsynaptic gain. In words: each kick is a small pulse of input, and a neuron's gain says how many extra spikes a pulse of that size buys. The gain $\hat{G}_i$ comes from the Siegert formula (11c), which predicts a noisy LIF neuron's rate and gain from the mean and spread of its input.

**Measured.**

| Check | Value |
| --- | --- |
| Siegert rate and gain vs simulation | within 1% (0.998–1.008; 0.993–1.007) |
| Counter vs weight-scaled derivative | correlation 0.945, ratio 1.04 |
| Removing a synapse outright vs the derivative | about 0.65 of it (counter/removal ratio 1.54, correlation 0.940) |
| Pair counts vs pure chance, $E_{ij} / (r_i w)$ | 0.99 (causal share about 1%) |
| Stability (largest real part of the module offspring table) | 0.59–0.64 |
| Single neuron, inhibitory and excitatory kicks (E5) | simulated/predicted 0.85–1.05 |

**Limitations in plain words.**

- **The counter needs noise.** A noiseless LIF neuron is all-or-nothing at threshold, so "how many extra spikes does a small kick buy" has no smooth answer. With noise the answer is smooth and Siegert predicts it. The walk itself does not need noise.
- **The counter measures a derivative, not a removal.** Near threshold the transfer curve bends upward on the scale of one kick, so deleting a synapse changes less (about 0.65×) than the slope suggests. The rule needs the slope, which the counter gets right (ratio 1.04); just do not read the counter as "what this synapse is worth if removed".
- **Offspring must stay low.** Without inhibition, the curved LIF transfer ignites a high-rate state well before 0.8 offspring per spike, so the excitatory-only LIF engine runs at 0.5, not the branching engine's 0.8.
- **Pair counts are almost all chance.** At 5–10 Hz and a 60 ms window, about 99% of observed transitions are chance. The baseline then sits near the module constant $1 - p_{m(j)}$, and $M^{\ast}$ near its chance ceiling. The map term still points the right way, but it cannot tell well-sealed modules from poorly sealed ones as sharply as in branching.
- **The race readout saturates.** The race-to-bound action readout (1h) reaches an effective sharpness of only about $\beta \approx 2.35$ at 16 heads; sharper policies need more spikes.
- **Gains are an abstraction.** The Siegert gain assumes a neuron knows its own operating point. A regression alternative is untested and biased by recurrent feedback.
- **No reset-aware time-resolved counter.** Weighting each kick by the neuron's instantaneous sensitivity sounds more precise, but with reset it overstates the effect several-fold (each added spike's reset cancels most of the gain), so the engine uses the mean counter.
- **First order only.** Excitatory kicks do slightly more than predicted and inhibitory kicks slightly less; separate gains for the two would fix most of this (untested).

---

## 7. QIF: the quadratic integrate-and-fire neuron

**What it is.** A neuron whose voltage grows quadratically near threshold, so spikes are sharp, self-generated events, close to real cortical neurons. Its population behaviour has an exact mean-field description (Montbrió, Pazó and Roxin 2015). The repo's QIF engine (`mapstdp/qif.py`, §5.4) adds fixed inhibitory pools that keep it stable; these pools sit outside $W$ and outside the walk.

**How it meets the rule.** As for LIF: pair counts for the walk, and the mean counter (11b) with the QIF time constant in place of $\tau_m$. The gain has no closed form at the operating point, so it is **regressed**: each neuron's rate is regressed on the injected noise current across frames. The noise is independent of the network, so the estimate is unbiased; per-neuron estimates are shrunk toward their module's.

**Measured.**

| Check | Value |
| --- | --- |
| Counter vs removing the synapse | correlation 0.89, ratio 1.04 |
| Noise-regressed gain vs finite-difference slope | correlation 0.93, ratio 1.01 |
| Causal share of transitions | about 4–5% |
| Stability (largest real part of the module offspring table) | 0.87 |

**Limitations in plain words.**

- **The gain must be regressed on the noise, not on the input.** The neuron's own spikes recruit the inhibitory pools, so its total input goes *down* when it fires more. Regressing rate on total input then gives a gain of the wrong sign. Only the injected noise is independent of the neuron. Knowing which part of the input is noise is an **abstraction** (on hardware, the injected noise is known).
- **Inhibition is fixed and outside the rule.** The pools are not plastic and are not part of the walk. QIF with inhibitory neurons inside $W$ is untested.
- **The excitatory part may be supercritical on its own.** At the tested point the excitatory block has 1.1–1.7 offspring per spike; inhibition holds the whole system stable (an inhibition-stabilised network, a real cortical regime). This is allowed now: the walk is observed, so it exists anyway, and the counter needs only the full system to be stable (Eq. 12). The risk is that the map term strengthens within-module excitation and weakens within-module inhibition, both of which push toward instability (§7 of the derivation).
- **Noise is still needed** to keep gains bounded and smooth and to break phase locking, which the counter's average assumes away.
- **The time-resolved sensitivity is not used.** Its shape matched ground truth only at 0.70 at the noisy operating point.

---

## 8. Limitations of LIF and QIF, in one place

Both realistic neurons can host the rule, because the walk is measured and the counter only needs a gain. What they cost:

| Issue | LIF | QIF |
| --- | --- | --- |
| Counter accuracy | corr 0.945 vs derivative | corr 0.89 vs removal |
| Where the gain comes from | Siegert formula (abstraction) | regression on injected noise (abstraction) |
| Needs noise | yes, for the counter | yes, for the counter and against phase locking |
| Causal share of pair counts | about 1% | about 4–5% |
| Consequence of chance | baseline ≈ module constant; $M^{\ast}$ near its chance ceiling | same, slightly less |
| Stability | 0.59–0.64; offspring capped at 0.5 (ignites near 0.8) | 0.87 with fixed inhibition; E block 1.1–1.7 allowed |
| Inhibition | none in the tested engine | fixed pools, outside $W$ and the walk |
| Tested | frozen weights only | frozen weights only |

The plain summary: **in branching the rule is exact; in LIF and QIF it is a good first-order estimate that leans on noise and on knowing each neuron's gain.** Chance dominates the observed walk in both, which blunts the map term's sense of how sealed each module is, but not its direction.

**TD-STDP's noiseless LIF.** The TD-STDP baseline (Chung and Kozma 2020) uses a deterministic LIF neuron. Its walk exists, since pair counts always exist. Its counter does not: without noise there is no smooth gain. Running Map-STDP there needs input noise or a surrogate gain.

---

## 9. Side by side

| | Flow | Branching | LIF | QIF |
| --- | --- | --- | --- | --- |
| Spikes | none (expectations) | stochastic, one step = one hop | noisy, $\tau_m$ 20 ms | sharp, self-generated |
| Window | one hop | one step | 60 ms | about 3 response times |
| Counter | expectation $J_{ij}$ | exact (11a) | mean (11b), Siegert gain | mean (11b), regressed gain |
| Counter accuracy | exact | cosine 0.999995 (rule average) | corr 0.945 | corr 0.89 |
| Causal share | 100% (no chance) | about 12% | about 1% | about 4–5% |
| Stability | 0.8 by construction | 0.79 | 0.59–0.64 | 0.87 (with inhibition) |
| Inhibition | none (Proposal) | E–I tested standalone | none | fixed pools |
| Closed-loop results | yes (v4) | yes (v4, excitatory) | not yet | not yet |
| Biological realism | lowest | Poisson-like noise, memoryless | realistic membrane | most realistic |

The pattern: **all four engines share one walk and one rule; what changes is how cheaply and how exactly each neuron can say what one kick did.** Flow and branching know it exactly. LIF and QIF estimate it from a gain, and their long windows fill the walk with chance.

---

## 10. What this means for the experiments

- **Fair comparison with TD-STDP** runs both rules on the same engine. On flow and branching Map-STDP is on solid ground. On TD-STDP's deterministic LIF, add input noise or a surrogate gain first, and say which.
- **LIF and QIF are ready for closed-loop tests**, with frozen-weight checks as the bar to keep passing: counter correlation, gain error and the stability measure, each frame.
- **Expect a weaker map signal on LIF and QIF.** With 95–99% chance, $\bar{e}^{obs}$ barely moves; if sealing matters for a task, shorten the window as far as the offspring allow, or compare against branching at matched settings.
- **Action readout on LIF.** The race-to-bound saturates near $\beta \approx 2.35$ at 16 heads; adaptive $\beta$ (1g) above that needs more heads or a different readout.
- **Inhibition.** Monitor the $2K \times 2K$ offspring table each frame (Eq. 12). Within-module inhibition is what the map term weakens and what stability may need; exempting it is the first remedy to test.
- **The local split is unmeasured in spiking engines.** Its cosine with the full gradient is known only for flow (0.55–0.92).

---

## Glossary

- **Walk / walker:** an imaginary token that steps from $j$ to $i$ whenever $i$ fires in the window after $j$; the map equation describes its path.
- **Window ($w$):** how long after a spike of $j$ a spike of $i$ still counts as a step: one step in branching, about three response times in LIF and QIF.
- **Flow ($\pi$):** each neuron's share of the frame's spikes.
- **Link flow ($J$):** the walk's rate of steps from one neuron to another.
- **Chance:** steps that happen because the target fires anyway; the spiking form of teleportation.
- **Causal share:** the part of a neuron's steps caused by its own kicks rather than chance.
- **Pair count ($\kappa$):** how often $i$ fires in the window after $j$; measures the walk and gives the baseline.
- **Signed counter ($c$):** the spikes of $i$ that $j$'s kicks added (positive) or removed (negative); gives the step.
- **Baseline ($\bar{e}^{obs}$):** the share of $j$'s observed steps that leave its module.
- **Gain ($G$):** how many extra spikes a neuron fires per unit of extra input.
- **Offspring:** extra spikes caused by one extra spike; above 1 per spike in the excitatory part alone means it needs inhibition to stay stable.
- **Stability measure:** the largest eigenvalue (magnitude for branching, real part for LIF and QIF) of the module offspring table; below 1 is stable.
- **Map equation:** the description length of a walk given a partition into modules; low when walkers rarely leave their module.
- **Modulator ($M^{\ast}$):** the per-module broadcast saying how many bits a unit of leakage costs.
- **Local split:** taking gradients with the flow, chance, indirect terms and gains held fixed.
- **Siegert formula:** a closed-form rate (and gain) for a noisy LIF neuron from the mean and spread of its input.
