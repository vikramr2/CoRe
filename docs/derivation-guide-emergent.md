# Map-STDP Derivation Guide: Emergent-Walk Edition

Oct 26, 2026 · @Vikram Ramavarapu

## 1. The reframing in one page

The walk is now the network's own spike sequence. A step from j to i means "i fired within a short window after j fired". Its transition probabilities are measured from the spikes rather than built from the weights, so they are non-negative whatever the signs of the synapses, and inhibition fits inside the walk.

This follows the map equation's own framing. Rosvall and Bergstrom call the random walk "a proxy for real flow", and later work runs the map equation directly on observed trajectories. Here the observed trajectories are spikes.

**What stays the same**

- The map equation, the marginal exit cost M\*, the EM split (flow held fixed while weights change) and the fixed partition.
- The three-factor shape of the rule, the routing exemption, the action readout and the TD-modulated task term.
- The flow engine. For an excitatory-only network it gives exactly the old rule, so its results stand.

**What changes**

1. **The walk.** Link flow is the expected pair count: how often a spike of i follows a spike of j within the window. Chance coincidences are part of it and act like teleportation in proportion to firing rate. Inhibition shows up as transitions below chance.
2. **The two statistics in the rule have separate jobs.** Pair counts κ\_ij estimate the walk, and therefore the baseline. A signed offspring counter c\_ij estimates how the walk changes when W\_ij changes, and therefore sets each synapse's step.
3. **Inhibitory synapses learn with the same rule.** Their sign flips the effect: the map term strengthens inhibition between modules and weakens it within a module.
4. **The stability condition is the whole E–I system's.** The excitatory block alone no longer has to be subcritical, so inhibition-stabilised operating points are allowed.

**Why the walk is sound.** P(i fires in the window after j) is an expected count, so it cannot be negative. Inhibition lowers it, and when i is silenced it reaches zero and stops: a neuron can only lose spikes it would otherwise have fired. The linearised effect of an inhibitory synapse can be negative, but the probability it modifies cannot.

**The rule in one line**

```latex
\Delta W_{ij} = \eta\,\delta_f\,e_{ij} \;-\; \lambda\,M^*_{m(j)}\big(\mathbb{1}[i\notin m(j)] - \bar e^{\,obs}_j\big)\,c_{ij} \;-\; \epsilon_h\,(d_j-1)\,\frac{W_{ij}}{d_j}\,n_j
```

W\_ij is a non-negative magnitude, and the presynaptic neuron's type fixes the sign (Dale's law). c\_ij carries that sign. ē^obs\_j is the fraction of j's observed transitions that leave its module, and n\_j is j's spike count. Every update to W\_ij is proportional to W\_ij, so a synapse never changes sign.

**Status.** The derivation is checked numerically on small models (Derivation Details, D12). Nothing here has been run in the repo.

## 2. Notation

Synapses run from presynaptic j to postsynaptic i. A frame is one environment step of length T\_f.

| Symbol | Meaning | Lives at |
| --- | --- | --- |
| W\_ij ≥ 0 | magnitude of the synapse j → i | synapse |
| sgn\_j = ±1 | type of j: +1 excitatory, −1 inhibitory (Dale's law) | neuron j |
| d\_j = Σ\_i W\_ij, T\_ij = W\_ij / d\_j | j's total output; j's share to i | neuron j; synapse |
| b\_ij = sgn\_j w\_0 T\_ij | signed kick one spike of j gives i (output-normalised) | synapse |
| r\_j, n\_j, π\_j | firing rate; spike count in the frame; spike share r\_j / Σ r | neuron j |
| w | the walk's window: how soon after j a spike of i counts as a step | global |
| E\_ij ≥ 0 | expected spikes of i in the window after a spike of j | synapse |
| a\_i = r\_i w | the chance part of E\_ij | neuron i |
| D\_ij | the direct part, through the synapse j → i (signed) | synapse |
| X\_ij | the indirect part: chains through other neurons and common input | synapse |
| E\_j = Σ\_i E\_ij, T^obs\_ij = E\_ij / E\_j | j's total; observed transition probability | neuron j; synapse |
| J\_ij = π\_j T^obs\_ij | link flow of the observed walk | synapse |
| κ\_ij | pair count: spikes of i in the window after spikes of j | synapse |
| c\_ij | signed offspring counter: spikes of i added (or removed) by j's kicks | synapse |
| G\_i, ω\_i | gain of i; share of i's response that lands inside the window | neuron i |
| Λ̃\_ij = χ ω\_i G\_i T\_ij | size of the windowed direct effect, so D\_ij = sgn\_j Λ̃\_ij | synapse |
| Λ\_ij | total (unwindowed) signed offspring, used for task credit | synapse |
| m(j), K | module of j; number of modules | neuron j |
| p\_m, q\_m, M\*\_m | module visit rate; module exit rate; marginal bits per unit of exit flow | module |
| ē^obs\_j | share of j's observed transitions that leave m(j) | neuron j |
| Φ\_i | cascade value: policy score summed over a spike's descendants | neuron i |

χ collects the constants that turn output weight into kick size (w\_0 / 2θ for branching, w\_0 τ\_m for LIF).

## 3. Step 1: spikes define the walk

A step of the walk from j to i is a spike of i within the window w after a spike of j. Its weight is an expected spike count, which is never negative, so the walk exists for any mix of excitation and inhibition.

**The transition.** After a spike of j, i fires E\_ij times on average within the window. That count has three parts:

```latex
E_{ij} \;=\; \underbrace{a_i}_{\text{chance}} \;+\; \underbrace{D_{ij}}_{\text{direct, via } j\to i} \;+\; \underbrace{X_{ij}}_{\text{chains, common input}} \;\ge\; 0, \qquad a_i = r_i\,w
```

Normalising each column gives the observed walk, and the spike shares give its flow:

```latex
T^{obs}_{ij} = \frac{E_{ij}}{\sum_k E_{kj}}, \qquad \pi_j = \frac{r_j}{\sum_k r_k}, \qquad J_{ij} = \pi_j\,T^{obs}_{ij}
```

- **Pair counts measure it exactly.** E\[κ\_ij\] = n\_j E\_ij, so κ\_ij / n\_j is an unbiased estimate of the walk. In the old framing the chance and indirect terms were bias. Here they are part of the walk.
- **Chance acts as teleportation.** a\_i does not depend on j, so it sends j's walker to i in proportion to i's rate. That is PageRank-style teleportation by firing rate.
- **Inhibition lowers transitions.** For an inhibitory j, D\_ij < 0, so i is less likely than chance to follow j. The count stops at zero because a silenced neuron has no spikes left to lose. The walk never needs a negative probability.
- **Flow balance.** Outflow from j is exactly π\_j. Inflow to i can differ slightly from π\_i, and that difference is treated as unrecorded teleportation, as the old guide treated the stimulus term.

**What the old construction needed and this one does not.** The old walk was built from the offspring matrix Λ and needed Λ ≥ 0, ρ(Λ) < 1 and a non-negative immigration term, so it excluded inhibition. The observed walk needs only a stationary frame. Stability still matters for learning (Section 9) but not for the walk to exist.

**Choosing the window.** Branching uses one time step, which is its natural hop. For LIF and QIF, use about three response times, so most of a spike's direct effect lands inside the window (ω\_i ≈ 1). A longer window adds chance: the causal share of j's transitions is roughly ν\_j / (R w + ν\_j), with R the total population rate. With 100 neurons at 10 Hz and w = 30 ms that share is about 3%. Chance does not change which way any synapse moves, but it lowers M\* and so slows learning.

## 4. Step 2: the cost of describing the walk

This step is unchanged. With the flow held fixed, the weights affect the map equation only through the module exit rates, and each unit of exit flow costs M\* ≥ 0 bits.

```latex
L = q_{\curvearrowright}\log q_{\curvearrowright} - 2\sum_m q_m\log q_m - \sum_j \pi_j\log\pi_j + \sum_m (p_m+q_m)\log(p_m+q_m)
```

```latex
p_m = \sum_{j\in m}\pi_j, \qquad q_m = \sum_{j\in m}\sum_{i\notin m} J_{ij} = \sum_{j\in m}\pi_j\,\bar e^{\,obs}_j, \qquad M^*_m = \frac{\partial L}{\partial q_m} = \log\frac{q_{\curvearrowright}(p_m+q_m)}{q_m^2}
```

Two things differ in practice:

- **q\_m now includes chance.** The chance part of j's transitions leaves the module with probability about 1 − p\_m. In an excitatory-only network q\_m therefore never reaches zero and M\* stays bounded. Inhibition between modules can push exit flow below chance, so keep the clip at 20.
- **M* is estimated from pair counts.*\* A module's exit rate is the share of its pair counts that land outside it, a K-number statistic that can be broadcast as before.

The partition stays fixed (controller, actions, latents), and latent modules can still be re-detected now and then. The objective is D(W) = E\_f\[ L(M; J\_f) \], averaged over frames, with J\_f the observed walk of frame f.

## 5. Step 3: the gradient

The gradient has the same shape as before — presynaptic flow × direct effect × (modulator minus a baseline) — with two changes. The presynaptic sign multiplies it, and the baseline is the observed walk's.

**The local split.** Hold fixed everything a single synapse does not touch directly: the flow π, the chance term a, the indirect term X, and each neuron's gain and window share. This extends the old EM split, which held π fixed. With these held, W\_ij changes only the direct part of column j:

```latex
D_{kj} = \mathrm{sgn}_j\,\tilde\Lambda_{kj}, \qquad \tilde\Lambda_{kj} = \chi\,\omega_k G_k\,\frac{W_{kj}}{d_j}
```

**Lemma E.** With g\_ij = M\*\_{m(j)} 1\[i ∉ m(j)\] the marginal cost of exit flow:

```latex
W_{ij}\,\frac{\partial L}{\partial W_{ij}} \;=\; \frac{\pi_j\,\mathrm{sgn}_j}{E_j}\Big[\tilde\Lambda_{ij}\,\big(g_{ij} - \bar g^{\,obs}_j\big) \;-\; T_{ij}\,C_j\Big], \qquad \bar g^{\,obs}_j = \sum_k T^{obs}_{kj}\,g_{kj} = M^*_{m(j)}\,\bar e^{\,obs}_j
```

C\_j = Σ\_k Λ̃\_kj (g\_kj − ḡ^obs\_j) is the same for every i in column j. The term T\_ij C\_j is therefore a uniform rescaling of column j. It cannot change T, so it cannot change L, and the homeostasis term absorbs it.

**Reading it term by term.**

- **π\_j / E\_j** is presynaptic flow per observed transition: a per-neuron step size.
- **sgn\_j** flips the direction for inhibitory synapses. Strengthening an inhibitory synapse onto a target outside the module lowers exit flow, so the map term strengthens it.
- **Λ̃\_ij** is the direct effect: the step scales with how much j actually moves i.
- **g\_ij − ḡ^obs\_j** compares this synapse's marginal cost with the average over j's observed transitions, chance included.

**Old rule as a special case.** With no chance or indirect terms (a = X = 0) and only excitation, T^obs becomes the gain-weighted walk of the old Step 6, and Lemma E reduces exactly to Lemma K. With equal gains it reduces to the original Lemma (Details, D4).

**Checked.** Against finite differences on a 16-neuron E–I network with chance and indirect terms: relative error 7 × 10⁻⁷ (Details, D12).

## 6. Step 4: from gradient to a local rule

The rule needs two statistics with different jobs. Pair counts measure the walk, which gives the baseline. A signed offspring counter measures each synapse's direct effect, which gives the step. Swap either one and the rule stops descending.

**The structure rule.**

```latex
\Delta W_{ij} \;=\; -\lambda\,M^*_{m(j)}\,\big(\mathbb{1}[i\notin m(j)] - \bar e^{\,obs}_j\big)\,c_{ij}, \qquad \bar e^{\,obs}_j = \frac{\sum_{i\notin m(j)}\kappa_{ij}}{\sum_i \kappa_{ij}}
```

**Theorem E.** If the counter is unbiased for the windowed direct effect, E\[c\_ij\] = n\_j D\_ij, then:

```latex
\mathbb{E}[\Delta W_{ij}] \;=\; -\lambda\,\zeta\,E_j\,W_{ij}\,\frac{\partial L}{\partial W_{ij}} \;+\; (\text{rescaling of column } j), \qquad \Delta L = -\lambda\zeta\sum_{ij} E_j\,W_{ij}\Big(\frac{\partial L}{\partial W_{ij}}\Big)^2 \le 0
```

ζ = T\_f Σ r is a global constant. The rescaling term is orthogonal to the gradient, so it does not affect the descent.

| Statistic | Estimates | Used for | If you use the other one |
| --- | --- | --- | --- |
| Pair count κ\_ij | the walk, n\_j E\_ij | ē^obs\_j and M\* | as the step: learning stalls (test: code length 4.61 → 4.56 bits in 40 blocks, against 4.61 → 4.10) |
| Signed counter c\_ij | the direct effect, n\_j D\_ij | the step | as the baseline weights: direction off (cosine 0.978 with the gradient) |

**Properties.**

- **Dale's law is kept.** c\_ij is proportional to W\_ij, so updates are multiplicative and a synapse never crosses zero.
- **Inhibition needs nothing extra.** c\_ij carries sgn\_j. The same formula strengthens excitation and weakens inhibition within a module, and does the opposite across modules.
- **Locality.** c\_ij needs the presynaptic spikes and the postsynaptic gain (Section 8). ē^obs\_j needs j's own spikes plus a broadcast of module activity: after each spike of j, count population spikes in the window and the share outside m(j). M\* is a K-number broadcast, as before.

## 7. Step 5: the full rule, readout and Markov chain

The full rule keeps its three terms. The task term uses the signed counter with the cascade value, and homeostasis uses a presynaptic-only factor.

```latex
\Delta W_{ij} = \underbrace{\eta\,\delta_f\,e_{ij}}_{\text{task}} \;-\; \underbrace{\lambda\,M^*_{m(j)}\big(\mathbb{1}[i\notin m(j)] - \bar e^{\,obs}_j\big)\,c_{ij}}_{\text{structure}} \;-\; \underbrace{\epsilon_h\,(d_j-1)\,\frac{W_{ij}}{d_j}\,n_j}_{\text{homeostasis}}
```

**Task term.** The policy depends only on firing rates, so the task score is unchanged by the reframing. It now uses signed offspring:

```latex
e_{ij} \leftarrow \Big(1-\tfrac{1}{\tau_e}\Big)e_{ij} + c^{tot}_{ij}\,\Phi_{m(i)}, \qquad \Phi^K = \big(I - \hat\Lambda^{K\top}\big)^{-1} s^K, \qquad \hat\Lambda^K_{ba} = \frac{\sum_{i\in b}\sum_{j\in a} c^{tot}_{ij}}{N_a}
```

- **c^tot** counts a spike's direct offspring over all time, not just inside the window. With a window of about three response times the two are nearly equal, so one counter serves both terms.
- **Λ̂^K is signed.** Inhibitory modules contribute negative entries, so credit flows through inhibitory paths too. Solve the K × K system directly; the Neumann series needs ρ < 1, and the solve does not.
- **Exactness.** With the exact per-neuron Φ and signed Λ, the eligibility matches the true score up to a column rescaling (cosine 1 in tests with inhibition). The module-level Φ^K reached cosine 0.96–0.99.
- **One-hop credit** (h\_m(i) minus a baseline) is still fine when gains are equal.

**Homeostasis.** W\_ij n\_j / d\_j is an exact column rescaling. It conserves d\_j and leaves the walk alone. Using c\_ij here would move T toward low-gain targets whenever gains differ.

**Routing exemption.** Controller → action and controller → latent synapses keep g = 0, as before.

**Readout.** Unchanged: P(a | o) = p\_a / Σ p\_a', the spike share of each action column, sampled by a race between columns.

**The extracted Markov chain.** Coarse-graining the observed walk gives T^K\_ba = J\_ba / p\_a. This is now the module-level chain of the actual spike sequence. It includes chance, so its diagonal is weaker than the old causal chain's. The causal part alone is Λ̂^K.

## 8. The four engines

Every engine now uses the same rule. What differs is the window, how the counter c\_ij is computed, and what each engine needs for that counter to be accurate.

|  | Flow | Branching | LIF | QIF |
| --- | --- | --- | --- | --- |
| Window w | one hop (expectations) | one time step | about 3 response times | about 3 response times |
| Chance term a\_i | none (excitatory only); w r\_i if inhibition is added | r\_i per step | r\_i w | r\_i w |
| Counter c\_ij | expectation sgn\_j Λ̃\_ij r\_j | exact: Σ\_t x\_j(t) \[p(u\_i) − p(u\_i − b\_ij)\] | τ\_m Ĝ\_i b\_ij n\_j (b\_ij carries the sign) | τ Ĝ\_i b\_ij n\_j |
| Needs a gain estimate | no | no (read off the neuron's input) | yes: Siegert or regression | yes: regression |
| Needs noise | no | built in | for the counter, not the walk | for the counter and against phase locking |
| Stability condition | max Re λ(Λ) < 1 | ρ(Λ) < 1 (one-step map) | max Re λ(Λ) < 1 | max Re λ(Λ) < 1; an E block above 1 is allowed if inhibition holds it |
| Tested here | reduces to the old rule | E–I: counter, sign pattern, learning | excitatory and inhibitory kicks (single neuron) | excitatory kicks (single neuron) |

**Flow.** For an excitatory-only network there are no spikes and so no chance, and Lemma E reduces exactly to the old Lemma. Keep the flow engine as it is; its results (CartPole 40/40, LunarLander 8/40) stand. To add inhibition, compute rates from the rectified rate fixed point and add a chance floor a\_i = w r\_i matching the spiking engine's window. Without that floor the expected walk could have negative entries, and the floor also makes the flow engine predict the spiking engines' objective.

**Branching.** The window is the natural one-step hop. Use the exact finite-difference counter; the neuron already computes u\_i, and p(u\_i − b\_ij) costs one extra clip. The simpler gate 1\[0 < u\_i < 2θ\] understates inhibition by 16% in the E–I test, because an inhibitory kick near zero removes only what is left. In that test, the rule cut excitatory weight leaving each module from 62% to 11%, moved inhibitory output to other modules (76% → 98%), and kept rates steady.

**LIF.** One kick of size b changes i's total spike count by τ\_m G\_i b to first order, for either sign. The counter therefore needs only the presynaptic spike count and a gain estimate (Siegert from the neuron's input statistics, or regression across frames). It is second-order accurate: on a noisy single neuron, inhibitory kicks gave 12% fewer lost spikes than predicted at 7–14 Hz and 2% fewer at 33 Hz, while excitatory kicks gave 5% more. A separate gain estimate for inhibitory inputs would correct most of this. The walk itself is defined even for TD-STDP's noiseless LIF; only the counter needs a smooth response, from noise or a surrogate gain.

**QIF.** Use the same mean counter, τ Ĝ\_i w\_ij n\_j, at the noisy operating point. The time-resolved phase-response gate has the wrong shape there. The reframing removes the old disqualifier: an excitatory block with ρ of 1.1–1.7 held stable by inhibition is allowed, as long as the full E–I linearisation is stable. Noise is still needed against phase locking and to bound the gain.

## 9. Inhibition

Inhibition enters in three places: as below-chance transitions in the walk, as a sign flip in the rule, and as negative entries in the credit cascade. The new requirement is stability of the whole E–I system, and the main new risk is that the map term weakens the inhibition inside each module.

**What the map term does to inhibition.** For an inhibitory neuron in module m, the rule:

- strengthens its synapses onto other modules, because suppressing them lowers exit flow, and
- weakens its synapses onto its own module, because suppressing them lowers within-module flow.

Left alone, inhibitory neurons become purely lateral, and modules separate by excitation within and inhibition between them. In the branching E–I test, inhibitory output to other modules rose from 76% to 98% over 40 blocks.

**The stability condition.** The walk exists regardless, but linear response, and so the counters, need a stable operating point. For LIF and QIF, whose rate dynamics are τ dr/dt = −r + F(Br + I), the condition is the one below. Branching is a one-step map and needs the stricter ρ(Λ) < 1, with every eigenvalue inside the unit circle (Details, D10):

```latex
\max_k \,\mathrm{Re}\,\lambda_k(\Lambda) \;<\; 1 \qquad (\text{full signed } \Lambda,\; E \text{ and } I)
```

- **Excitation-only bound.** The old bound ρ(Λ\_EE) ≤ χ G\_max still guarantees stability with no help from inhibition. It is sufficient but no longer necessary.
- **Inhibition-stabilised regime.** An excitatory block with ρ(Λ\_EE) > 1 is allowed when inhibition holds the full system stable. This is QIF's regime, and plausibly cortex's.
- **Monitoring.** Check the condition on the module-level signed Λ̂^K, which the counters already provide.

**The risk, and a fix.** In the inhibition-stabilised regime, the inhibition inside a module is what keeps it stable. The map term weakens exactly those synapses while strengthening the excitation inside the module. Two remedies, neither tested:

- Exempt inhibitory → same-module synapses from the structure term (g = 0, like the routing exemption).
- Set those synapses with an inhibitory homeostatic rule that holds excitatory rates at a target. Vogels et al. (2011) proposed such a rule (cited from memory).

The excitation-only bound avoids the problem entirely, at the cost of excluding the inhibition-stabilised regime.

**Subtractive and divisive inhibition both fit.** The walk is observed, so the form of inhibition only changes the counter's gain. Divisive (shunting) inhibition lowers G\_i. Subtractive inhibition lowers the input and, through rectification, G\_i as well. The excitatory–inhibitory branching process (Corral López et al.) and transmission neural networks (Gao and Caines) are literature models that keep firing probabilities non-negative in these two ways.

## 10. Hybrid with TD-STDP

TD-STDP's backbone stays: Fourier input spikes, a feed-forward controller, TD error times an action-gated eligibility, and a spiking critic. Map-STDP adds recurrent modules inside the actor, shaped by the map term, and under the new walk those modules can contain inhibitory neurons.

&#91;embedded content: TD-STDP backbone with Map-STDP modules · 4 broadcast signals\]

Solid arrows carry spikes and the dashed arrow carries the broadcasts. Routing out of the controller learns from the TD error alone. Inside the shaded block the structure term also acts. Decision units read column spikes but send nothing back, so they stay outside the walk.

**Which rule acts on which synapses**

| Block | Structure term | Task term | Note |
| --- | --- | --- | --- |
| C → A, C → L (routing) | exempt | δ × eligibility | TD-STDP's own pathway |
| Excitatory, within a module | strengthens | δ × eligibility | persistent column activity feeds the race |
| Excitatory, across modules | weakens | δ × eligibility | separates columns and latent states |
| Inhibitory, within a module | weakens, or exempt in the inhibition-stabilised regime | δ × eligibility | see Section 9 |
| Inhibitory, across modules | strengthens (lateral inhibition) | δ × eligibility | new under the observed walk |
| Inputs → critic | none | critic rule | outside the walk |
| Columns → decision units | fixed | none | outside the walk |

**What changed from the previous guide's Section 14**

- **Inhibition among action columns.** It used to be excluded because it put negative entries into the walk. Under the observed walk it is allowed, and the map term grows it. The decision units' mutual inhibition still sits outside the walk, because they do not project back.
- **Eligibility.** The task eligibility is e\_ij += c^tot\_ij Φ, with the signed counter. For the exact Hawkes-LIF the counter is presynaptic-only.
- **A fourth broadcast.** Module activity lets each neuron estimate ē^obs\_j (Details, D5 Step 9).
- **Stability monitor.** The 2K × 2K module-level E–I offspring matrix is checked each frame (Details, D10 Step 7).
- **Neuron.** Noisy LIF or Hawkes-LIF first, QIF later, as before.

## 11. What changed, what is open, what to test

The reframing changes the walk and the baseline, splits the rule's statistics into two jobs, and admits inhibition. Everything else carries over.

| Piece | Previous guide | This guide |
| --- | --- | --- |
| The walk | built from the offspring matrix Λ | observed: spike of i within w after a spike of j |
| Requirements for the walk | Λ ≥ 0, ρ(Λ) < 1, non-negative immigration | a stationary frame |
| Chance coincidences | bias to remove | teleportation by firing rate, part of the walk |
| Inhibition | not supported | below-chance transitions; same rule, sign flipped |
| Baseline | weight- or counter-weighted | observed-walk-weighted, from pair counts |
| Step factor | pair count κ, or counter c | signed counter c |
| Homeostasis | κ or c | W\_ij n\_j / d\_j |
| Stability | ρ(Λ\_EE) ≤ χ G\_max | the full signed E–I Λ: ρ < 1 for branching, max Re λ < 1 for LIF and QIF |
| Gradient lemma | Lemma K | Lemma E (Lemma K is the case a = X = 0, excitation only) |

**Open problems.**

- **The local split drops indirect effects.** Changing W\_ij also changes X and the rates. In the inhibition-stabilised regime the indirect feedback through inhibition is large (the "paradoxical" response), so the split may approximate the full gradient poorly there.
- **Inhibition inside a module** is weakened by the map term and may be needed for stability (Section 9).
- **Chance dilutes the signal** in large asynchronous networks. The direction of learning is right, but M\* is smaller and learning is slower. A shorter window helps until it starts cutting off real offspring.
- **Flow balance is approximate.** The imbalance is treated as unrecorded teleportation; how large it is in each engine has not been measured.
- **Gain estimates for LIF and QIF**, including separate estimates for inhibitory inputs, are untested in recurrent networks.
- **Nothing has run in the repo.** All checks are standalone, on small models.

**Tests to run in the repo, in order.**

1. Frozen weights, branching E–I: exact counter vs gated counter; structure cosine ≥ 0.98; observed-walk code length falls under structural learning.
2. Flow engine with a chance floor and inhibition: does it predict the branching E–I engine's objective and updates?
3. Recurrent Hawkes-LIF with inhibition: counter vs ground-truth offspring by synapse removal; gain error separately for E and I inputs.
4. QIF at the noisy operating point, inhibition-stabilised: stability of the full Λ̂^K, structure cosine against the local-split gradient, and whether exempting same-module inhibition is needed.
5. Closed loop CartPole: flow (unchanged) against branching E–I with the new rule.

## Sources

- [Rosvall & Bergstrom, The map equation](https://arxiv.org/pdf/0906.1405): the random walk as "a proxy for real flow", and working with trajectories directly where they exist.
- [Rosvall et al. 2014, Memory in network flows (Nature Communications)](https://www.mapequation.org/assets/publications/RosvallEtAlNatureComm14.pdf): map-equation flows built from observed pathways.
- [Corral López, Buendía & Muñoz, The excitatory-inhibitory branching process](https://arxiv.org/pdf/2203.16374): firing rate as a rectified function of excitation minus inhibition.
- [Gao & Caines, Transmission Neural Networks: Inhibitory and Excitatory Connections](https://arxiv.org/pdf/2604.04246): non-negative transmission probabilities with multiplicative inhibition; stability set by the excitatory matrix.
- [Babul, Tian & Lambiotte 2024, Strong and Weak Random Walks on Signed Networks](https://arxiv.org/abs/2406.08034v1): the signed-walker alternative.
- Your notes: the previous derivation guide and the eligibility-adaptation document with its tests.
- Cited from memory, not re-checked: Vogels et al. 2011 (inhibitory plasticity), Lambiotte & Rosvall (unrecorded teleportation), Siegert / Brunel–Hakim (LIF rate and gain).
