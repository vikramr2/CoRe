# Map-STDP Derivation Details: Emergent-Walk Edition

Oct 26, 2026 · @Vikram Ramavarapu

## D0. How to read this document

This document works through, step by step, the pieces of the Derivation Guide that need algebra. Each section states its result first, then the steps, then what was checked. Notation follows the guide's Section 2.

**Conventions**

- Time runs in steps of width Δ (one step for branching; small integration steps for LIF and QIF). x\_i(t) ∈ {0, 1} marks a spike of i in step t, and r\_i = E\[x\_i(t)\] / Δ is its rate.
- W\_ij ≥ 0 is a magnitude, and sgn\_j = ±1 is fixed by the type of j. The signed kick is b\_ij = sgn\_j w\_0 T\_ij, with T\_ij = W\_ij / d\_j.
- N\_i(t, t + w\] is the number of spikes of i in the window after step t.
- "Held fixed" always means held fixed while differentiating with respect to one weight. D3 lists exactly what is held.
- "Checked" refers to the standalone tests in D12: test\_emergent.py, plus test\_eligibility.py from the previous round.

**Map of the argument**

1. D1 defines the observed walk and shows it is non-negative and measured exactly by pair counts.
2. D2 recalls the map equation's marginal exit cost.
3. D3 computes how one weight changes the walk.
4. D4 combines these into the gradient (Lemma E).
5. D5 turns the gradient into a local rule and proves descent (Theorem E).
6. D6–D9 specialise to each engine.
7. D10 covers stability with inhibition, and D11 the task term.

## D1. The observed walk

**Result.** The observed walk T^obs\_ij = E\_ij / E\_j is a valid stochastic matrix for any network, excitatory or inhibitory. Pair counts estimate it without bias. Its inflow matches the spike shares up to a small imbalance, which is treated as teleportation.

### Step 1: define the transition count

For a spike of j in step t, count the spikes of i in the window that follows, and average over all of j's spikes:

```latex
E_{ij} \;=\; \mathbb{E}\big[\,N_i(t, t+w] \;\big|\; x_j(t) = 1\,\big]
```

### Step 2: it is never negative

N\_i is a count, so N\_i ≥ 0, and so is any conditional expectation of it: E\_ij ≥ 0. This holds for any dynamics, any signs and any nonlinearity. Normalising each column gives a stochastic matrix:

```latex
T^{obs}_{ij} = \frac{E_{ij}}{E_j}, \qquad E_j = \sum_k E_{kj} = \mathbb{E}\big[\text{population spikes in the window}\;\big|\;x_j(t)=1\big]
```

### Step 3: split the count into chance, direct and indirect parts

Write E\_ij as the unconditional count plus the excess caused by conditioning on j's spike:

```latex
E_{ij} \;=\; \underbrace{r_i\,w}_{a_i} \;+\; \frac{\mathrm{Cov}\big(N_i(t,t+w],\,x_j(t)\big)}{\Pr[x_j(t)=1]}
```

The covariance has two sources:

- **Direct, D\_ij:** the effect of j's kick on i through the synapse j → i alone, holding every other input fixed.
- **Indirect, X\_ij:** everything else. That includes chains (j → k → i), common input that drives both j and i, and module-wide fluctuations.

To first order in the kick, D\_ij = sgn\_j Λ̃\_ij (D3).

### Step 4: pair counts are unbiased for the walk

The pair count adds up, over j's spikes, the spikes of i in each following window:

```latex
\kappa_{ij} = \sum_t x_j(t)\,N_i(t,t+w], \qquad \mathbb{E}[\kappa_{ij}] = \sum_t \Pr[x_j(t)=1]\;\mathbb{E}\big[N_i \mid x_j(t)=1\big] = \bar n_j\,E_{ij}
```

The last step uses stationarity within the frame. So κ\_ij / n\_j estimates E\_ij, and Σ\_{i∉m(j)} κ\_ij / Σ\_i κ\_ij estimates the exit share ē^obs\_j. Both are ratios of counts, biased only at order 1/count.

### Step 5: flow and flow balance

Use the spike shares as the flow, and the observed walk for the link flow:

```latex
\pi_j = \frac{r_j}{\sum_k r_k}, \qquad J_{ij} = \pi_j\,T^{obs}_{ij}, \qquad \sum_i J_{ij} = \pi_j \;\text{exactly}
```

Inflow Σ\_j J\_ij equals π\_i only if E\_j is the same for every j. It differs because some neurons' spikes are followed by more population activity than others'. The difference π\_i − Σ\_j J\_ij is treated as unrecorded teleportation, as the old guide treated the stimulus term. In the branching E–I test it averaged 1.5% of π\_i (at most 4.2%), with E\_j varying by 10% across neurons.

### Step 6: how inhibition is bounded

For inhibitory j, D\_ij < 0. Step 2 bounds how negative it can get:

```latex
D_{ij} \;\ge\; -\,(a_i + X_{ij})
```

In words, j can remove at most the spikes i would otherwise have fired in the window. Linear response can predict a larger removal than this. The real network does not produce one: i falls silent, its gain drops to zero, and the effect saturates. This is the sense in which the probabilities are emergent. They come from the full nonlinear dynamics, not from the weights.

### Step 7: how much of the walk is causal

The share of j's transitions that its synapses cause directly is Σ\_i |D\_ij| / E\_j. In the branching E–I test (48 neurons, one-step window) the median was 16% for excitatory and 18% for inhibitory columns. In a large network with a long window it shrinks roughly as ν\_j / (R w). Chance never changes the sign of any update (D5), but it lowers M\* and so slows learning.

## D2. The marginal exit cost M\*

**Result.** With the flow held fixed, L depends on the link flows only through the module exit rates. Each unit of exit flow from module m costs M\*\_m ≥ 0 bits. With the observed walk, the chance term keeps M\* bounded.

### Step 1: identify what depends on the link flows

```latex
L = q_{\curvearrowright}\log q_{\curvearrowright} - 2\sum_m q_m\log q_m - \sum_j \pi_j\log\pi_j + \sum_m (p_m+q_m)\log(p_m+q_m), \qquad q_m = \sum_{j\in m}\sum_{i\notin m} J_{ij}
```

With π held fixed, the third term and every p\_m are constants, so J enters only through q\_m and q↷ = Σ\_m q\_m.

### Step 2: differentiate with respect to one module's exit rate

Use d(x log x)/dx = log x + 1 on each term containing q\_m:

```latex
\frac{\partial L}{\partial q_m} = (\log q_{\curvearrowright} + 1) - 2(\log q_m + 1) + (\log(p_m+q_m) + 1) = \log\frac{q_{\curvearrowright}\,(p_m+q_m)}{q_m^2} \;\equiv\; M^*_m
```

### Step 3: it is non-negative

q↷ ≥ q\_m and p\_m + q\_m ≥ q\_m, so the argument of the log is at least 1.

### Step 4: the marginal cost of one link

q\_m counts J\_ij exactly when j ∈ m and i ∉ m, so:

```latex
g_{ij} \equiv \frac{\partial L}{\partial J_{ij}} = M^*_{m(j)}\,\mathbb{1}[i\notin m(j)]
```

### Step 5: when the chance floor bounds M\*

The chance part of j's transitions sends a\_i / E\_j of j's flow to each i. If no cross-module effect is negative (D\_ij + X\_ij ≥ 0 for i ∉ m(j), as in an excitatory-only network), exit flow is at least that chance part, a floor that does not depend on the weights:

```latex
q_m \;\ge\; q^{chance}_m = \sum_{j\in m}\pi_j\,\frac{\sum_{i\notin m} a_i}{E_j} \;>\; 0
```

In that case M\*\_m ≤ log(q↷ (p\_m + q\_m) / (q\_m^chance)²), and M\* cannot diverge the way it could in the old causal walk. With inhibition between modules the floor does not hold: lateral inhibition pushes exit flow below chance, which is exactly what the map term rewards, and q\_m can approach zero again. Keep the clip at 20.

**Checked.** The finite-difference checks of Lemma E (D12, E1) use this expression for g.

## D3. How one weight changes the walk

**Result.** Under the local split, W\_ij changes only the direct part of column j. It raises the effect on i and, through output normalisation, lowers the effect on j's other targets:

```latex
\frac{\partial E_{kj}}{\partial W_{ij}} = \frac{\mathrm{sgn}_j\,h_k}{d_j}\big(\delta_{ki} - T_{kj}\big), \qquad \frac{\partial E_j}{\partial W_{ij}} = \frac{\mathrm{sgn}_j}{d_j}\big(h_i - \bar h_j\big), \qquad h_k = \chi\,\omega_k G_k,\;\; \bar h_j = \sum_k T_{kj}h_k
```

### Step 1: the direct effect of one kick

A spike of j kicks i by b\_ij. To first order, the extra spikes of i in the window are the kick times i's windowed response:

```latex
D_{ij} = \kappa_0\,\omega_i\,G_i\,b_{ij}, \qquad b_{ij} = \mathrm{sgn}_j\,w_0\,\frac{W_{ij}}{d_j}
```

Here κ\_0 converts a kick into an input pulse: 1/2θ for branching, τ\_m for LIF, τ for QIF. G\_i is i's gain and ω\_i is the share of i's response that lands inside the window. Writing χ = κ\_0 w\_0 and h\_i = χ ω\_i G\_i gives D\_ij = sgn\_j h\_i T\_ij = sgn\_j Λ̃\_ij.

### Step 2: what is held fixed (the local split)

While differentiating with respect to W\_ij, hold fixed:

| Held fixed | Why it would change | Order of what is dropped |
| --- | --- | --- |
| π and the chance term a\_i = r\_i w | rates move when W moves | the old EM split's ∂π/∂W term |
| indirect term X | chains j → k → i and shared input also move | Λ² and higher, plus input correlations |
| gains G and window shares ω | the operating point moves | the ∂G/∂W term, which is zero for linear transfer |

### Step 3: differentiate the kick

Only column j of T depends on W\_ij, through both W\_kj and d\_j:

```latex
\frac{\partial T_{kj}}{\partial W_{ij}} = \frac{\delta_{ki}\,d_j - W_{kj}}{d_j^2} = \frac{\delta_{ki} - T_{kj}}{d_j}
```

So raising W\_ij gives i more of j's output and every other target proportionally less.

### Step 4: carry it into E

With a and X fixed, E\_kj changes only through D\_kj = sgn\_j h\_k T\_kj:

```latex
\frac{\partial E_{kj}}{\partial W_{ij}} = \mathrm{sgn}_j\,h_k\,\frac{\delta_{ki} - T_{kj}}{d_j}
```

Summing over k gives the change in j's total, which is zero when all targets have the same h:

```latex
\frac{\partial E_j}{\partial W_{ij}} = \frac{\mathrm{sgn}_j}{d_j}\Big(h_i - \sum_k T_{kj}h_k\Big) = \frac{\mathrm{sgn}_j}{d_j}\big(h_i - \bar h_j\big)
```

### Step 5: column rescaling changes nothing

Multiplying every W\_kj in column j by the same factor leaves T\_kj, and so E, unchanged. As a check, Σ\_i W\_ij ∂T\_kj/∂W\_ij = (W\_kj − T\_kj d\_j)/d\_j = 0. Every expression downstream inherits this invariance, which is why a uniform column rescaling can be dropped from any update.

## D4. The gradient (Lemma E)

**Result.**

```latex
\frac{\partial L}{\partial W_{ij}} = \frac{\pi_j\,\mathrm{sgn}_j}{E_j\,d_j}\Big[h_i\big(g_{ij} - \bar g^{\,obs}_j\big) - C_j\Big], \qquad \bar g^{\,obs}_j = \sum_k T^{obs}_{kj}\,g_{kj}, \qquad C_j = \sum_k h_k T_{kj}\big(g_{kj} - \bar g^{\,obs}_j\big)
```

### Step 1: chain rule through column j

L depends on W through J, and only column j of J depends on W\_ij (D3):

```latex
\frac{\partial L}{\partial W_{ij}} = \sum_k \frac{\partial L}{\partial J_{kj}}\,\frac{\partial J_{kj}}{\partial W_{ij}} = \sum_k g_{kj}\,\frac{\partial J_{kj}}{\partial W_{ij}}
```

### Step 2: differentiate the normalised link flow

J\_kj = π\_j E\_kj / E\_j with π\_j fixed. By the quotient rule:

```latex
\frac{\partial J_{kj}}{\partial W_{ij}} = \frac{\pi_j}{E_j}\Big(\frac{\partial E_{kj}}{\partial W_{ij}} - T^{obs}_{kj}\,\frac{\partial E_j}{\partial W_{ij}}\Big)
```

### Step 3: substitute D3

```latex
\frac{\partial L}{\partial W_{ij}} = \frac{\pi_j\,\mathrm{sgn}_j}{E_j\,d_j}\Big[\sum_k g_{kj}\,h_k(\delta_{ki} - T_{kj}) - \bar g^{\,obs}_j\,(h_i - \bar h_j)\Big]
```

### Step 4: regroup

The first sum is h\_i g\_ij − Σ\_k h\_k T\_kj g\_kj. Add and subtract ḡ^obs\_j inside it, using Σ\_k h\_k T\_kj = h̄\_j:

```latex
h_i g_{ij} - \sum_k h_k T_{kj} g_{kj} - \bar g^{\,obs}_j h_i + \bar g^{\,obs}_j \bar h_j = h_i\big(g_{ij} - \bar g^{\,obs}_j\big) - \sum_k h_k T_{kj}\big(g_{kj} - \bar g^{\,obs}_j\big)
```

This is the result, with C\_j the second sum.

### Step 5: weight-scaled form

Multiply by W\_ij and use h\_i T\_ij = Λ̃\_ij:

```latex
W_{ij}\,\frac{\partial L}{\partial W_{ij}} = \frac{\pi_j\,\mathrm{sgn}_j}{E_j}\Big[\tilde\Lambda_{ij}\big(g_{ij} - \bar g^{\,obs}_j\big) - T_{ij}\,C_j\Big]
```

### Step 6: column sums vanish

Summing Step 5 over i, the first term gives Σ\_i h\_i T\_ij (g\_ij − ḡ^obs\_j) = C\_j and the second gives C\_j Σ\_i T\_ij = C\_j. So:

```latex
\sum_i W_{ij}\,\frac{\partial L}{\partial W_{ij}} = 0
```

This is D3 Step 5 seen from the gradient: L cannot change under a column rescaling. The T\_ij C\_j term is itself proportional to W\_ij within the column, so it is exactly such a rescaling.

### Step 7: special cases

- **No chance, no indirect terms, excitation only** (a = X = 0, sgn = +1). Then E\_kj = h\_k T\_kj, E\_j = h̄\_j and T^obs\_kj = h\_k T\_kj / h̄\_j, the gain-weighted walk T^eff of the old Step 6. C\_j = h̄\_j (Σ\_k T^obs\_kj g\_kj − ḡ^obs\_j) = 0, leaving π\_j h\_i (g\_ij − ḡ^eff\_j) / (h̄\_j d\_j). With h = χ G this is Lemma K, since S\_j = Σ\_l G\_l W\_lj = d\_j h̄\_j / χ.
- **Also equal gains.** Then h is constant and the result is the original Lemma, (π\_j / d\_j)(g\_ij − ḡ\_j).

**Checked (D12, E1).** On a 16-neuron network with four inhibitory neurons, heterogeneous h, a chance term and random indirect terms, the formula matches central finite differences of L to a relative error of 7 × 10⁻⁷. The column sums of W ⊙ ∇L are zero to 8 × 10⁻¹⁶.

## D5. The local rule (Theorem E)

**Result.** If the signed counter is unbiased for the windowed direct effect, the structure rule with the observed-walk baseline is weight-scaled gradient descent on L, plus a column rescaling that cannot change L. Any other baseline is exact only when every target of j has the same h.

### Step 1: the rule

```latex
\Delta W_{ij} = -\lambda\,\big(g_{ij} - \bar g^{\,obs}_j\big)\,c_{ij} = -\lambda\,M^*_{m(j)}\big(\mathbb{1}[i\notin m(j)] - \bar e^{\,obs}_j\big)\,c_{ij}
```

### Step 2: the counter's requirement

```latex
\mathbb{E}[c_{ij}] = \bar n_j\,D_{ij} = T_f\,r_j\,\mathrm{sgn}_j\,\tilde\Lambda_{ij}
```

That is: per spike of j, c adds the change in i's spikes within the window caused by j's kick alone. D7–D9 give a counter for each engine.

### Step 3: expected update

Treat M\* and ē^obs as their expectations (both are ratios of large counts):

```latex
\mathbb{E}[\Delta W_{ij}] = -\lambda\,T_f\,r_j\,\mathrm{sgn}_j\,\tilde\Lambda_{ij}\big(g_{ij} - \bar g^{\,obs}_j\big)
```

### Step 4: match it to Lemma E

From D4 Step 5, sgn\_j Λ̃\_ij (g\_ij − ḡ^obs\_j) = (E\_j / π\_j) W\_ij ∂L/∂W\_ij + sgn\_j T\_ij C\_j. Since r\_j / π\_j = Σ\_k r\_k:

```latex
\mathbb{E}[\Delta W_{ij}] = -\lambda\zeta\,E_j\,W_{ij}\,\frac{\partial L}{\partial W_{ij}} \;-\; \lambda\,T_f\,r_j\,\mathrm{sgn}_j\,C_j\,T_{ij}, \qquad \zeta = T_f\sum_k r_k
```

The second term is a constant times W\_ij within column j, so it is a column rescaling.

### Step 5: descent

The first-order change in L is the update dotted with the gradient. The rescaling term contributes Σ\_i W\_ij ∂L/∂W\_ij = 0 (D4 Step 6), so:

```latex
\Delta L = \sum_{ij}\frac{\partial L}{\partial W_{ij}}\,\mathbb{E}[\Delta W_{ij}] = -\lambda\zeta\sum_{ij}E_j\,W_{ij}\Big(\frac{\partial L}{\partial W_{ij}}\Big)^2 \;\le\; 0
```

E\_j > 0 and W\_ij ≥ 0, so every term is non-negative. The preconditioner E\_j W\_ij is diagonal and positive, and it never needs computing.

### Step 6: why the baseline must be the observed walk's

Replace ḡ^obs\_j by any other per-neuron baseline b\_j. The update gains a term −λ T\_f r\_j sgn\_j Λ̃\_ij (ḡ^obs\_j − b\_j). Within column j that term is proportional to h\_i T\_ij:

- If h is the same for every target of j, it is proportional to W\_ij, so it is a harmless column rescaling.
- If h differs (unequal gains or window shares), it moves weight toward high-h targets and is not descent.

In the E2 test, with heterogeneous h, the counter-weighted and weight-normalised baselines reached cosine 0.978 and 0.979 with the true direction, against 1.000 for ḡ^obs. This reverses the previous round's conclusion for the causal walk, where the counter-weighted baseline was the exact one. The baseline must come from the walk being described.

### Step 7: sign pattern

The factor (1\[i ∉ m(j)\] − ē^obs\_j) is positive across modules and negative within, so the sign of each update depends only on module membership and sgn\_j:

| Synapse | Across modules | Within module |
| --- | --- | --- |
| Excitatory | weakened | strengthened |
| Inhibitory | strengthened | weakened |

Chance changes the size of these updates but never their sign. In E2 the pattern held for every synapse.

### Step 8: Dale's law

c\_ij is proportional to W\_ij in every engine, so every update is multiplicative. For small enough steps a magnitude never reaches zero, and a synapse never changes sign.

### Step 9: estimating the baseline locally

ē^obs\_j is the share of spikes, in windows after j's spikes, that come from outside m(j). Neuron j can keep two running sums: population spikes in the window after each of its spikes, and those from outside its module. Both need only j's own spike times and a broadcast of module activity (K numbers per step), not per-synapse coincidences.

## D6. Flow engine

**Result.** For an excitatory-only network the flow engine's rule is exactly Theorem E, so nothing changes and its results stand. Adding inhibition needs two changes: a rectified rate fixed point and a chance floor.

### Step 1: what the flow engine computes

The flow engine has no spikes. It computes π by matrix–vector hops of the walk π = (1 − α) T π + α v(o), and replaces each pair count by its expectation J\_ij = π\_j T\_ij.

### Step 2: show it is a case of the observed walk

With no spikes there are no chance coincidences (a = 0) and no indirect correlations (X = 0). With gain 1 and no window loss, h\_k = χ for every k:

```latex
E_{kj} = \chi\,T_{kj}, \qquad E_j = \chi, \qquad T^{obs} = T, \qquad \bar g^{\,obs}_j = M^*_{m(j)}\,\bar e_j
```

### Step 3: show the rule is unchanged

The counter's expectation is r\_j Λ̃\_ij ∝ π\_j T\_ij = J\_ij, which is what the flow engine already uses in place of κ. The baseline M\* ē\_j is the one it already uses. So the implemented rule −λ(g\_ij − ḡ\_j) J\_ij is Theorem E term for term.

### Step 4: adding inhibition

Two changes are needed:

1. **Rates from a rectified fixed point.** With inhibitory columns, π is no longer the PageRank vector of a non-negative walk. Iterate r = F(B r + I\_ext), with B\_ij = sgn\_j χ T\_ij and F rectified, then set π = r / Σ r.
2. **A chance floor.** Set the expected walk to:

```latex
E_{ij} = \max\big(0,\; w\,r_i + \mathrm{sgn}_j\,\chi\,G_i\,T_{ij}\big)
```

with w matching the spiking engine's window. The max mirrors silencing (D1 Step 6): j cannot remove more of i's spikes than chance would have produced. Where it binds, cap the counter's expectation at the same place, T\_f r\_j × (−w r\_i) instead of T\_f r\_j sgn\_j χ G\_i T\_ij, and the derivative there is zero.

### Step 5: recommendation

- Keep the flow engine unchanged for excitatory-only work.
- Use the Step 4 version when inhibition is added, and pick w to match the spiking engine you are developing for, so that the flow engine predicts that engine's objective and updates.
- Indirect terms X can be added from linear response, X ≈ \[(I − Λ)⁻¹ − I − Λ\] terms over the window, but they are held fixed in the gradient anyway and only shift the baseline.

The Step 4 version is untested.

## D7. Branching engine with inhibition

**Result.** In the branching engine the walk's natural step is one time step, and the counter is exact rather than first-order: the neuron's own firing probability with and without j's kick. Rectification at zero is what keeps inhibition inside the walk.

### Step 1: model

```latex
u_i(t+1) = \sum_j b_{ij}\,x_j(t) + 2\theta\,\mu_i(t), \qquad b_{ij} = \mathrm{sgn}_j\,w_{0,j}\,\frac{W_{ij}}{d_j}, \qquad p_i = \mathrm{clip}\Big(\frac{u_i}{2\theta},\,0,\,1\Big), \qquad x_i(t+1)\sim\mathrm{Bernoulli}(p_i)
```

w\_0,j is the output scale for j's type. The membrane forgets every step, so one step is one hop.

### Step 2: the walk

With a one-step window, E\_ij = E\[p\_i(t+1) | x\_j(t) = 1\] and the pair count is κ\_ij = Σ\_t x\_j(t) x\_i(t+1). Both are always non-negative because p\_i ≥ 0.

### Step 3: the exact direct effect

Given the state at t, removing j's kick changes i's firing probability at t+1 by a known amount. That is j's direct effect on i in that step, exactly, with no linearisation:

```latex
\Delta_{ij}(t) = p\big(u_i(t+1)\big) - p\big(u_i(t+1) - b_{ij}\big)
```

- For excitatory j it lies between 0 and b\_ij / 2θ.
- For inhibitory j it lies between b\_ij / 2θ and 0, and it is exactly −p(u\_i − b\_ij) when the kick pushes u\_i below zero: j removes only the spike probability i actually had.

This is D1 Step 6 in its exact form.

### Step 4: the counter

```latex
c_{ij} = \sum_t x_j(t)\,\Delta_{ij}(t), \qquad \mathbb{E}[c_{ij}] = \bar n_j\,D_{ij}
```

The equality holds by construction: c averages, over j's actual spikes, the exact conditional effect. It needs neither independence nor first-order approximation. The neuron already computes u\_i and p(u\_i). Evaluating p(u\_i − b\_ij) costs one subtraction and one clip per presynaptic spike.

### Step 5: the gated approximation and its error

The cheaper gate Δ ≈ (b\_ij / 2θ) 1\[0 < u\_i < 2θ\] is exact inside the linear range. At the lower edge it errs: an inhibitory kick that pushes u\_i below zero gets gate 0, so the counter misses the spikes it did remove. In the E–I test this made the gated counter understate inhibition by 16%, although its overall correlation with the exact counter was 0.9994 and the structure-update cosine 0.9995. Use the exact version.

### Step 6: gain, offspring and stability

The linear gain is G\_i = Pr\[0 < u\_i < 2θ\]. The signed offspring matrix and stability condition are:

```latex
\Lambda_{ij} = \mathrm{sgn}_j\,\frac{w_{0,j}}{2\theta}\,G_i\,T_{ij}, \qquad \rho(\Lambda) = \max_k|\lambda_k(\Lambda)| < 1 \quad (\text{one-step map: every eigenvalue inside the unit circle})
```

With w\_0,E / 2θ = 0.8 the excitatory block is subcritical by itself (ρ(Λ\_EE) ≤ 0.8), so inhibition is not needed for stability here.

### Step 7: test results (D12, E4)

The test used 48 neurons in 4 modules, each module with 10 excitatory and 2 inhibitory neurons, w\_0 / 2θ = 0.8 for both types, module stimuli redrawn every 50 steps.

| Check | Result |
| --- | --- |
| Observed transitions below zero | none |
| Gated vs exact counter | correlation 0.9994; inhibition understated by 16% |
| Structure update: exact counter vs unsigned pair count as the factor | cosine 0.747 |
| Inflow imbalance (D1 Step 5) | 1.5% mean, 4.2% max |
| Causal share of transitions (D1 Step 7) | 16% excitatory columns, 18% inhibitory |

A 40-block structural learning run (5,000 steps per block, normalised step size, column renormalisation in place of homeostasis):

| Factor in the rule | Code length L (bits), start → end | E weight leaving module | I weight to other modules | Mean rate per step |
| --- | --- | --- | --- | --- |
| exact signed counter c | 4.611 → 4.096 | 62% → 11% | 76% → 98% | 0.064 → 0.069 |
| pair count κ, given the presynaptic sign | 4.611 → 4.556 | 62% → 53% | 76% → 83% | 0.064 → 0.067 |

The pair count moves the weights in roughly the right direction but slowly: most of each column's pair count is chance, which gives a nearly uniform update that renormalisation cancels.

## D8. LIF engine

**Result.** One kick of either sign changes a noisy LIF neuron's total spike count by τ\_m G\_i b\_ij to first order. The counter therefore needs only presynaptic spike counts and a gain estimate. Choose the window so that most of that change lands inside it.

### Step 1: model

```latex
\tau_m\,\frac{dV_i}{dt} = -(V_i - E_L) + \mu_i(t) + \sigma_V\sqrt{2\tau_m}\,\xi_i(t) + \tau_m\sum_j b_{ij}\sum_f\delta\big(t-t^f_j\big), \qquad V_i \ge V_{th} \Rightarrow \text{spike},\; V_i \leftarrow V_r
```

b\_ij is signed. The noise term stands for background input. TD-STDP's neuron is the case σ\_V = 0.

### Step 2: one kick is a pulse of input

A kick b at t\_0 adds b e^{−(t−t\_0)/τ\_m} to V. That is the same as a pulse in the mean input of area τ\_m b. In linear response, the extra spike count is the static gain times the pulse area:

```latex
\int \delta r_i\,dt = G_i\,\tau_m\,b_{ij}, \qquad G_i = \frac{\partial r_i}{\partial \mu_i}
```

This holds for either sign of b. The response is spread over time, but its area is fixed.

### Step 3: the window share ω\_i

The walk counts only the part of that area that falls inside the window, ω\_i = ∫\_0^w δr\_i / ∫ δr\_i. The response time of a neuron with escape noise is about τ\_m, and a recurrent network's slowest mode decays at (1 − ρ) / τ\_m. With w at about three response times, ω\_i ≈ 0.95 for every neuron, and differences in ω across targets stop mattering (D5 Step 6).

### Step 4: the counter

```latex
c_{ij} = \mathrm{sgn}_j\,\tau_m\,\hat\omega_i\,\hat G_i\,|b_{ij}|\,n_j \;\approx\; \tau_m\,\hat G_i\,b_{ij}\,n_j \quad (\omega \approx 1)
```

It is presynaptic count × kick × postsynaptic gain, an outer product with no per-synapse accumulation during the frame. For an escape-noise neuron (Hawkes-LIF), the time-resolved version is exact to first order:

```latex
c_{ij} = b_{ij}\sum_f\int_{t^f_j}^{t^f_j + w} e^{-(t - t^f_j)/\tau_m}\,\psi_i(t)\,dt, \qquad \psi_i = \frac{\partial \rho_i}{\partial V_i}
```

### Step 5: the gain estimate

- **From the neuron's own input statistics** (Siegert, cited from memory; the gain formula was checked numerically). σ\_B is √2 times the stationary standard deviation of the free membrane potential.

```latex
r = \Big[\tau_{ref} + \tau_m\sqrt\pi\int_{y_r}^{y_\theta} e^{x^2}(1+\mathrm{erf}\,x)\,dx\Big]^{-1}, \qquad \hat G = \frac{r^2\,\tau_m\sqrt\pi}{\sigma_B}\big[g(y_\theta) - g(y_r)\big], \qquad g(x) = e^{x^2}(1+\mathrm{erf}\,x)
```

- **By regression across frames** of i's rate on its measured input. This is untested, and biased upward by recurrent feedback.

### Step 6: the second-order asymmetry between E and I

To second order the effect of a kick is G τ\_m b + ½ F''(τ\_m b)². Below threshold the transfer is convex (F'' > 0), so excitatory kicks do slightly more than G τ\_m b predicts and inhibitory kicks slightly less. Measured on a single noisy neuron (τ\_m = 20 ms, stationary voltage s.d. 0.25θ, 60,000 paired trials):

| Mean drive μ / θ | Rate | Kick b / θ | Simulated / predicted |
| --- | --- | --- | --- |
| 0.6 | 7 Hz | −0.05 | 0.88 |
| 0.8 | 14 Hz | −0.05 | 0.88 |
| 1.2 | 33 Hz | −0.05 | 0.98 |
| 0.8 | 14 Hz | −0.15 | 0.85 |
| 0.8 | 14 Hz | +0.05 | 1.05 |

The prediction uses the Siegert gain, which also runs about 3% high in this simulation (seen in the previous round's tests). A practical fix is to estimate separate gains for excitatory and inhibitory inputs, Ĝ⁺\_i and Ĝ⁻\_i, as the slopes of the transfer on each side. Untested.

### Step 7: noise, and the noiseless LIF

The walk is defined for any LIF, including TD-STDP's noiseless one: pair counts always exist. The counter needs a smooth response, because a noiseless hard threshold has gain 0 or infinite. So add input or escape noise, or use a surrogate gain calibrated against measured responses. The previous round's tests (excitatory, feed-forward) found the counter with the true gain within 1.5–5% of ground-truth offspring on average by input type (8% low at the lowest-rate neuron), and the Siegert gain within 1.5% of the true gain at three of four operating points.

## D9. QIF engine

**Result.** QIF uses the same mean counter as LIF, at a noisy operating point. The reframing removes QIF's old disqualifier: an excitatory block with ρ > 1, held stable by inhibition, is allowed as long as the full E–I linearisation is stable.

### Step 1: model

```latex
\tau\,\frac{dV_i}{dt} = V_i^2 + \eta_i + I^{ext}_i(t) + \sigma\sqrt{\tau}\,\xi_i(t) + \tau\sum_j b_{ij}\sum_f \delta\big(t - t^f_j\big), \qquad V_i \ge V_{peak} \Rightarrow \text{spike},\; V_i \leftarrow -V_{peak}
```

This is the Montbrió–Pazó–Roxin normalisation (cited from memory). A kick b adds exactly b to V.

### Step 2: why noise bounds the gain

Without noise and with constant drive I > 0, the rate is r = √I / (πτ) and the gain is G = 1 / (2π²τ² r), which diverges near rheobase. With noise, rescaling V = σ^{2/3} v and t = (τ / σ^{2/3}) s removes σ from the equation:

```latex
r = \frac{\sigma^{2/3}}{\tau}\,g\Big(\frac{I}{\sigma^{4/3}}\Big) \quad\Longrightarrow\quad G_{max} \propto \frac{1}{\tau\,\sigma^{2/3}}
```

This scaling is my derivation and is not tested directly. The gains measured in the previous round (17–19 at τ = 10 ms, σ = 0.5–1) are consistent with it.

### Step 3: one kick, mean effect

The area argument of D8 Step 2 carries over with τ in place of τ\_m: one kick changes i's spike count by τ G\_i b\_ij on average. In the previous round's tests this held within 2% (tonic regime) and 7% (fluctuation-driven regime, kick 0.2), for excitatory kicks.

### Step 4: the counter

```latex
c_{ij} = \tau\,\hat G_i\,b_{ij}\,n_j
```

The gain comes from regression across frames, because no closed form is used for the noisy QIF. The time-resolved phase-response gate r\_i / V̇\_i has the right shape only for tonic firing. At the noisy operating point its shape correlation with ground truth was 0.70, so do not use it there.

### Step 5: inhibition-stabilised operating points

The old walk needed ρ(Λ\_EE) < 1, and QIF's measured 1.1–1.7 disqualified it. The observed walk exists regardless (D1). The counters need a stable linearisation of the full system (D10), which inhibition can provide. What still needs checking:

- Stability of the full signed Λ̂ (excitatory and inhibitory parts kept separate, D10 Step 6).
- Response time: the slowest stable mode decays at rate (1 − Re λ\_max) / τ, and it must fit within a frame.
- How well the local split approximates the full gradient. Inhibition-stabilised networks respond to input through strong indirect feedback, which the local split holds fixed (D3 Step 2). Expect a lower cosine than for branching.

### Step 6: what noise is still for

- Bounding and smoothing the gain (Step 2), so that Ĝ exists.
- Breaking phase locking, so that j's arrivals are uncorrelated with i's phase. The counter's expectation assumes this.

None of D9 has been tested with inhibition.

## D10. Stability with inhibition

**Result.** The counters need a stable operating point of the full E–I system, and the condition depends on the engine's time structure. Branching, a one-step map, needs every eigenvalue of the signed Λ inside the unit circle. LIF and QIF, continuous in time, need every eigenvalue's real part below 1. In both, an excitatory block with ρ > 1 can be stable if inhibition holds it.

### Step 1: linearise

Around a stationary operating point, small rate deviations obey:

```latex
\text{branching (one-step map):}\;\; \delta r(t+1) = \Lambda\,\delta r(t), \qquad \text{LIF, QIF (rate dynamics):}\;\; \tau\,\frac{d\,\delta r}{dt} = (\Lambda - I)\,\delta r
```

### Step 2: the two conditions

```latex
\text{map:}\;\; \max_k |\lambda_k(\Lambda)| < 1, \qquad \text{continuous:}\;\; \max_k \mathrm{Re}\,\lambda_k(\Lambda) < 1
```

The map condition is the stricter one. A strongly negative eigenvalue, which strong inhibition can create, is harmless in continuous time but makes a one-step map oscillate and diverge.

### Step 3: excitation only

For Λ ≥ 0, Perron–Frobenius makes the largest eigenvalue real, λ\_max = ρ(Λ), so both conditions reduce to ρ < 1. With output normalisation each column sum is at most χ G\_max, and the spectral radius is at most the largest column sum:

```latex
\rho(\Lambda_{EE}) \;\le\; \max_j \sum_i \Lambda_{ij} \;\le\; \chi\,G_{max}
```

This is the old bound, valid for every W. It still guarantees stability with no help from inhibition.

### Step 4: a sufficient condition with inhibition

Any matrix norm bounds every eigenvalue, so χ\_E G\_max + χ\_I G\_max < 1 for column sums of |Λ| guarantees both conditions. It is conservative: it excludes the inhibition-stabilised regime.

### Step 5: the inhibition-stabilised regime, two populations

Take one excitatory and one inhibitory population, with all λ ≥ 0:

```latex
\Lambda = \begin{pmatrix}\lambda_{EE} & -\lambda_{EI}\\ \lambda_{IE} & -\lambda_{II}\end{pmatrix}
```

In continuous time, A = Λ − I is stable when its trace is negative and its determinant positive:

```latex
\lambda_{EE} < 2 + \lambda_{II}, \qquad \lambda_{EI}\,\lambda_{IE} > (\lambda_{EE} - 1)(1 + \lambda_{II})
```

With λ\_EE > 1 the excitatory loop alone would run away, and the second inequality says how strong the E → I → E loop must be to stop it.

### Step 6: the map term's pressure on stability

Inside a module, the structure term raises λ\_EE (strengthens within-module excitation) and lowers λ\_EI (weakens within-module inhibition). Both push the determinant in Step 5 toward zero. With an excitation-only bound in force (χ\_E G\_max < 1) this cannot cause instability. In the inhibition-stabilised regime it can. Remedies, untested:

- Exempt inhibitory → same-module synapses from the structure term (g = 0).
- Let an inhibitory homeostatic rule set those synapses (Vogels et al. 2011, cited from memory).
- Add a stability penalty when the monitored eigenvalue (Step 7) nears its limit.

### Step 7: monitoring

The counters give a module-level offspring matrix. For stability, keep excitatory and inhibitory parts apart: a 2K × 2K matrix with entries (Σ c over source and target populations) / (source spike count). Check its spectral radius (branching) or largest real part (LIF, QIF) each frame. It costs 4K² sums and a small eigenvalue computation, done centrally.

## D11. The task term

**Result.** The policy depends only on firing rates, so the walk's reframing does not touch the task score. With signed offspring, the exact eligibility is the total (unwindowed) signed counter times the cascade value, up to a column rescaling. The module-level cascade is computed from the same counters.

### Step 1: the score as a derivative of rates

The readout is P(a) = R\_a / Σ\_{a'} R\_{a'}, with R\_a the summed rate of action column a (excitatory neurons). Define s\_i = ∂ log P(a\*) / ∂r\_i:

```latex
s_i = \frac{\mathbb{1}[m(i)=a^*] - P(m(i))}{R_{m(i)}} \;\;\text{for excitatory } i \text{ in an action column}, \qquad s_i = 0 \;\text{otherwise}
```

### Step 2: linear response with signed couplings

At the fixed point r = F(Br + I\_ext), with B\_kj = sgn\_j χ\_j T\_kj, a change δB gives:

```latex
\delta r = (I - \Lambda)^{-1}\,\mathrm{diag}(G)\,\delta B\,r, \qquad \Lambda = \mathrm{diag}(G)\,B \;\;(\text{signed})
```

I − Λ is invertible whenever the operating point is stable (D10).

### Step 3: differentiate one coupling

As in D3 Step 3, ∂B\_kj / ∂W\_ij = sgn\_j χ\_j (δ\_ki − T\_kj) / d\_j, and only column j moves.

### Step 4: the cascade value

Define Φ = (I − Λ)^{−⊤} s, the score summed over all of a spike's linear-response descendants, through inhibitory paths too. Then s⊤ δr = Φ⊤ diag(G) δB r, and:

```latex
\frac{\partial \log P}{\partial W_{ij}} = \frac{\chi_j\,\mathrm{sgn}_j\,r_j}{d_j}\Big(G_i\Phi_i - \sum_k T_{kj}\,G_k\Phi_k\Big)
```

### Step 5: weight-scaled form

Multiply by W\_ij and use Λ\_ij = sgn\_j χ\_j G\_i T\_ij:

```latex
W_{ij}\,\frac{\partial\log P}{\partial W_{ij}} = \Lambda_{ij}\,r_j\,\Phi_i \;-\; W_{ij}\,q_j, \qquad q_j = \frac{\chi_j\,\mathrm{sgn}_j\,r_j}{d_j}\sum_k T_{kj}\,G_k\Phi_k
```

The second term is a column rescaling, so the exact eligibility needs no presynaptic baseline.

### Step 6: the eligibility

The total signed counter has E\[c^tot\_ij\] = T\_f Λ\_ij r\_j, so:

```latex
e_{ij} \leftarrow \Big(1 - \tfrac{1}{\tau_e}\Big)e_{ij} + c^{tot}_{ij}\,\Phi_i, \qquad \mathbb{E}\big[c^{tot}_{ij}\Phi_i\big] = T_f\,\Lambda_{ij}\,r_j\,\Phi_i
```

c^tot counts direct offspring over all time. With a window of about three response times it differs from the windowed c by about 5%, so one counter can serve both terms (D8 Step 3).

### Step 7: the module-level cascade

Assume Φ\_i ≈ Φ^K\_{m(i)} and average the per-neuron equation Φ = s + Λ⊤Φ over module a with flow weights r\_j / R\_a:

```latex
\Phi^K_a = s^K_a + \sum_b \Lambda^K_{ba}\,\Phi^K_b, \qquad \Lambda^K_{ba} = \sum_{j\in a}\frac{r_j}{R_a}\sum_{i\in b}\Lambda_{ij} \;\approx\; \frac{\sum_{i\in b}\sum_{j\in a} c^{tot}_{ij}}{N_a}
```

The last form estimates Λ^K from the counters with N\_a the module's spike count. With inhibition Λ^K has negative entries, so solve the K × K system directly rather than iterating.

**Checked (D12, E3).** Eight networks of 24 neurons in 4 modules, one in six inhibitory, heterogeneous gains, largest real eigenvalue 0.11–0.41. The test's readout summed all neurons of an action module, inhibitory ones included:

- the score formula matched finite differences to ≤ 1.9 × 10⁻⁸ relative error;
- the exact eligibility reached cosine 1.000000 with W ⊙ ∇ log P, after removing the column rescaling;
- Φ^K reached 0.959–0.994;
- one-hop credit reached 0.959–0.992.

Cascade credit adds little here because these networks are far from critical.

## D12. Checks and tests

**Result.** The algebra of D3–D5 and D11 is confirmed to numerical precision, including with inhibition. In a branching E–I network the exact signed counter produces the predicted sign pattern and lowers the observed walk's code length. Inhibition in LIF and QIF networks is untested beyond single-neuron kicks.

All tests are standalone (test\_emergent.py, 6 pytest tests in about 6 seconds; the full report runs in about 30 seconds). They use small illustrative models, not the repo.

### E1: Lemma E against finite differences

16 neurons in 4 modules (3 excitatory and 1 inhibitory each), lognormal h, chance term a\_i ∈ \[0.3, 0.6\], random fixed indirect terms, π fixed.

| Check | Result |
| --- | --- |
| Lemma E vs central differences of L | relative error 6.7 × 10⁻⁷ |
| Column sums of W ⊙ ∇L | 7.7 × 10⁻¹⁶ |

### E2: Theorem E and baselines

Same network, expected updates (E\[c\] in place of c).

| Baseline in the rule | Cosine with the gradient direction | First-order change in L |
| --- | --- | --- |
| observed walk, ē^obs | 1.000000 | −1.993 × 10⁻⁴, equal to the predicted −Σ E\_j W (∂L)² |
| counter-weighted | 0.978 | −1.942 × 10⁻⁴ |
| weight-normalised | 0.979 | −1.984 × 10⁻⁴ |

The sign pattern of D5 Step 7 held for 100% of synapses in each of the four classes.

### E3: task score with inhibition

See D11. The score formula matched to ≤ 1.9 × 10⁻⁸; the exact eligibility reached cosine 1.000000; Φ^K reached 0.959–0.994.

### E4: branching engine with inhibition

See D7 Step 7: no negative transitions; gated vs exact counter correlation 0.9994, with inhibition understated by 16%; learning lowered L from 4.611 to 4.096 bits with the exact counter, against 4.556 with pair counts.

### E5: single LIF neuron, signed kicks

See D8 Step 6: simulated / predicted 0.85–0.98 for inhibitory kicks and 1.05 for an excitatory one.

### Carried over from the previous round (test\_eligibility.py)

- **LIF feed-forward, excitatory:** the counter with the true gain correlated 0.992 with ground-truth offspring; pair counts correlated 0.34.
- **QIF, excitatory kicks:** the area argument held within 2% (tonic) and 7% (fluctuation-driven); the time-resolved gate had the wrong shape in the fluctuation-driven regime.
- **Lemma K and the score identity:** exact. These are the special cases of D4 Step 7.

### Not tested

- Recurrent LIF or QIF with inhibition.
- The flow engine with a chance floor (D6 Step 4).
- The inhibition-stabilised regime, and whether exempting same-module inhibition is needed (D10 Step 6).
- How well the local split approximates the full gradient (D3 Step 2), in any engine.
- Separate gains for E and I inputs (D8 Step 6).
- Anything in closed loop.
