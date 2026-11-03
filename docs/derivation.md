# Map-STDP: communities as states

Author: Vikram Ramavarapu

**Goal.** Give a spiking network a **state-space abstraction** whose states are its own communities. A random walk on a graph with communities has a minimum description length, the map equation [Rosvall & Bergstrom 2008]. Community detection solves the *forward* problem: given the graph, find the partition that minimises it. Map-STDP solves the **inverse** problem. The partition is given (one controller community, one community per action, some latent communities), and plasticity shapes the synapses until that partition is the minimum-description-length one. The coarse-grained walk between communities is then a stimulus-conditioned Markov model over the network's states. It can be read out, and in an RL environment it chooses the action.

**The walk is the network's own spike sequence.** A step of the walk from $j$ to $i$ is a spike of $i$ within a short window after a spike of $j$ (§2.1). Its transition probabilities are measured from spikes rather than built from the weights, so they are non-negative whatever the signs of the synapses, and inhibition fits inside the walk as transitions below chance. The map equation itself calls the random walk "a proxy for real flow" and has been run directly on observed trajectories [Rosvall & Bergstrom 2008; Rosvall et al. 2014]; here the observed trajectories are spikes. The flow engine (numpy power iteration, no spikes) is the special case without chance or indirect terms, in which the walk is the weight-normalised walk $T = W / d$ re-injected by the stimulus (1).

**Each environment frame is one walk.** The stimulus of frame $f$ re-injects activity, and the walk is run to its per-frame flow $\pi_f$ within the frame (§2.3). Description length, modulators and the extracted Markov model are all per frame.

**Side hypothesis (secondary).** Description length tracks the metabolic cost of communication, which is hypothesised to be higher when activity crosses into modules a neuron rarely talks to. The state abstraction does not depend on it.

## At a glance

Each synapse $j \to i$ keeps a magnitude $W_{ij} \ge 0$; the type of $j$ fixes its sign $\mathrm{sgn}_j = \pm 1$ (Dale's law). Per frame,

$$
\Delta W_{ij} = \underbrace{\eta \delta_f e_{ij}}_{\text{task}} - \underbrace{\lambda M^{\ast}_{m(j)} \left( \mathbb{I}[i \notin m(j)] - \bar{e}^{obs}_j \right) c_{ij}}_{\text{structure}} - \underbrace{\epsilon_h \left( d_j - 1 \right) \frac{\pi_j W_{ij}}{d_j}}_{\text{homeostasis}}
$$

- $\delta_f$ is a global TD error from a critic (4c), and $e_{ij}$ an action-gated eligibility trace built from the signed counter (4b).
- $c_{ij}$ is the **signed counter**: the spikes of $i$ that $j$'s kicks add (or remove), $\mathbb{E}[c_{ij}] = \bar{n}_j D_{ij}$. It is presynaptic arrival times postsynaptic sensitivity, closer to voltage-based plasticity [Artola et al. 1990; Clopath et al. 2010] than to pair-timing STDP. It carries $\mathrm{sgn}_j$.
- $\bar{e}^{obs}_j$ is the share of $j$'s observed transitions that leave its module, measured from **pair counts** $\kappa_{ij}$ (8).
- $M^{\ast}_m$ is the marginal bits per unit of exit flow from module $m$ (§3), a $K$-number broadcast.
- $d_j$ is $j$'s total output weight; the third term holds it near 1 (4d).

The two statistics have separate jobs: pair counts measure the walk and so give the baseline; the signed counter measures each synapse's direct effect and so gives the step (§4.1). On average the structural term is weight-scaled gradient descent on the description length, preconditioned by each neuron's observed transition count $E_j$, plus a column rescaling that cannot change it (Theorem E, (10)). It weakens cross-module excitation and within-module inhibition, and strengthens within-module excitation and cross-module inhibition (§4.4). Controller→action and controller→latent routing is **exempt** from it (§4.5). In each frame the module-to-module flow is a Markov chain $T^K(\mathbf{o})$ over communities (1c); its flow into the action modules is the policy (1d).

## 1. Motivation

- **Communities as states.** A spiking network's microstate is far too large to be a state space. If the network is modular, the coarse-grained walk between modules is a small Markov chain that can be read out. Modules mapped onto actions and latent situations make it an abstract model of the task.
- **A central controller.** The stimulus enters one controller community, which routes activity to latent and action communities; this is what makes the extracted chain depend on the stimulus (§2.4). It is a functional abstraction, closer to thalamus or sensory cortex than to any single cortical router. The action readout resembles channel selection in the basal ganglia [Redgrave et al. 1999], latent communities resemble task-state codes in orbitofrontal cortex [Wilson et al. 2014] and latent-structure learning [Gershman & Niv 2010].
- **Neuromodulators gate costly plasticity.** Memory formation has a measurable metabolic cost [Plaçais & Preat 2013; Li & van Rossum 2020], and three-factor rules, where a local eligibility flag becomes a weight change only when a neuromodulator arrives, have direct experimental support [Gerstner et al. 2018]. Here the structural neuromodulator carries a description-length signal, so plasticity is spent where it makes the network's activity cheaper to describe.

## 2. Setup: activity as flow

### Notation

| Symbol | Meaning | Lives at |
| --- | --- | --- |
| $W_{ij} \ge 0$, $\mathrm{sgn}_j = \pm 1$ | magnitude of the synapse from pre $j$ to post $i$; type of $j$ (+1 excitatory, −1 inhibitory) | synapse; neuron $j$ |
| $d_j = \sum_k W_{kj}$, $T_{ij} = W_{ij} / d_j$ | total output weight of $j$; $j$'s output share to $i$ | neuron $j$; synapse |
| $b_{ij} = \mathrm{sgn}_j w_0 T_{ij}$ | signed kick one spike of $j$ gives $i$ (output-normalised) | synapse |
| $r_i$, $n_i$, $G_i$ | firing rate; spike count in the frame; gain (slope of the rate transfer) | neuron $i$ |
| $w$, $\omega_i$ | the walk's window; share of $i$'s response to a kick that lands inside it | global; neuron $i$ |
| $h_i = \chi \omega_i G_i$, $\tilde{\Lambda}_{ij} = h_i T_{ij}$ | windowed sensitivity of $i$; size of $j$'s windowed direct effect on $i$ | neuron $i$; synapse |
| $E_{ij} = a_i + D_{ij} + X_{ij}$ | expected spikes of $i$ in the window after a spike of $j$: chance, direct, indirect | synapse |
| $E_j = \sum_i E_{ij}$, $T^{obs}_{ij} = E_{ij} / E_j$ | $j$'s observed transition count; observed transition probability | neuron $j$; synapse |
| $\pi_j$, $\pi_{f,j}$ | flow through $j$ (spike share $r_j / \sum_k r_k$); in frame $f$ | neuron $j$ |
| $J_{ij} = \pi_j T^{obs}_{ij}$ | link flow of the observed walk | synapse |
| $\kappa_{ij}$, $c_{ij}$, $c^{tot}_{ij}$ | pair count; signed counter (windowed); signed counter (all time) | synapse |
| $\Lambda_{ij}$ | signed offspring: total extra spikes of $i$ per spike of $j$ | synapse |
| $v(\mathbf{o})$, $\alpha$ | normalised stimulus drive; share of activity it injects per step (flow engine) | input; global |
| $m(j)$, $K$ | module of $j$; number of modules | neuron $j$ |
| $\mathcal{C}$, $\mathcal{A}$, $\mathcal{L}$ | controller; action modules (one per action); latent modules | module labels |
| $\bar{e}^{obs}_j = \sum_{i \notin m(j)} T^{obs}_{ij}$; $\bar{e}_j = \sum_{i \notin m(j)} T_{ij}$ | share of $j$'s observed transitions, or of its output weight, that leaves $m(j)$ | neuron $j$ |
| $p_m = \sum_{j \in m} \pi_j$, $q_m = \sum_{j \in m} \pi_j \bar{e}^{obs}_j$, $q_{\curvearrowright} = \sum_m q_m$ | visit rate, exit rate, total switching rate | module; network |
| $J_{ba} = \sum_{i \in b, j \in a} J_{ij}$, $T^K_{ba} = J_{ba} / p_a$ | module-pair flow and transition probability ($K \times K$ tables indexed [post, pre]) | module pair |
| $f$, $t$, $n$ | environment frame; simulation step; walk hops per frame (flow engine) | global |

$\chi$ converts output weight into kick size times input-pulse area: $w_0 / 2 \theta$ for branching, $w_0 \tau_m$ for LIF, $w_0 \tau$ for QIF, and $1 - \alpha$ in the flow engine.

### 2.1 The observed walk

For a spike of $j$ at step $t$, let $N_i(t, t + w]$ be the number of spikes of $i$ in the window that follows. The walk's transition count from $j$ to $i$ is its conditional mean, $E_{ij} = \mathbb{E}[N_i(t, t + w] \mid x_j(t) = 1]$. Splitting off the unconditional count,

$$
E_{ij} = a_i + D_{ij} + X_{ij} \ge 0, \qquad a_i = r_i w, \qquad D_{ij} = \mathrm{sgn}_j \tilde{\Lambda}_{ij}, \qquad T^{obs}_{ij} = \frac{E_{ij}}{E_j}, \qquad J_{ij} = \pi_j T^{obs}_{ij} \qquad \text{(8)}
$$

- **Chance** $a_i$ is the count $i$ would fire anyway. It does not depend on $j$, so it sends $j$'s walker to $i$ in proportion to $i$'s rate: teleportation by firing rate.
- **Direct** $D_{ij}$ is the effect of $j$'s kick through the synapse $j \to i$ alone, to first order $\mathrm{sgn}_j h_i T_{ij}$ (§4.2).
- **Indirect** $X_{ij}$ is everything else: chains $j \to k \to i$, common input and module-wide fluctuations [Pernice et al. 2011; Trousdale et al. 2012].

$E_{ij}$ is the mean of a count, so it is non-negative for any dynamics, signs or nonlinearity, and $T^{obs}$ is column-stochastic for any network. **Inhibition lowers transitions** without making any negative: for inhibitory $j$, $D_{ij} < 0$, and $D_{ij} \ge -(a_i + X_{ij})$ because $j$ can remove at most the spikes $i$ would otherwise have fired. Linear response can predict a larger removal; the network does not produce one, because a silenced neuron's gain drops to zero. The walk needs only a stationary frame; stability matters for the counters, not for the walk to exist (§7).

**Pair counts measure the walk without bias.** With $\kappa_{ij} = \sum_t x_j(t) N_i(t, t + w]$ and stationarity within the frame,

$$
\mathbb{E} \left[ \kappa_{ij} \right] = \bar{n}_j E_{ij}, \qquad \hat{T}^{obs}_{ij} = \frac{\kappa_{ij}}{\sum_k \kappa_{kj}}, \qquad \hat{\bar{e}}^{obs}_j = \frac{\sum_{i \notin m(j)} \kappa_{ij}}{\sum_i \kappa_{ij}}
$$

Both estimates are ratios of counts, biased only at order 1/count. $\bar{e}^{obs}_j$ needs only $j$'s own spike times and two local population signals: after each spike of $j$, count spikes in the window in its own module (local pool or dendritic summation) and in the whole network (a population, LFP-like signal); outside is global minus own. No per-synapse coincidence is needed for it.

**Flow balance.** Outflow from $j$ is exactly $\pi_j$. Inflow $\sum_j J_{ij}$ equals $\pi_i$ only if $E_j$ is the same for every $j$; the difference is treated as unrecorded teleportation [Lambiotte & Rosvall 2012]. In a 48-neuron branching E–I test it averaged 1.5% of $\pi_i$ (at most 4.2%), with $E_j$ varying by 10% across neurons (Appendix A).

**The window.** Branching uses one time step, its natural hop. LIF and QIF use about three response times, so that $\omega_i \approx 1$. A longer window adds chance: the causal share of $j$'s transitions is roughly $\nu_j / (R w + \nu_j)$, with $R$ the population rate and $\nu_j$ the offspring of $j$ (about 3% for 100 neurons at 10 Hz and $w = 30$ ms). Chance does not dilute the causal step, which uses $c$ and not $\kappa$ (§4.1). It caps $M^{\ast}$ through a floor on $q_m$ (§3) and pulls the baseline $\bar{e}^{obs}_j$ toward the module constant $1 - p_{m(j)}$; Appendix A gives measured causal shares.

**The flow engine is the special case.** With no spikes there is no chance ($a = 0$) and no indirect correlation ($X = 0$), and every gain is 1, so $E_{kj} = \chi T_{kj}$, $E_j = \chi$ and $T^{obs} = T$. With the stimulus re-injecting a share $\alpha$ per step and $\chi = 1 - \alpha$,

$$
\pi = (1 - \alpha) T \pi + \alpha v(\mathbf{o}), \qquad \sum_i \pi_i = 1 \qquad \text{(1)}
$$

the walk that follows spikes forward from $j$ to $i$ with probability $T_{ij}$ and is re-injected by the stimulus with probability $\alpha$ per step. For any $\alpha > 0$ it has a unique solution, reached by power iteration (Theorem 1). It is also the mean-field description of the excitatory branching engine (SuperNeuroMAT branching mode, `leak=inf`, uniform threshold noise, $P(\text{spike}) = u / (2 \theta)$, output weights $(1 - \alpha) W / d$, Bernoulli stimulus input), whose mean offspring is $\Lambda = (1 - \alpha) T$ while no neuron saturates; on CartPole the measured lag-1 offspring regresses on $0.8 T$ with slope 1.04 (Appendix A). Spike shares of that engine follow (1), but its observed walk (8) also contains chance and indirect transitions.

### 2.2 Estimating flow and counts from spikes

Each neuron can estimate its own flow from its input,

$$
\hat{\pi}_i \leftarrow (1 - \beta) \hat{\pi}_i + \beta \left[ (1 - \alpha) \sum_j \frac{W_{ij}}{d_j} \hat{z}_j(t) + \alpha v_i \right], \qquad \hat{z}(t) = \frac{z(t)}{\sum_k z_k(t)} \qquad \text{(2)}
$$

and in batch form $\hat{\pi}$ is the frame's normalised spike counts, $\hat{\pi}_j = n_j / \sum_k n_k$, which is the observed walk's flow.

**Constant presynaptic output.** Spiking drive depends on $W_{ij}$, not $W_{ij} / d_j$. The engine applies a per-neuron output gain $1 / d_j$, one column-sum read, so the kick is $b_{ij} = \mathrm{sgn}_j w_0 T_{ij}$. This output normalisation and the homeostasis term (4d) are one abstraction, presynaptic resource conservation (§4.7).

**Global normalisation.** $\hat{z} = z / \sum_k z_k$ is network-wide; biologically it needs inhibitory gain control (§7).

**Pair counts and the covariance count.** The raw pair count $\kappa_{ij}$ estimates the walk, chance included (8), and is what the baseline uses. The **covariance count** $\kappa_{ij} - n_i n_j (k - 1) / k^2$ over the $k$ steps of a frame [Sejnowski 1977; Kempter et al. 1999] removes the chance term and estimates $\bar{n}_j (D_{ij} + X_{ij})$, the direct and indirect parts. It approximates the counter $c_{ij}$ where indirect terms are small, and is unsigned noise on weak synapses that a $w_{min}$ floor rectifies; the exact counter of each engine is in §5.

### 2.3 One walk per frame

Each frame $f$ brings a stimulus $\mathbf{o}_f$. In the flow engine, $\pi_f$ solves (1) with $v_f = v(\mathbf{o}_f)$. It is implicit, $\pi_f = \alpha (I - (1 - \alpha) T)^{-1} v_f$, and fast: $n$ hops of $x_{k+1} = (1 - \alpha) T x_k + \alpha v_f$ from $x_0 = v_f$ give

$$
\lVert x_n - \pi_f \rVert_1 \le (1 - \alpha)^n \lVert v_f - \pi_f \rVert_1 \le 2 (1 - \alpha)^n \qquad \text{(1a)}
$$

independently of network size; $\alpha = 0.2$ and $n = 20$ give at most 2.3%. Modular networks mix slowly and approach the bound (0.78 per hop against 0.8), so budget for it. One hop is a sparse matrix–vector product in software and one analogue read on a crossbar. In the branching engine one simulation step is one hop, and the module-level error of spike shares is sampling-limited at about $1.3 / \sqrt{S_{mod}}$, with $S_{mod}$ the spikes per module per frame, so frame windows are stated in spikes, not time (iteration 001). In continuous time the network relaxes with time constant about $\tau / \alpha$, a frame of 100–250 ms. Three timescales must be ordered: hops (or windows), then frames, then plasticity ($\lambda, \eta$ small), which is the split of §8.

**Proposal: carry-over.** To let latent communities carry state across frames, mix the previous frame's latent flow into the teleportation vector,

$$
v_f = (1 - \rho) v(\mathbf{o}_f) + \rho \frac{P_{\mathcal{L}} \pi_{f-1}}{p_{\mathcal{L}}(f - 1)} \qquad \text{(1b)}
$$

with $P_{\mathcal{L}}$ the restriction to latent neurons; $\rho = 0$ gives independent frames. With $\pi_{f-1}$ held fixed every gradient below is unchanged. Linear carry-over is a contraction with no attractors, so it forgets: two histories converge by a factor of 0.29–0.39 per frame ($\rho = 0.5$–$0.9$). Holding a cue needs a module-level nonlinearity on the carried vector (a soft winner-take-all over latent modules, still crossbar-native), an eligibility time constant at least as long as the cue-to-reward delay, and controller→latent routing that the map term does not seal (§4.5).

### 2.4 Communities as states: the extracted Markov model

Coarse-graining the frame's observed walk to modules gives a $K$-state, stimulus-conditioned Markov chain,

$$
T^K_{ba}(\mathbf{o}_f) = \frac{J_{ba}}{p_a} = \sum_{j \in a} \frac{\pi_{f,j}}{p_a} \sum_{i \in b} T^{obs}_{ij} \qquad \text{(1c)}
$$

the module-level chain of the actual spike sequence. In the flow engine $T^{obs} = T$, so each neuron's split of output across modules is fixed by $W$, and **the stimulus enters the module chain only by choosing which neurons in a module carry its flow.** Concentrating the stimulus in the controller makes the controller's column of $T^K$ the stimulus-dependent one (standard deviation across stimuli 0.0196 against 0.0006 when stimuli differ only in per-module drive). In the spiking engines the chain includes chance, so its diagonal is weaker than that of the causal chain; the causal part alone is the module offspring table $\hat{\Lambda}^K$ of §4.6.

**Proposal: world model.** With carry-over, $\pi_f = R v_f$ with $R = \alpha (I - (1 - \alpha) T)^{-1}$, so the latent mass carried out of module $a$ lands in latent module $b$ in proportion to the latent-$b$ share of $R u_a$ ($u_a$ the normalised restriction of $\pi_{f-1}$ to $a$). That defines an across-frame latent chain $P(b \mid a, \mathbf{o}_f)$, not yet checked in simulation. Biologically, metastable cluster switching takes hundreds of milliseconds [Litwin-Kumar & Doiron 2012; Mazzucato et al. 2015], a frame timescale, so this across-frame chain, not $T^K$ within a frame, is the one to compare with cortical state sequences.

### 2.5 Action readout

The action is read from the flow into the action modules,

$$
P(a \mid \mathbf{o}_f) = \frac{p_a(f)}{\sum_{a' \in \mathcal{A}} p_{a'}(f)}, \qquad a \in \mathcal{A} \qquad \text{(1d)}
$$

**Spiking readout.** The spiking analogue of (1d) is a **race**: the action is the module whose population spikes first. For independent Poisson populations with rates proportional to $p_a$ the race picks $a$ with probability exactly $p_a / \sum_{a'} p_{a'}$, but only once the walk has relaxed. The race therefore starts after a burn-in of $b = 10$ steps ($0.8^{10} \approx 0.11$), which brings its KL from (1d) to the sampling floor (iteration 002); biologically, a "hold your horses" subthalamic signal withholding the response [Frank 2006].

**Proposal: sharpened and adaptive policy.** (1d) has no temperature. The family

$$
P_\beta(a \mid \mathbf{o}_f) = \frac{p_a(f)^\beta}{\sum_{a' \in \mathcal{A}} p_{a'}(f)^\beta} \qquad \text{(1e)}
$$

has the matched score $h_b = \beta \left( \mathbb{I}[b = a^{\ast}] - \tilde{P}_\beta(b) \right) / J^{in}_b$ in (4b). A fixed $\beta$ is unstable (M2). The adopted Proposal (model v4) adapts $\beta$ to the trend of episode return: after episode $e$, with running averages $\bar{R}_{fast}$, $\bar{R}_{slow}$ over 20 and 100 episodes and $s_R$ the running deviation about $\bar{R}_{slow}$,

$$
\log \beta \leftarrow \log \beta + \kappa_\beta z_e, \qquad z_e = \mathrm{clip} \left( \frac{\bar{R}_{fast} - \bar{R}_{slow}}{s_R}, -2, 2 \right), \qquad \beta \in [1, \beta_{max}] \qquad \text{(1g)}
$$

with $\kappa_\beta = 0.01$, $\beta_{max} = 16$, and no update over the first 20 episodes. It is one global scalar and not a gradient. Because the matched score carries the factor $\beta$, it is also a global step-size gain on the task term. Biologically it abstracts the utility-driven gain of the adaptive gain theory of locus coeruleus–noradrenaline [Aston-Jones & Cohen 2005], with reward-range normalisation as in dopamine neurons [Tobler et al. 2005]. It needs episode boundaries, and a $\beta$ that falls after a collapse does not recover, since $z_e \approx 0$ on a new plateau.

**Proposal (spiking only): a race on count differences to a bound.** With $n_a(t)$ the action spikes of module $a$ counted from the burn-in, the action is

$$
a^{\ast} = a \quad \text{at the first } t \text{ with } n_a(t) - \max_{b \ne a} n_b(t) \ge \theta \qquad \text{(1h)}
$$

or the leader at the end of the frame. If counted spikes are independent draws with probabilities $p$, then for two actions the lead is a $\pm 1$ walk absorbed at $\pm \theta$, and by gambler's ruin [Feller 1968] $P(a^{\ast} = a) = p_a^\theta / (p_a^\theta + p_b^\theta)$: (1e) with $\beta = \theta$ exactly, at an expected $\theta^2$ spikes at a tie [Wald 1947]. For three actions it is close to (1e) (KL at most 0.0065). It is a discrete drift-diffusion model with mutual inhibition [Usher & McClelland 2001; Bogacz et al. 2006], so $\beta$ maps onto a decision threshold set by cortico-striatal and subthalamic circuits [Lo & Wang 2006; Bogacz et al. 2010; Cavanagh et al. 2011]. Action modules are nearly closed, so spikes arrive in bursts with overdispersion $F$, giving $\beta_{eff} \approx \theta / F$; the agent maps $\beta$ to $\theta$ through a measured table (iteration 010).

**Limits of the policy family.** $P(a \mid \mathbf{o})$ is a ratio of functions linear in $v$, a mixture of each controller neuron's routing. Recurrence inside the controller blurs the stimulus, and its credit depends on each postsynaptic neuron's own routing, which the module-level score of (4b) cannot represent; on trained weights controller→controller synapses carried 15–78% of the true score with cosine 0.03–0.45 on that block. So the controller is built without recurrence ($p_{cc} = 0$, a feed-forward encoder; iteration 006). For the same reason the controller uses **conjunctive** receptive fields: a policy that averages per-dimension votes reached 84 steps on CartPole, one vote per conjunctive cell 491 (iteration 001).

## 3. The map equation

Describe the walk with a two-level code: inside a module, name the next neuron or "exit" with a module-local codeword; on exit, name the next module from a global index. $L(M)$ is the average number of bits per step [Rosvall & Bergstrom 2008]:

$$
L(M) = q_{\curvearrowright} \log q_{\curvearrowright} - 2 \sum_m q_m \log q_m - \sum_j \pi_j \log \pi_j + \sum_m (p_m + q_m) \log (p_m + q_m) \qquad \text{(3)}
$$

with $q_m = \sum_{j \in m} \sum_{i \notin m} J_{ij}$ the exit flow of the observed walk. With $\pi$ held fixed, the weights enter $L$ only through the exit rates, and

$$
M^{\ast}_m := \frac{\partial L}{\partial q_m} = \log \frac{q_{\curvearrowright} (p_m + q_m)}{q_m^2} \ge 0
$$

because $q_{\curvearrowright} \ge q_m$ and $p_m + q_m \ge q_m$. **Leakage always costs bits**, and $M^{\ast}_m$ is the marginal number of bits per unit of exit flow from module $m$. It can be computed from the $K$ module population traces alone (spike shares and exit shares per module), and is the neuromodulatory broadcast of the rule. **Chance bounds it.** If no cross-module effect is negative ($D_{ij} + X_{ij} \ge 0$ for $i \notin m(j)$, as in an excitatory-only network), exit flow is at least its chance part, $q_m \ge \sum_{j \in m} \pi_j \sum_{i \notin m} a_i / E_j > 0$, so $M^{\ast}_m$ cannot diverge. Lateral inhibition can push exit flow below chance, and $q_m$ can then approach zero; in the flow engine there is no chance at all. So the implementation clips $M^{\ast}$ (at 20).

**The inverse problem.** Community detection solves $\min_M L(M; W)$. Map-STDP pins the controller and action modules and minimises over the weights, averaged over the frames the environment produces:

$$
\min_W D(W), \qquad D(W) = \mathbb{E}_f \left[ L(M; J_f) \right] \qquad \text{(3a)}
$$

with $J_f$ the observed walk of frame $f$. Latent modules may be pinned or re-detected from time to time (§8), which makes training an alternating minimisation. With each frame's flow held fixed, $D$ is an average of per-frame description lengths, each depending on $W$ only through that frame's link flows, so every result of §4 holds frame by frame and for $D$ by averaging. The modulator can use each frame's module statistics or running averages over 10–50 frames, which is also the more plausible timescale for a neuromodulator; the frame-level version is the special case, and in a flow-engine test the difference was negligible (Appendix A).

## 4. Plasticity as a three-factor rule

### 4.1 The rule

With $c_{ij}$ the signed counter of the frame (§5) and $\kappa_{ij}$ its pair counts, the structural update is

$$
\Delta W_{ij} = -\lambda \left( g_{ij} - \bar{g}^{obs}_j \right) c_{ij}, \qquad g_{ij} = \frac{\partial D}{\partial J_{ij}}, \qquad \bar{g}^{obs}_j = \sum_k T^{obs}_{kj} g_{kj} \qquad \text{(4)}
$$

**Two statistics, two jobs.** The baseline $\bar{g}^{obs}_j$ is the modulator averaged over $j$'s *observed* transitions, chance included, so it comes from pair counts (8). The step $c_{ij}$ must measure how the walk changes with $W_{ij}$, which is the direct effect, so it comes from the signed counter, whose requirement is

$$
\mathbb{E} \left[ c_{ij} \right] = \bar{n}_j D_{ij} = T_f r_j \mathrm{sgn}_j \tilde{\Lambda}_{ij}
$$

per spike of $j$, the change in $i$'s spikes within the window caused by $j$'s kick alone. Swapping either statistic breaks descent: pair counts as the step are mostly chance, which renormalisation cancels, so learning stalls; any baseline other than the observed walk's is off direction when sensitivities differ (§4.2, Appendix A).

For hardware, $g_{ij}$ depends only on the module pair $(m(i), m(j))$: a $K \times K$ table computed from module statistics and broadcast per module. $\bar{g}^{obs}_j$ is a per-neuron running mean; for the map equation it is $M^{\ast}_{m(j)} \bar{e}^{obs}_j$, built from $j$'s own spikes and two local population signals (§2.1). On a crossbar it is a per-row register; biologically the population signals are local, but the per-neuron average is an **abstraction**, as is $\bar{h}_j$ in (4b). The signed counter is presynaptic arrival times a postsynaptic sensitivity $\hat{G}_i$ (§5), a voltage-dependent rather than timing-dependent form of plasticity [Artola et al. 1990; Clopath et al. 2010]; a plausible carrier of $\hat{G}_i$ is an NMDA or Ca²⁺ readout of the distance to threshold. Description costs are available at the moment of the transition, so the structural term needs no trace; a biological modulator lags by 0.3–2 s [Yagishita et al. 2014] and would need one.

### 4.2 What the rule descends

**The local split.** While differentiating with respect to one weight, hold fixed everything a single synapse does not touch directly: the flow $\pi$ and chance $a$, the indirect term $X$, and the gains $G$ and window shares $\omega$. This extends the EM split, which held $\pi$ fixed. The dropped terms are the change of rates with $W$, chains and common input (order $\Lambda^2$), and $\partial G / \partial W$. With these held, $W_{ij}$ changes only the direct part of column $j$, $D_{kj} = \mathrm{sgn}_j h_k W_{kj} / d_j$.

**Lemma E.** If $D$ depends on $W$ only through the observed link flows $J$, with $g_{ij} = \partial D / \partial J_{ij}$, then under the local split

$$
W_{ij} \frac{\partial D}{\partial W_{ij}} = \frac{\pi_j \mathrm{sgn}_j}{E_j} \left[ \tilde{\Lambda}_{ij} \left( g_{ij} - \bar{g}^{obs}_j \right) - T_{ij} C_j \right], \qquad C_j = \sum_k \tilde{\Lambda}_{kj} \left( g_{kj} - \bar{g}^{obs}_j \right) \qquad \text{(9)}
$$

$C_j$ is the same for every $i$ in column $j$, so $T_{ij} C_j$ is a uniform rescaling of column $j$; it cannot change $T$, so it cannot change $D$, and $\sum_i W_{ij} \partial D / \partial W_{ij} = 0$. Term by term: $\pi_j / E_j$ is presynaptic flow per observed transition, a per-neuron step size; $\mathrm{sgn}_j$ flips the direction for inhibitory synapses; $\tilde{\Lambda}_{ij}$ scales the step by how much $j$ actually moves $i$; and $g_{ij} - \bar{g}^{obs}_j$ compares this synapse's marginal cost with the average over $j$'s observed transitions.

**Theorem E (the structural rule descends $D$).** If the counter is unbiased for the windowed direct effect, and $M^{\ast}$ and $\bar{e}^{obs}$ are taken at their expectations (ratios of large counts), then (4) averages to

$$
\mathbb{E} \left[ \Delta W_{ij} \right] = -\lambda \zeta E_j W_{ij} \frac{\partial D}{\partial W_{ij}} - \lambda T_f r_j \mathrm{sgn}_j C_j T_{ij}, \qquad \Delta D = -\lambda \zeta \sum_{ij} E_j W_{ij} \left( \frac{\partial D}{\partial W_{ij}} \right)^2 \le 0, \qquad \zeta = T_f \sum_k r_k \qquad \text{(10)}
$$

The second term is a column rescaling, orthogonal to the gradient. The preconditioner $E_j W_{ij}$ is diagonal and positive, and never needs computing. **The baseline must be the observed walk's.** Replacing $\bar{g}^{obs}_j$ by any other per-neuron baseline $b_j$ adds $-\lambda T_f r_j \mathrm{sgn}_j \tilde{\Lambda}_{ij} (\bar{g}^{obs}_j - b_j)$, proportional to $h_i T_{ij}$ within column $j$: a harmless column rescaling when $h$ is the same for every target of $j$, and a bias toward high-$h$ targets otherwise.

**The flow engine.** With $a = X = 0$, $\mathrm{sgn} = +1$ and equal $h$, $T^{obs} = T$, $C_j = 0$ and $E_j = \chi$, and Lemma E reduces to the Lemma,

$$
\frac{\partial D}{\partial W_{ij}} = \frac{\pi_j}{d_j} \left( g_{ij} - \bar{g}_j \right), \qquad \bar{g}_j = \sum_k T_{kj} g_{kj} \qquad \text{(5)}
$$

The counter's expectation is proportional to $J_{ij} = \pi_j T_{ij}$, which the flow engine uses in place of the counter, so (10) becomes Theorem 3, $\mathbb{E}[\Delta W_{ij}] = -\lambda W_{ij} \partial D / \partial W_{ij}$: gradient descent with each coordinate scaled by its own positive weight, so strong synapses move more and silent synapses stay silent. Without the baseline, an extra term $-\lambda J_{ij} \bar{g}_j$ acts as cost-weighted decay, not descent. To first order in $\lambda$, the baselined rule equals the unbaselined rule followed by rescaling each neuron's outputs back to the same $d_j$, so $\bar{g}_j$ reads as a presynaptic resource constraint. Row rescaling is not crossbar-native, so on hardware the baseline stays explicit. With heterogeneous sensitivities but still $a = X = 0$ and excitation only, $T^{obs}_{kj} = h_k T_{kj} / \bar{h}_j$ is the gain-weighted walk, and (9) gives the corresponding gain-weighted lemma.

### 4.3 Marginal cost, not pointwise cost

The *pointwise* cost of a step (its codeword length) averages to $L$; the *marginal* cost $g_{ij} = \partial D / \partial J_{ij}$ is how much $D$ grows with more flow along that transition. The modulator must broadcast the marginal cost. Broadcasting the codeword length also rewards landing on high-traffic neurons with cheap codewords, and in simulation it correlated 0.976 with the descent direction, against 0.9999 for the marginal cost.

### 4.4 The map equation

With $\pi$ fixed, $L$ depends on $J$ only through $q_m = \sum_{j \in m} \sum_{i \notin m} J_{ij}$, so

$$
g_{ij} = M^{\ast}_{m(j)} \mathbb{I}[i \notin m(j)], \qquad \bar{g}^{obs}_j = M^{\ast}_{m(j)} \bar{e}^{obs}_j \qquad \text{(6)}
$$

**The modulator of module $m$ is silent while activity stays inside and fires $M^{\ast}_m$ when it leaves.** The structural rule (4) becomes $\Delta W_{ij} = -\lambda M^{\ast}_{m(j)} (\mathbb{I}[i \notin m(j)] - \bar{e}^{obs}_j) c_{ij}$, and by (9) the gradient is

$$
W_{ij} \frac{\partial L}{\partial W_{ij}} = M^{\ast}_{m(j)} \frac{\pi_j \mathrm{sgn}_j}{E_j} \left[ \tilde{\Lambda}_{ij} G^{obs}(i, j) - T_{ij} C'_j \right], \qquad G^{obs}(i, j) = \mathbb{I}[i \notin m(j)] - \bar{e}^{obs}_j \qquad \text{(7)}
$$

with $C'_j = \sum_k \tilde{\Lambda}_{kj} G^{obs}(k, j)$. In the flow engine it is $(\pi_j / d_j) G(i, j)$ with $G(i, j) = \mathbb{I}[i \notin m(j)] - \bar{e}_j$, which is exactly the centred modulator of lag-1 Markov stability [Delvenne et al. 2010], reweighted per module by $M^{\ast}_m$; since $\sum_i W_{ij} G(i, j) = 0$, the multiplicative update conserves each $d_j$ and only reallocates output. The additive rule $\Delta W_{ij} = \eta \mathrm{STDP}(i, j) - \eta' G(i, j)$ is kept as the exact-gradient simulation baseline, not as the hardware rule.

**Sign pattern.** $G^{obs}(i, j)$ is positive across modules and negative within, so the direction of each update depends only on module membership and $\mathrm{sgn}_j$:

| Synapse | Across modules | Within module |
| --- | --- | --- |
| Excitatory | weakened | strengthened |
| Inhibitory | strengthened (lateral inhibition) | weakened |

$G^{obs}$ sets the direction, $\bar{e}^{obs}_j$ how hard, and $M^{\ast}_m$ how costly the module's leakage is. Chance changes the size of updates, never their sign; but it keeps $\bar{e}^{obs}_j$ near $1 - p_{m(j)}$ and $M^{\ast}$ finite however sealed a module becomes, so the map term keeps shrinking non-exempt cross-module excitation, and only the task term opposes it.

### 4.5 Routing exemption and latent output

The policy is carried by cross-module flow (controller→action and controller→latent), and the map term pushes every module to keep its flow inside. Applied to those pairs it drives the controller's exit flow to zero: the action modules starve, $M^{\ast}_{\mathcal{C}}$ diverges, the routing synapses' counts vanish, and the task term cannot learn (flow CartPole return 53 against 343 with protection; iteration 001). The implemented choice therefore **exempts** the routing pairs, $g_{ba} = 0$ for $a = \mathcal{C}$, $b \in \mathcal{A} \cup \mathcal{L}$. It has no free parameter, and the controller drops out of the structural term as an input layer. With $M^{\ast}$ computed from the full exit rates it is not an exact gradient (cosine 0.985); it is exact if $M^{\ast}$ uses exit rates without the exempt pairs. With routing exempt, the map term raises return in both backends (iteration 002) and seals latent and action modules almost completely, which is the map equation's optimum by construction (§3).

A dual routing term, kept only as an ablation in `experiments/m1.py`, minimises $L - \mu (J_{route} - q^{\ast})$ with $J_{route} = \sum_{b \in \mathcal{A} \cup \mathcal{L}} J_{b \mathcal{C}}$:

$$
g_{ba} = M^{\ast}_a \mathbb{I}[b \ne a] - \mu \mathbb{I}[a = \mathcal{C}, b \in \mathcal{A} \cup \mathcal{L}], \qquad \mu \leftarrow \max \left( 0, \mu + \eta_\mu \left( q^{\ast} - J_{route} \right) \right) \qquad \text{(4a)}
$$

Once latents and actions seal, the target $q^{\ast}$ is infeasible and $\mu$ diverges, so it acts as routing potentiation rather than a constraint (iteration 003).

**Latent modules become sinks.** The task term removes latent→action flow, because latent flow does not depend on the stimulus and only dilutes (1d); the map term seals latents against other latents and the controller, which is where its benefit comes from (iteration 003).

**Proposal: latent-output barrier.** To give each latent module a path to the actions,

$$
D_{LA} = D - \beta_{LA} \sum_{\ell \in \mathcal{L}} \mathbb{E}_f \left[ p_\ell \ln J_{\mathcal{A} \ell} \right], \qquad g_{ba} = M^{\ast}_a \mathbb{I}[b \ne a] - \beta_{LA} \frac{p_a}{J_{\mathcal{A} a}} \mathbb{I}[a \in \mathcal{L}, b \in \mathcal{A}] \qquad \text{(4e)}
$$

with $J_{\mathcal{A} \ell} = \sum_{b \in \mathcal{A}} J_{b \ell}$. It is still a $K \times K$ table, and (9) and (10) apply to $D_{LA}$. Its marginal cost grows like $1 / J_{\mathcal{A} \ell}$ while $M^{\ast}$ grows only like $\log (1 / q)$, so an interior fixed point exists, and the growing modulator cancels the vanishing count rate, so it regrows sealed latent→action synapses. On fully observed tasks latent→action flow carries no stimulus information and (4e) trades a little return for latent output; it is meant for M4.

### 4.6 The task term

Plain reward-modulated STDP, $\eta (R - \bar{R}) e_{ij}$ [Frémaux et al. 2010], gets no action credit when the action is sampled from flow shares independently of the spikes that build $e_{ij}$, and it stalls under sparse reward; spiking actor-critics use a TD error instead [Frémaux et al. 2013]. The policy depends only on firing rates, so the task score does not depend on how the walk is defined.

**Action-gated eligibility.** With $\pi_f$ fixed, take the surrogate policy $\tilde{P}(a \mid \mathbf{o}) = J^{in}_a / \sum_{a' \in \mathcal{A}} J^{in}_{a'}$, where $J^{in}_b = \sum_{i \in b} \sum_j J_{ij}$. Its score with respect to $J_{ij}$ depends only on the module of $i$:

$$
h_b = \frac{\mathbb{I}[b = a^{\ast}]}{J^{in}_{a^{\ast}}} - \frac{\mathbb{I}[b \in \mathcal{A}]}{\sum_{a' \in \mathcal{A}} J^{in}_{a'}}, \qquad \bar{h}_j = \sum_k T_{kj} h_{m(k)}
$$

and the trace accumulates the all-time signed counter times a credit,

$$
e_{ij} \leftarrow \left( 1 - \frac{1}{\tau_e} \right) e_{ij} + c^{tot}_{ij} \psi_{ij}, \qquad \psi_{ij} = h_{m(i)} - \bar{h}_j \ \text{(one-hop)} \quad \text{or} \quad \psi_{ij} = \Phi^K_{m(i)} \ \text{(cascade)} \qquad \text{(4b)}
$$

$c^{tot}$ counts a spike's direct offspring over all time rather than within the window; with a window of about three response times the two differ by about 5%, so one counter serves both terms. In the flow engine $c^{tot}_{ij}$ is replaced by its expectation $J_{ij}$, and by the Lemma the one-hop increment is $W_{ij} \partial \log \tilde{P}(a^{\ast} \mid \mathbf{o}) / \partial W_{ij}$, the one-hop surrogate of the exact score through the resolvent: cosine 0.98–0.998 on untrained weights, 0.83–0.99 on trained weights without controller recurrence (Appendix A). The one-hop credit is adequate when gains are equal; its baseline $\bar{h}_j$ is then harmless.

**Cascade credit (Proposal).** With signed offspring $\Lambda = \mathrm{diag}(G) B$, $B_{kj} = \mathrm{sgn}_j \chi_j T_{kj}$, and $s_i = \partial \log P / \partial r_i$, the exact weight-scaled score is $\Lambda_{ij} r_j \Phi_i$ minus a column rescaling, with the **cascade value** $\Phi = (I - \Lambda)^{-\top} s$: the score summed over all of a spike's linear-response descendants, through inhibitory paths too (Appendix A). So $c^{tot}_{ij} \Phi_i$ is the exact eligibility modulo column rescaling, **with no presynaptic baseline**. The crossbar-native version uses the module-level value

$$
\Phi^K = \left( I - \hat{\Lambda}^{K \top} \right)^{-1} s^K, \qquad \hat{\Lambda}^K_{ba} = \frac{\sum_{i \in b} \sum_{j \in a} c^{tot}_{ij}}{N_a}
$$

with $N_a$ the spike count of module $a$, estimated from the same counters (cosine 0.96–0.99 with the exact score in E–I tests). $\hat{\Lambda}^K$ is signed when inhibitory neurons are present, so the $K \times K$ system is solved directly rather than by a Neumann series. $\Phi^K$ is not a plausible neuromodulator payload: it needs an adjoint solve and a signed per-module signal every frame. The closest biological analogues are heterogeneous, target-specific dopamine [Engelhard et al. 2019; Hamid et al. 2021] and a selected-channel tag times a global $\delta$ [Gurney et al. 2015]. Treat it as a peripheral computation; its indexing by the *postsynaptic* module is a point in its favour, since volume transmission reaches synapses by their postsynaptic site (§6).

$\tau_e = 3$ frames is 0.3–0.75 s at 100–250 ms per frame, inside the dopamine-gated window [Yagishita et al. 2014]. The chosen action must be broadcast to the action modules; in basal-ganglia terms the selected channel is tagged by its own activity when dopamine arrives [Gurney et al. 2015], which is plausible but untested here.

**TD error.** The task update is $\Delta W_{ij} = \eta \delta_f e_{ij}$, with a global TD error from a linear critic on the controller's flow shares, trained by normalised LMS:

$$
\delta_f = r_f + \gamma V_{f+1} - V_f, \qquad V_f = u \cdot x_f, \qquad x_f = \frac{\pi_{\mathcal{C}}(f)}{p_{\mathcal{C}}(f)}, \qquad u \leftarrow u + \eta_V \delta_f \frac{x_f}{\lVert x_f \rVert^2} \qquad \text{(4c)}
$$

$V_{f+1} = 0$ on termination, and the critic bootstraps on truncation. **Proposal (model v4): TD normalisation.** The task term uses $\delta'_f = \sigma_\delta \delta_f / \sqrt{\overline{\delta^2}}$, with $\overline{\delta^2}$ a running mean square over 1000 frames and $\sigma_\delta = 0.04$, so the task update has the same size on every task; the critic still learns from the raw $\delta_f$. The critic weights $u$ live in the periphery and $\delta$ is a global scalar, so the term stays crossbar-native.

**Proposal: a separate critic.** The critic reads its own fixed layer of nonnegative sensory receptive fields $y_f = \phi(\mathbf{o}_f)$ (a Gaussian grid, or a Fourier basis with ON/OFF rates):

$$
V_f = u \cdot y_f, \qquad u \leftarrow u + \eta_V \delta_f \frac{y_f}{\lVert y_f \rVert^2} \qquad \text{(4c'')}
$$

The communities remain the actor's states; the value function's representation is designer-given, like a ventral-striatal critic on shared input [O'Doherty et al. 2004; Lansink et al. 2009; Foster et al. 2000]. On MountainCar the critic's resolution is what limits learning (iteration 012).

**Proposal: an entropy term on the action modules.** With $\tilde{P}(b)$ the frame's $\beta = 1$ action shares, a module-level modulator acts through the same counter and baseline as (4b), every frame and not gated by $\delta$:

$$
\Delta W_{ij} = \eta c_e c^{tot}_{ij} \left( \frac{g^H_{m(i)}}{J^{in}_{m(i)}} - \bar{g}^H_j \right), \qquad g^H_b = -\tilde{P}(b) \left( \log \tilde{P}(b) - \sum_{a \in \mathcal{A}} \tilde{P}(a) \log \tilde{P}(a) \right) \qquad \text{(4h)}
$$

on action modules ($g^H = 0$ elsewhere). With $\pi$ fixed and $c^{tot}$ replaced by $J$ (flow engine) it averages to $\eta c_e W \odot \nabla H(\tilde{P})$ [cf. Chung & Kozma 2020]. Optionally $c_e$ tracks a target entropy, $\log c_e \leftarrow \log c_e + k_H ( \phi_H \log \lvert \mathcal{A} \rvert - H(\tilde{P}) )$ per frame, with $\phi_H$ the target fraction of the maximum entropy [Haarnoja et al. 2018]. Logs in (4h) and here are natural (nats). It is useful only far below the task term's size, as a decaying exploration bonus (iteration 012). It abstracts usage homeostasis of action channels, not a neuromodulator; exploration-related noradrenaline acts on choice stochasticity, which $\beta$ already models [Tervo et al. 2014].

### 4.7 Homeostasis

The conductance range is held by a homeostasis term on all plasticity,

$$
\Delta W_{ij} = -\epsilon_h \left( d_j - 1 \right) \frac{\pi_j W_{ij}}{d_j} \qquad \text{(4d)}
$$

with $\pi_j$ the frame's spike share in the spiking engines. It is an exact uniform rescaling of column $j$: it pulls $d_j$ toward 1 and leaves $T$, the walk and $D$ unchanged. Its factor is presynaptic only; using the counter $c_{ij}$ here would move $T$ toward low-gain targets (for $d_j > 1$) whenever gains differ. Without it the task term ratchets $d_j$ upward through the $w_{min}$ floor (iteration 002). Biologically it is **presynaptic resource conservation (abstraction)**, and the output normalisation inside $b_{ij}$ (§2.2) is part of the same abstraction. Heterosynaptic plasticity is the closest known mechanism [Zenke & Gerstner 2017; Chistiakova et al. 2014]; documented conservation of total synaptic weight [Royer & Paré 2003] and synaptic scaling [Turrigiano et al. 1998] are postsynaptic, so presynaptic conservation is plausible but untested.

**Dale's law.** Every structural increment is bounded by a multiple of $W_{ij}$ ($\lvert c_{ij} \rvert \le n_j \lvert b_{ij} \rvert / 2 \theta$ in branching, $c_{ij} \propto W_{ij}$ in LIF and QIF), as is (4d), so for infinitesimal steps a magnitude never reaches zero and a synapse never changes sign. With finite steps the implementation clips magnitudes at $w_{min} > 0$.

## 5. The four engines

Every engine uses the same rule. What differs is the window, how the counter $c_{ij}$ is computed, and what each engine needs for that counter to be accurate.

| | Flow | Branching | LIF | QIF |
| --- | --- | --- | --- | --- |
| Window $w$ | one hop (expectations) | one time step | about 3 response times | about 3 response times |
| Chance $a_i$ | none (excitatory only); $w r_i$ if inhibition is added | $r_i$ per step | $r_i w$ | $r_i w$ |
| Counter $c_{ij}$ | expectation, $J_{ij}$ | exact, (11a) | $\tau_m \hat{G}_i b_{ij} n_j$, (11b) | $\tau \hat{G}_i b_{ij} n_j$, (11b) |
| Gain estimate | not needed | not needed (read off the input) | Siegert (11c) or regression | regression |
| Noise | none | built in | for the counter, not the walk | for the counter and against phase locking |
| Stability (12) | map form | map form | continuous form | continuous form; an E block above 1 allowed if inhibition holds it |

### 5.1 Flow

The flow engine is the case $a = X = 0$, equal gains, excitation only (§2.1): it computes $\pi_f$ by the hops of (1a) and uses the expected count $J_{ij} = \pi_j T_{ij}$ in place of $c_{ij}$, and its baseline is $M^{\ast} \bar{e}_j$. Its implemented rule $-\lambda (g_{ij} - \bar{g}_j) J_{ij}$ is therefore Theorem E term for term, and its results stand. **Proposal: flow with inhibition.** Rates come from the rectified fixed point $r = F(B r + I^{ext})$ with $B_{ij} = \mathrm{sgn}_j \chi T_{ij}$, and a chance floor matching a spiking engine's window, $E_{ij} = \max(0, w r_i + \mathrm{sgn}_j \chi G_i T_{ij})$, mirrors silencing; where the max binds, the counter's expectation is capped at $-T_f r_j w r_i$ and its derivative is zero. Untested.

### 5.2 Branching

The neuron is $u_i(t + 1) = \sum_j b_{ij} x_j(t) + 2 \theta \mu_i(t)$, $p(u) = \mathrm{clip}(u / 2 \theta, 0, 1)$, $x_i(t + 1) \sim \mathrm{Bernoulli}(p(u_i))$, with a per-type output scale $w_{0,j}$. The membrane forgets every step, so one step is one hop, and $E_{ij} = \mathbb{E}[p(u_i(t + 1)) \mid x_j(t) = 1]$ with $\kappa_{ij} = \sum_t x_j(t) x_i(t + 1)$. Given the state, removing $j$'s kick changes $i$'s firing probability by a known amount, so the counter is exact, with no linearisation:

$$
c_{ij} = \sum_t x_j(t) \left[ p \left( u_i(t + 1) \right) - p \left( u_i(t + 1) - b_{ij} \right) \right], \qquad \mathbb{E} \left[ c_{ij} \right] = \bar{n}_j D_{ij} \qquad \text{(11a)}
$$

For excitatory $j$ each term lies in $[0, b_{ij} / 2 \theta]$; for inhibitory $j$ it lies in $[b_{ij} / 2 \theta, 0]$ and equals $-p(u_i - b_{ij})$ when the kick pushes $u_i$ below zero, so $j$ removes only the spike probability $i$ actually had. The neuron already computes $u_i$; $p(u_i - b_{ij})$ costs one subtraction and one clip per presynaptic spike. The cheaper gated counter $(b_{ij} / 2 \theta) \mathbb{I}[0 < u_i < 2 \theta]$ is exact inside the linear range and misses removals at the lower edge, understating inhibition by 16% in the E–I test (correlation 0.9994 with the exact counter). The linear gain is $G_i = \Pr[0 < u_i < 2 \theta]$, and the signed offspring is $\Lambda_{ij} = \mathrm{sgn}_j (w_{0,j} / 2 \theta) G_i T_{ij}$; with $w_{0,E} / 2 \theta = 0.8$ the excitatory block is subcritical by itself.

### 5.3 LIF

With signed kicks, $\tau_m \dot{V}_i = -(V_i - E_L) + \mu_i(t) + \sigma_V \sqrt{2 \tau_m} \xi_i(t) + \tau_m \sum_j b_{ij} \sum_f \delta(t - t^f_j)$, reset to $V_r$ at $V_{th}$; the noise stands for background input. One kick of size $b$ is an input pulse of area $\tau_m b$, so in linear response it changes $i$'s spike count by $G_i \tau_m b$ for either sign, spread over about a response time. The counter therefore needs only the presynaptic spike count and a gain estimate:

$$
c_{ij} = \tau \hat{G}_i b_{ij} n_j \qquad (\tau = \tau_m \text{ for LIF}) \qquad \text{(11b)}
$$

an outer product of presynaptic count and postsynaptic gain, scaled by the kick, with no per-synapse accumulation during the frame. What it estimates is the **weight-scaled derivative** $b_{ij} \partial n_i / \partial b_{ij}$, the quantity Lemma E and Theorem E use. When kicks are comparable to the membrane noise, the transfer is convex on the scale of one kick, and the finite effect of removing the synapse is smaller (about 0.65 of the derivative on the v4 LIF network, Appendix A); the requirement $\mathbb{E}[c_{ij}] = \bar{n}_j D_{ij}$ then holds with $D_{ij}$ read as the linear-response direct effect. For an escape-noise neuron **without reset** the time-resolved version $c_{ij} = b_{ij} \sum_f \int_{t^f_j}^{t^f_j + w} e^{-(t - t^f_j) / \tau_m} \psi_i(t) dt$, with $\psi_i = \partial \rho_i / \partial V_i$, is exact to first order. With reset it overstates the effect several-fold, because each added spike's reset cancels most of the hazard increase, so the LIF engine uses (11b). The gain can come from the neuron's own input statistics, through the Siegert rate [Siegert 1951; Brunel & Hakim 1999],

$$
r = \left[ \tau_{ref} + \tau_m \sqrt{\pi} \int_{y_r}^{y_\theta} g(x) dx \right]^{-1}, \qquad \hat{G} = \frac{r^2 \tau_m \sqrt{\pi}}{\sigma_B} \left[ g(y_\theta) - g(y_r) \right], \qquad g(x) = e^{x^2} \left( 1 + \mathrm{erf}(x) \right) \qquad \text{(11c)}
$$

with $y = (V - \mu) / \sigma_B$ and $\sigma_B$ equal to $\sqrt{2}$ times the stationary standard deviation of the free membrane potential; the Siegert gain is an **abstraction** of what a neuron could know about its own operating point. The alternative, regression of $i$'s rate on its measured input across frames, is untested and biased upward by recurrent feedback. The counter is first-order: below threshold the transfer is convex, so excitatory kicks do slightly more than predicted and inhibitory kicks slightly less (Appendix A); separate gains for excitatory and inhibitory inputs would correct most of this (untested). The walk exists for a noiseless LIF too, since pair counts always exist; only the counter needs a smooth response, from input or escape noise or a surrogate gain. Without inhibition the convex LIF transfer ignites a high-rate state well below the branching engine's offspring of 0.8, so the excitatory-only LIF engine runs at a first-order offspring $\chi G = 0.5$. At 5–10 Hz and $w = 60$ ms its pair counts are about 99% chance (causal share about 1%), so $\bar{e}^{obs}_j$ sits near $1 - p_{m(j)}$ and $M^{\ast}$ near its chance ceiling.

### 5.4 QIF

The neuron is $\tau \dot{V}_i = V_i^2 + \eta_i + I^{ext}_i(t) + \sigma \sqrt{\tau} \xi_i(t) + \tau \sum_j b_{ij} \sum_f \delta(t - t^f_j)$, with spike and reset at $\pm V_{peak}$, in the normalisation of Montbrió et al. (2015); a kick $b$ adds exactly $b$ to $V$. Without noise the gain diverges near rheobase; with noise, rescaling $V = \sigma^{2/3} v$ removes $\sigma$, so $r = (\sigma^{2/3} / \tau) \tilde{g}(I / \sigma^{4/3})$ and $G_{max} \propto 1 / (\tau \sigma^{2/3})$. The area argument of §5.3 carries over with $\tau$ in place of $\tau_m$, so QIF uses the mean counter (11b) at a noisy operating point, with the gain from regression across frames. The regressor must be exogenous: the total input is dominated by the fixed inhibitory pools that the neuron's own spikes recruit, so rate regressed on total input gives a gain of the wrong sign, and on excitatory input alone it is biased by recurrence. The injected noise current is zero-mean and independent of the network, so rate regressed on it is unbiased (within 1% of the finite-difference slope); the per-neuron estimate is shrunk toward its module's and that toward the mean-field transfer gain. Knowing which part of the input is noise is an **abstraction** (on hardware, the injected noise). The fixed inhibitory pools are outside $W$ and outside the walk. The time-resolved phase-response gate has the right shape only for tonic firing; at the noisy operating point its shape correlation with ground truth was 0.70, so it is not used. An excitatory block with $\rho(\Lambda_{EE}) = 1.1$–$1.7$, measured at the tested QIF operating point, is allowed as long as inhibition keeps the full system stable (12). Noise is still needed to bound and smooth the gain and to break phase locking, which the counter's expectation assumes away. None of §5.4 has been tested with inhibition.

## 6. Locality and hardware mapping (memristive crossbars)

Target: $W$ as conductances in a crossbar (rows presynaptic, columns postsynaptic), with neurons and modulators in peripheral circuits.

| Operation | Crossbar implementation |
| --- | --- |
| Per-frame walk (1a), E-step (2) | $n$ analogue matrix–vector reads, with $\alpha v_f$ added in the periphery |
| $d_j$, module flows | $K + 1$ masked reads, summing current per row |
| Pair counts for $\bar{e}^{obs}_j$ | per-row counters of own-module and global spikes in the window after each spike of $j$ |
| Signed counter $c_{ij}$ | presynaptic pulse times a postsynaptic sensitivity ($\hat{G}_i$, or the gate of (11a)) times the device's own transmitted kick |
| Modulator $M^{\ast}$, table $g_{ba}$, task vector $h_b$, cascade value $\Phi^K$ | peripheral logic over $K \times K$ module statistics, broadcast per module |
| Baselines $\bar{g}^{obs}_j$, $\bar{h}_j$ | per-row registers |
| Applying (4), (4b), (4d) | row-pulse amplitude scaled by the per-row factor, one phase per target-module column mask; the map equation needs two per module (exit and home columns) |
| Task eligibility $e_{ij}$ | the only per-synapse state; e.g. volatile diffusive memristors [Wang et al. 2017] |
| TD error $\delta$, critic $u$ | a global scalar and a peripheral vector (4c) |
| Stability monitor (12) | $4 K^2$ sums of counters into a $2K \times 2K$ E/I table and a small eigenvalue computation, central |

No step needs the transposed element $W_{ji}$, and every modulator is a module-pair table. **Open.** Two parts are weight-proportional: the mean counter (11b) makes the structural step $W_{ij}$ times a row factor times a column factor, and (4d) is $W_{ij}$ times a row factor. Each needs either a conductance-proportional programming response or a read-then-write per device. The exact branching counter (11a) is also a nonlinear (clipped) function of one device's conductance near the bounds of $u_i$; the gated counter is linear in it. Which of these a given device technology supports is not settled.

**Biological locality.** A synapse $j \to i$ sits on $i$'s dendrite, so a modulator released in a module reaches synapses by their *postsynaptic* module. The map modulator is indexed by the *presynaptic* module, so biologically it needs $M^{\ast}$ roughly uniform across modules or a projection-specific signal; dendritic segregation of local and long-range inputs [Petreanu et al. 2009] could supply the bit $\mathbb{I}[a \ne b]$ but not $a$, and a presynaptic-locus factor is the natural carrier of a presynaptic-module index. Only $M^{\ast}$ ($K$ numbers, on a frame-to-episode timescale) is a neuromodulatory broadcast; the module-activity signals behind $\bar{e}^{obs}_j$ are local (§2.1). Neuromodulators are slow (0.3–2 s [Yagishita et al. 2014]) and at most regionally specific [Engelhard et al. 2019; Hamid et al. 2021], so a per-region scalar is plausible and a full $K \times K$ table is not. Astrocytes gate plasticity locally [Henneberger et al. 2010] and tile exclusive domains [Bushong et al. 2002], but a domain covers about $10^5$ synapses, not a module pair.

## 7. Inhibition and stability

Inhibition enters in three places: as below-chance transitions in the walk (§2.1), as a sign flip in the rule (§4.4), and as negative entries in the credit cascade (§4.6). The rule treats inhibitory synapses exactly as excitatory ones: the same formula, with $\mathrm{sgn}_j$ carried by the counter. Left alone, the map term makes inhibitory neurons purely lateral, so modules separate by excitation within and inhibition between them. **Plastic inhibition is not implemented**: the implemented networks (model v4) are excitatory-only, with normalisation done in software (§2.2).

**Stability.** The walk exists regardless, but linear response, and so the counters, need a stable operating point of the full E–I system. Around it, small rate deviations obey $\delta r(t + 1) = \Lambda \delta r(t)$ in the branching engine (a one-step map) and $\tau d \delta r / dt = (\Lambda - I) \delta r$ in LIF and QIF, with the full signed $\Lambda$, so

$$
\text{map (branching):} \quad \max_k \lvert \lambda_k(\Lambda) \rvert < 1, \qquad \text{continuous (LIF, QIF):} \quad \max_k \mathrm{Re} \lambda_k(\Lambda) < 1 \qquad \text{(12)}
$$

The map form is stricter: a strongly negative eigenvalue, which strong inhibition can create, is harmless in continuous time but makes a one-step map oscillate.

- **Excitation-only bound.** For $\Lambda \ge 0$ the largest eigenvalue is real (Perron–Frobenius), and with output normalisation every column sum is at most $\chi G_{max}$, so $\rho(\Lambda_{EE}) \le \chi G_{max}$ for every $W$. It still guarantees stability with no help from inhibition: sufficient, not necessary. With inhibition, $\max(\chi_E, \chi_I) G_{max} < 1$ bounds the column sums of $\lvert \Lambda \rvert$ and so guarantees both forms, conservatively.
- **Inhibition-stabilised regime.** An excitatory block with $\rho(\Lambda_{EE}) > 1$ is allowed when inhibition holds the full system stable [Tsodyks et al. 1997; Ozeki et al. 2009]. For one excitatory and one inhibitory population, $\Lambda = ((\lambda_{EE}, -\lambda_{EI}), (\lambda_{IE}, -\lambda_{II}))$ with all $\lambda \ge 0$, the continuous form holds when

$$
\lambda_{EE} < 2 + \lambda_{II}, \qquad \lambda_{EI} \lambda_{IE} > \left( \lambda_{EE} - 1 \right) \left( 1 + \lambda_{II} \right)
$$

so with $\lambda_{EE} > 1$ the second inequality says how strong the E→I→E loop must be.
- **Monitoring.** The counters give a $2K \times 2K$ module-level signed offspring table, excitatory and inhibitory parts kept apart (entries: summed $c^{tot}$ over source and target populations, divided by the source spike count). Check its spectral radius (branching) or largest real part (LIF, QIF) each frame.

**The risk.** Inside a module the map term raises $\lambda_{EE}$ (within-module excitation) and lowers $\lambda_{EI}$ (within-module inhibition); both push the determinant above toward zero. Under the excitation-only bound this cannot destabilise the network; in the inhibition-stabilised regime it can. Remedies, untested: exempt inhibitory→same-module synapses from the structural term ($g = 0$, like the routing exemption); set those synapses with a homeostatic inhibitory rule that holds excitatory rates at a target [Vogels et al. 2011]; or add a stability penalty as the monitored eigenvalue nears its limit.

**Forms of inhibition.** The walk is observed, so the form of inhibition only changes the counter's gain. Divisive (shunting) inhibition lowers $G_i$; subtractive inhibition lowers the input and, through rectification, $G_i$ as well. The excitatory–inhibitory branching process [Corral López et al. 2022] keeps firing probabilities non-negative by rectifying excitation minus inhibition, as the branching engine does. In cortex long-range inputs recruit feed-forward PV inhibition [Gabernet et al. 2005], which is the cross-module inhibition the map term favours.

## 8. Training

Per environment frame $f$:

1. **Observe.** Form $v_f$ from $\mathbf{o}_f$, concentrated on the controller; with carry-over, mix in the previous latent flow (1b).
2. **Walk.** Run the $n$ hops of (1a) (flow), or the $k = 100$-step spiking frame from a cold start and take spike shares as $\hat{\pi}_f$ (spiking), accumulating pair counts $\kappa$ and counters $c$, $c^{tot}$.
3. **Act.** Sample from (1d) or (1e), or run the race (or (1h)) after the burn-in.
4. **Module statistics.** Compute $p_a$, $J_{ba}$ and $q_a$ (from pair counts in spiking), $M^{\ast}$ and the modulator table $g_{ba}$ (with the exemption, and (4e) if used), and the per-row baselines $\bar{g}^{obs}_j$, $\bar{h}_j$; frame values or running averages over 10–50 frames.
5. **Plasticity.** Add (4b) to the traces; apply (4) with (6) and (4d) (in spiking, with $\lambda$ divided by the frame's spike count); after the next frame's walk, compute the TD error (4c), update the critic, and apply $\eta_e \delta_f e_{ij}$.
6. **Carry over.** Keep $P_{\mathcal{L}} \pi_f$ for the next frame.

**Stability.** $M^{\ast}$ is clipped (at 20) and magnitudes at $w_{min}$. The task rate is annealed, $\eta_e = \eta / (1 + e / 300)$ over episodes $e$: a constant rate stalls in the closed loop (iteration 002). It is a global scalar, crossbar-native. $\lambda$ is constant from frame 0. Multiplicative updates freeze near-zero synapses, so pruning and creating synapses (rewiring) is the route to new structure. Spiking engines with inhibition also need (12) monitored (§7).

**Local split.** $\pi_f$, the module statistics, chance, indirect terms and gains are fixed within each update. That needs the walk to converge within a frame and plasticity to be slow relative to frames. For the flow model the fixed-flow gradient keeps cosine 0.55–0.92 of the full gradient (Appendix A).

**Periodically.** Latent modules may be re-detected with Infomap or Leiden, the forward step of the alternating minimisation (3a); controller and action modules stay pinned. Initialisation is a stochastic block model with a recurrence-free controller.

### 8.1 Proposal: hybrid with the TD-STDP backbone

Keep the TD-STDP backbone [Chung & Kozma 2020] (Fourier input spikes, a feed-forward controller, TD error times an action-gated eligibility, a spiking critic) and add recurrent modules inside the actor, shaped by the map term; under the observed walk those modules may contain inhibitory neurons. Decision units read column spikes but send nothing back, so they stay outside the walk. Not implemented.

| Block | Structural term | Task term |
| --- | --- | --- |
| C→A, C→L (routing) | exempt | $\delta$ × eligibility |
| Excitatory, within a module | strengthens | $\delta$ × eligibility |
| Excitatory, across modules | weakens | $\delta$ × eligibility |
| Inhibitory, within a module | weakens, or exempt in the inhibition-stabilised regime (§7) | $\delta$ × eligibility |
| Inhibitory, across modules | strengthens (lateral inhibition) | $\delta$ × eligibility |
| Inputs→critic | none | critic rule |
| Columns→decision units | fixed | none |

The eligibility is (4b) with the signed counter; module activity supplies $\bar{e}^{obs}_j$ locally, and the $2K \times 2K$ offspring table is monitored each frame (§7). Noisy LIF first, QIF later.

**Proposal: the decision and the credit carried by spikes.** The decision units run the race (1h) on the columns' excitatory spike counts, and the task credit on an input synapse $j \to i$ of an actor neuron is the per-neuron score of the neuron's own noise model. For the noisy LIF (§5), with the pre-noise membrane $m_i(t) = a V_i(t-1) + (1-a)\mu + \xi_i(t)$, $a = e^{-\Delta t / \tau_m}$, and the step's noise $g_i(t) = (V_i(t) - m_i(t)) / \sigma_n$ with density $f$, the score at step $t$ is

$$
Y_i(t) = \frac{f(z_i)}{\sigma_n h_i} \text{ on a spike step}, \qquad Y_i(t) = -\frac{f'(g_i)}{\sigma_n f(g_i)} \text{ on an active no-spike step}, \qquad h_i = 1 - F(z_i), \quad z_i = \frac{\theta - m_i}{\sigma_n} \qquad \text{(13)}
$$

and $0$ in the refractory period. $Y_i(t)$ is $\partial \log p(V_i(t) \mid V_i(t-1), \xi_i(t)) / \partial \xi_i(t)$, so with $\partial \xi_i(t) / \partial W_{ij} \propto x_j(t-1)$ the eligibility increment is $Y_i(t) x_j(t-1)$, summed up to the step at which the race reaches its bound, and the update is $\delta$ times the trace. The conditional expectation of (13) given only the spike, $(S_i - h_i) f(z_i) / (\sigma_n h_i (1 - h_i))$, is the escape-noise score [Pfister et al. 2006; Frémaux et al. 2013] at the realised membrane; for this neuron it is zero-mean but not the gradient, because the membrane carries the noise forward. Both are local: a post factor, a pre factor and one broadcast $\delta$; (13) additionally needs the neuron to separate its step's noise from its input [Fiete & Seung 2006]. The recurrent synapses receive the column's summed score as a module-level signal, with the per-synapse form $Y_i(t) S_j(t-1)$ open. Checked in Appendix A (H1–H3); the procedure is `docs/columns-recipe.md`.

## 9. Assumptions, limitations and open problems

1. **Local split.** $\pi$, chance, indirect terms and gains are held fixed during updates, and (4) is exact only in expectation. The split drops indirect effects; in the inhibition-stabilised regime indirect feedback through inhibition is large (the "paradoxical" response), so the split may approximate the full gradient poorly there. Its cosine with the full gradient is unmeasured in any spiking engine.
2. **Coding choices.** Teleportation, escape and the flow-balance residual are not counted as exits (unrecorded teleportation); $M^{\ast}_m$ is clipped; the routing exemption is not an exact gradient (§4.5). The size of the flow-balance residual is measured only in the branching E–I test.
3. **Chance.** Chance does not dilute the causal step, but it pulls $\bar{e}^{obs}$ toward $1 - p_m$ and keeps $M^{\ast}$ finite, so the map term keeps shrinking non-exempt cross-module excitation; only the task term opposes it. A shorter window reduces chance until it cuts off real offspring.
4. **Inhibition.** The map term weakens within-module inhibition, which may be needed for stability (§7). Plastic inhibition is not implemented.
5. **Gains.** The LIF and QIF counters need gain estimates, including separate ones for inhibitory inputs; these are untested in recurrent networks.
6. **Policy.** The one-hop credit of (4b) is a surrogate score, so the controller is recurrence-free; (1g) needs episode boundaries and does not recover after a collapse.
7. **Memory.** Linear carry-over forgets geometrically (§2.3); $\rho$ and $\lvert \mathcal{L} \rvert$ are hyperparameters unless latents are re-detected.
8. **Evidence.** The observed-walk rule is checked on small standalone models (Appendix A). Closed-loop results so far come from the flow engine and the excitatory branching engine.

---

## Appendix A: Proofs and checks

**Theorem 1 (flow exists, is unique and is reachable).** If $T$ is column-stochastic and $\alpha \in (0, 1]$, (1) has one solution in the simplex, and power iteration with $F(x) = (1 - \alpha) T x + \alpha v$ converges to it geometrically. *Proof.* $F$ maps the simplex into itself, and $\lVert F(x) - F(y) \rVert_1 = (1 - \alpha) \lVert T (x - y) \rVert_1 \le (1 - \alpha) \lVert x - y \rVert_1$ since $\lVert T \rVert_1 = 1$. Banach's fixed-point theorem gives (1a). $\square$

**Pair counts (8).** $\mathbb{E}[\kappa_{ij}] = \sum_t \Pr[x_j(t) = 1] \mathbb{E}[N_i(t, t + w] \mid x_j(t) = 1] = \bar{n}_j E_{ij}$ by stationarity within the frame. $\square$

**How one weight changes the walk.** Only column $j$ of $T$ depends on $W_{ij}$: $\partial T_{kj} / \partial W_{ij} = (\delta_{ki} - T_{kj}) / d_j$. Under the local split $E_{kj}$ changes only through $D_{kj} = \mathrm{sgn}_j h_k T_{kj}$, so $\partial E_{kj} / \partial W_{ij} = \mathrm{sgn}_j h_k (\delta_{ki} - T_{kj}) / d_j$ and $\partial E_j / \partial W_{ij} = \mathrm{sgn}_j (h_i - \bar{h}_j) / d_j$, with $\bar{h}_j = \sum_k T_{kj} h_k$. A uniform rescaling of column $j$ leaves $T$, and so $E$, unchanged: $\sum_i W_{ij} \partial T_{kj} / \partial W_{ij} = 0$.

**Lemma E (9).** *Proof.* $\partial D / \partial W_{ij} = \sum_k g_{kj} \partial J_{kj} / \partial W_{ij}$, and with $J_{kj} = \pi_j E_{kj} / E_j$ and $\pi_j$ fixed, $\partial J_{kj} / \partial W_{ij} = (\pi_j / E_j) (\partial E_{kj} / \partial W_{ij} - T^{obs}_{kj} \partial E_j / \partial W_{ij})$. Substituting,

$$
\frac{\partial D}{\partial W_{ij}} = \frac{\pi_j \mathrm{sgn}_j}{E_j d_j} \left[ \sum_k g_{kj} h_k (\delta_{ki} - T_{kj}) - \bar{g}^{obs}_j (h_i - \bar{h}_j) \right] = \frac{\pi_j \mathrm{sgn}_j}{E_j d_j} \left[ h_i \left( g_{ij} - \bar{g}^{obs}_j \right) - \sum_k h_k T_{kj} \left( g_{kj} - \bar{g}^{obs}_j \right) \right]
$$

after adding and subtracting $\bar{g}^{obs}_j$ and using $\sum_k h_k T_{kj} = \bar{h}_j$. Multiplying by $W_{ij}$ and using $h_i T_{ij} = \tilde{\Lambda}_{ij}$ gives (9). Summing (9) over $i$, the first term gives $C_j$ and the second $C_j \sum_i T_{ij} = C_j$, so column sums vanish. $\square$

**Theorem E (10).** *Proof.* With $\mathbb{E}[c_{ij}] = T_f r_j \mathrm{sgn}_j \tilde{\Lambda}_{ij}$, $\mathbb{E}[\Delta W_{ij}] = -\lambda T_f r_j \mathrm{sgn}_j \tilde{\Lambda}_{ij} (g_{ij} - \bar{g}^{obs}_j)$. By (9), $\mathrm{sgn}_j \tilde{\Lambda}_{ij} (g_{ij} - \bar{g}^{obs}_j) = (E_j / \pi_j) W_{ij} \partial D / \partial W_{ij} + \mathrm{sgn}_j T_{ij} C_j$, and $r_j / \pi_j = \sum_k r_k$, which gives (10). The first-order change in $D$ is the update dotted with the gradient; the rescaling term contributes $\propto \sum_i W_{ij} \partial D / \partial W_{ij} = 0$, and $E_j > 0$, $W_{ij} \ge 0$ make every remaining term non-positive. $\square$

**Reduction to the Lemma (5) and Theorem 3.** *Proof.* With $a = X = 0$ and $\mathrm{sgn} = +1$, $E_{kj} = h_k T_{kj}$, $E_j = \bar{h}_j$, $T^{obs}_{kj} = h_k T_{kj} / \bar{h}_j$, and $C_j = \bar{h}_j (\sum_k T^{obs}_{kj} g_{kj} - \bar{g}^{obs}_j) = 0$, so dividing (9) by $W_{ij}$ gives $\partial D / \partial W_{ij} = \pi_j h_i (g_{ij} - \bar{g}^{obs}_j) / (\bar{h}_j d_j)$, the gain-weighted lemma. With equal $h$, $h_i / \bar{h}_j = 1$ and $T^{obs} = T$, which is (5). In that case $E_j = \chi$ for every $j$, the column-rescaling term of (10) vanishes, and (10) is $\mathbb{E}[\Delta W_{ij}] = -\lambda \zeta \chi W_{ij} \partial D / \partial W_{ij}$; with the flow engine's counter $J_{ij}$ the constant is 1 (Theorem 3). $\square$

**Theorem 2 (map-equation gradient).** With $\pi$ fixed, the $p_m$ and $\pi_j \log \pi_j$ terms of (3) are constant, and differentiating (3) in $q_m$, including through $q_{\curvearrowright}$, gives $(\log q_{\curvearrowright} + \log e) - 2 (\log q_m + \log e) + (\log (p_m + q_m) + \log e) = M^{\ast}_m$ (logs base 2; the $\log e$ terms cancel). Since $q_m$ counts $J_{ij}$ exactly when $j \in m$ and $i \notin m$, (6) follows; (9) then gives (7). $\square$

**Score with signed offspring (4b, cascade).** *Proof.* At the fixed point $r = F(B r + I^{ext})$ with $B_{kj} = \mathrm{sgn}_j \chi_j T_{kj}$, a change $\delta B$ gives $\delta r = (I - \Lambda)^{-1} \mathrm{diag}(G) \delta B r$ with signed $\Lambda = \mathrm{diag}(G) B$, invertible at a stable operating point (12). So $s^{\top} \delta r = \Phi^{\top} \mathrm{diag}(G) \delta B r$ with $\Phi = (I - \Lambda)^{-\top} s$. With $\partial B_{kj} / \partial W_{ij} = \mathrm{sgn}_j \chi_j (\delta_{ki} - T_{kj}) / d_j$,

$$
W_{ij} \frac{\partial \log P}{\partial W_{ij}} = \Lambda_{ij} r_j \Phi_i - W_{ij} \frac{\chi_j \mathrm{sgn}_j r_j}{d_j} \sum_k T_{kj} G_k \Phi_k
$$

and the second term is a column rescaling. Since $\mathbb{E}[c^{tot}_{ij}] = T_f \Lambda_{ij} r_j$, $c^{tot}_{ij} \Phi_i$ is the exact eligibility modulo column rescaling. Averaging $\Phi = s + \Lambda^{\top} \Phi$ over module $a$ with weights $r_j / R_a$, under $\Phi_i \approx \Phi^K_{m(i)}$, gives the module system of §4.6. $\square$

**Two-population stability (§7).** For $A = \Lambda - I$ with $\Lambda = ((\lambda_{EE}, -\lambda_{EI}), (\lambda_{IE}, -\lambda_{II}))$, a $2 \times 2$ system is stable iff $\mathrm{tr} A < 0$ and $\det A > 0$: $\mathrm{tr} A = \lambda_{EE} - \lambda_{II} - 2$ and $\det A = -(\lambda_{EE} - 1)(1 + \lambda_{II}) + \lambda_{EI} \lambda_{IE}$. For $\Lambda \ge 0$ the spectral radius is at most the largest column sum, $\chi G_{max}$ under output normalisation; a column of $\lvert \Lambda \rvert$ has one presynaptic type, so its sum is at most $\chi_{type(j)} G_{max}$. $\square$

**Bounded race (1h), two actions.** Each counted spike moves the lead by $\pm 1$ with probabilities $p_a, p_b$; absorption at $+\theta$ before $-\theta$ has probability $1 / (1 + (p_b / p_a)^\theta)$ (gambler's ruin). With bursts of size $B$ per independent draw, the walk moves in steps of $B$, giving $\beta_{eff} \approx \theta \mathbb{E}[B] / \mathbb{E}[B^2]$ to leading order. $\square$

**Numerical checks.** Exact identities are checked by finite differences (FD); three-factor rules by Monte Carlo (MC) or expected updates against $-W \odot \nabla D$. Rows marked (s) come from scratch scripts and are preliminary; the rest are repeated by the `mapstdp` self-checks or the iteration records. These are the flow-engine and branching-engine checks; the observed-walk checks follow.

| Check | Network | Metric | Value |
| --- | --- | --- | --- |
| (1a) walk; (1c) identity; (1b) forgetting | 15 neurons, 5 modules, $\alpha = 0.2$, 8 stimuli | worst per-hop contraction (bound 0.8); (1c) error; history contraction per frame at $\rho = 0.5$ / 0.9 | 0.78 modular, 0.28 dense; machine precision; 0.29 / 0.39 |
| controller column of $T^K$ | same | std across stimuli: controller / other columns / per-module drive only | 0.0196 / $\le 0.0035$ / 0.0006 |
| (7) and (3a) gradient, flow engine | 12 and 15 neurons, $\alpha$ 0.15–0.2 | max FD error | $\le 4 \times 10^{-7}$; $\le 10^{-9}$ per frame |
| (4) with (6), flow engine; pointwise codeword instead | 15 neurons, 4000 frames × 200 pairings; 12 neurons | MC correlation | 0.99996 per-frame, 0.99993 pooled modulator; codeword 0.976 |
| baseline as renormalisation | 13 neurons | relative difference | $0.025 \lambda$ (first order) |
| covariance count as the step (branching, excitatory) (s) | 80 neurons, 5000 spiking frames | MC cosine: covariance / causal / balanced | 0.9916 / 0.9787 / 0.8443 |
| (4a); (4e); routing exemption | 13–16 neurons, 6 stimuli; 80-neuron self-check | FD error; (4e) MC correlation and full-resolvent cosine ($\beta = 0.3$); exemption cosine with exact gradient | $\le 9 \times 10^{-10}$; 0.9999, 0.934; 0.985 |
| (4b) one-hop surrogate | 13 neurons; M3 networks | FD error; cosine with exact score | $\le 3 \times 10^{-10}$; untrained 0.981–0.998, trained $p_{cc} = 0.1$ 0.55–0.85, trained $p_{cc} = 0$ 0.83–0.99 |
| controller→controller credit (s) | trained M3, $p_{cc} = 0.1$ | share of exact score; block cosine of (4b) | 0.15–0.78; 0.03–0.45 |
| (1e) score; (1h) | 80 neurons, $\beta = 3$; exact absorbing chain and MC | FD error; $\lvert \mathcal{A} \rvert = 2$ identity, expected count, $\lvert \mathcal{A} \rvert = 3$ KL from (1e) | $\le 10^{-6}$; $\le 2 \times 10^{-15}$, $\le 10^{-12}$, $\le 0.0065$ |
| (4h) | 18 neurons, 3 seeds; repo SBM | FD error; MC correlation; full-resolvent cosine | $\le 2.1 \times 10^{-10}$; $\ge 0.9997$; 0.93 |
| branching offspring $\Lambda = 0.8 T$ (s) | CartPole, 400 frames × 16 heads | offspring on $0.8 T$: slope, correlation; formulas vs `core` | 1.04, 0.94; $\le 9 \times 10^{-16}$ |
| fixed-flow split, map equation (s) | random 36, CartPole init, CartPole trained, LunarLander init | cosine of fixed-flow with full gradient, flow engine | 0.85, 0.92, 0.55, 0.92 |

**Numerical checks for the observed-walk rule.** Rows E1–E4 are the standalone tests in `tests/test_emergent.py` (preliminary: small illustrative models, not the repo engines). Rows F1–F3 are from `.claude/skills/derivation-check/scripts/fdcheck.py --emergent` on a 16-neuron network in 4 modules (`make_net((4, 4, 4, 4))`), lognormal sensitivities $h$ (σ 0.5), chance $a_i \in [0.3, 0.6]$, fixed random indirect terms (s.d. 0.02), $\pi$ fixed; F4 is a scratch MC on the same network (s).

| Check | Network | Metric | Value |
| --- | --- | --- | --- |
| F1 Lemma E (9) | 16 neurons, excitatory only; and with 4 inhibitory (1 per module) | max relative FD error; max column sum of $W \odot \nabla L$ (relative) | $3.8 \times 10^{-7}$, $4.8 \times 10^{-16}$; $3.7 \times 10^{-7}$, $3.9 \times 10^{-16}$ |
| F2 Theorem E (10) | same, expected update with $\mathbb{E}[c]$ | cosine with $-E_j W \odot \nabla L$, column rescaling removed | 1.0000000 (both) |
| F3 reduction to (5) | same, $a = X = 0$, $\mathrm{sgn} = +1$, equal $h$ | max $\lvert$ Lemma E − Lemma $\rvert$ | $1.0 \times 10^{-17}$ |
| F4 (4) with sampled counts (s) | same; 4000 frames × about 2000 spikes, Poisson $n_j$, $\kappa$ and $c$; $M^{\ast}$, $\bar{e}^{obs}$ from sampled $\kappa$ | MC cosine with $-E_j W \odot \nabla L$, rescaling removed | 0.999999 (excitatory), 0.999998 (with inhibition) |
| E1 Lemma E | 16 neurons, 4 modules of 3 E + 1 I, lognormal $h$, chance, indirect terms | relative FD error; column sums | $6.7 \times 10^{-7}$; $7.7 \times 10^{-16}$ |
| E2 baselines | same, expected updates | cosine with descent direction: $\bar{e}^{obs}$ / counter-weighted / weight-normalised; first-order $\Delta L$ | 1.000000 / 0.978 / 0.979; $-1.993 \times 10^{-4}$ (equal to the predicted value) / $-1.942 \times 10^{-4}$ / $-1.984 \times 10^{-4}$; sign pattern of §4.4 on 100% of synapses |
| E3 score with inhibition | 8 networks of 24 neurons, 4 modules, 1 in 6 inhibitory, max Re λ 0.11–0.41 | FD error; cosine with $W \odot \nabla \log P$, rescaling removed: exact $c^{tot} \Phi$ / $\Phi^K$ / one-hop | $\le 1.9 \times 10^{-8}$; 1.000000 / 0.959–0.994 / 0.959–0.992 |
| E4 branching with inhibition | 48 neurons, 4 modules of 10 E + 2 I, $w_0 / 2 \theta = 0.8$, stimuli redrawn every 50 steps | negative transitions; gated vs exact (11a) correlation, inhibition understated; inflow imbalance mean / max; causal share E / I columns | none; 0.9994, 16%; 1.5% / 4.2%; 16% / 18% |
| E4 structural learning | same, 40 blocks × 5000 steps, columns renormalised | $L$ (bits), E weight leaving module, I weight to other modules: exact counter / pair count as the step | 4.611 → 4.096, 62% → 11%, 76% → 98% / 4.611 → 4.556, 62% → 53%, 76% → 83% |
| E5 single LIF neuron, signed kicks | $\tau_m = 20$ ms, voltage s.d. $0.25 \theta$, 60000 paired trials | simulated / predicted (11b) with (11c): $b = -0.05 \theta$ at 7, 14, 33 Hz; $-0.15 \theta$; $+0.05 \theta$ | 0.88, 0.88, 0.98; 0.85; 1.05 |
| H1 score (13), hazard $h$ | single noisy LIF, both noise kernels, frozen, $4 \times 10^4$ trajectories (`python -m mapstdp.lif`) | $\sum_t h$ vs spike count per neuron (s.d. units); $\mathbb{E}[\sum_t Y]$ of the escape-noise form | $\le 3.1$; $0.03 \pm 0.04$ |
| H2 score (13) vs FD | one neuron, one Poisson input, $R$ = a 20-ms count, common random numbers, $1.5 \times 10^5$ trajectories | $\mathbb{E}[\sum_t Y x(t-1) (R - \bar{R})]$ / central FD of $\partial \mathbb{E}[R] / \partial w$ | 1.001; escape-noise form per step 0.22–0.29, with the 20-ms pre trace 1.5–2.0 |
| H3 eligibility as computed (`python -m mapstdp.tdstdp`, h4) | one actor neuron, two Bernoulli inputs, `block_learn_ck_fast` driven as the runner drives it, $\tau_z \to \infty$, $\tau_q \to 0$ (the plain sum), $10^5$–$1.5 \times 10^5$ trajectories | per-input ratio to central FD of $\partial \mathbb{E}[R] / \partial W_j$ | 1.003, 1.005 (seed 11); 0.997, 1.009 (seed 3) |
| LIF engine (`python -m mapstdp.lif`) | single neuron, $\sigma_V = 0.2 \theta$, 1.3–29 Hz; frozen 3 × 10 network, 16 heads, 3 s, common random numbers; v4 networks | Siegert (11c) rate and gain vs simulation; counter (11b) vs weight-scaled derivative: correlation, ratio; vs finite removal: correlation, ratio; pair counts $E_{ij} / (r_i w)$; max Re eig of $\hat{\Lambda}^K$ | 0.998–1.008, 0.993–1.007; 0.945, 1.04; 0.940, 1.54; 0.99; 0.59–0.64 |
| QIF engine (`python -m mapstdp.qif`) | frozen CartPole-sized W, 32 heads, 300 frames, then 20 frames with common random numbers, 40 synapses removed, 20 neurons differentiated | counter (11b) vs finite removal: correlation, sum ratio; noise-instrument gain vs finite-difference f–I: correlation, ratio; max Re eig of $\hat{\Lambda}^K$ | 0.89, 1.04; 0.93, 1.01; 0.87 |

## References

- Artola, A., Bröcher, S., & Singer, W. (1990). Different voltage-dependent thresholds for inducing long-term depression and long-term potentiation in slices of rat visual cortex. *Nature*, 347, 69–72.
- Aston-Jones, G., & Cohen, J. D. (2005). An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance. *Annual Review of Neuroscience*, 28, 403–450.
- Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D. (2006). The physics of optimal decision making: a formal analysis of models of performance in two-alternative forced-choice tasks. *Psychological Review*, 113(4), 700–765.
- Bogacz, R., Wagenmakers, E.-J., Forstmann, B. U., & Nieuwenhuis, S. (2010). The neural basis of the speed-accuracy tradeoff. *Trends in Neurosciences*, 33(1), 10–16.
- Brunel, N., & Hakim, V. (1999). Fast global oscillations in networks of integrate-and-fire neurons with low firing rates. *Neural Computation*, 11(7), 1621–1671.
- Bushong, E. A., Martone, M. E., Jones, Y. Z., & Ellisman, M. H. (2002). Protoplasmic astrocytes in CA1 stratum radiatum occupy separate anatomical domains. *Journal of Neuroscience*, 22(1), 183–192.
- Cavanagh, J. F., Wiecki, T. V., Cohen, M. X., Figueroa, C. M., Samanta, J., Sherman, S. J., & Frank, M. J. (2011). Subthalamic nucleus stimulation reverses mediofrontal influence over decision threshold. *Nature Neuroscience*, 14(11), 1462–1467.
- Chistiakova, M., Bannon, N. M., Bazhenov, M., & Volgushev, M. (2014). Heterosynaptic plasticity: multiple mechanisms and multiple roles. *The Neuroscientist*, 20(5), 483–498.
- Chung, S., & Kozma, R. (2020). Reinforcement learning with feedback-modulated TD-STDP. *arXiv:2008.13044*.
- Clopath, C., Büsing, L., Vasilaki, E., & Gerstner, W. (2010). Connectivity reflects coding: a model of voltage-based STDP with homeostasis. *Nature Neuroscience*, 13(3), 344–352.
- Corral López, R., Buendía, V., & Muñoz, M. A. (2022). Excitatory-inhibitory branching process: a parsimonious view of cortical asynchronous states, excitability, and criticality. *Physical Review Research*, 4, L042027. (Venue not re-checked.)
- Delvenne, J.-C., Yaliraki, S. N., & Barahona, M. (2010). Stability of graph communities across time scales. *PNAS*, 107(29), 12755–12760.
- Engelhard, B., Finkelstein, J., Cox, J., et al. (2019). Specialized coding of sensory, motor and cognitive variables in VTA dopamine neurons. *Nature*, 570, 509–513.
- Feller, W. (1968). *An Introduction to Probability Theory and Its Applications*, Vol. 1 (3rd ed.). Wiley.
- Fiete, I. R., & Seung, H. S. (2006). Gradient learning in spiking neural networks by dynamic perturbation of conductances. *Physical Review Letters*, 97(4), 048104.
- Foster, D. J., Morris, R. G. M., & Dayan, P. (2000). A model of hippocampally dependent navigation, using the temporal difference learning rule. *Hippocampus*, 10(1), 1–16.
- Frank, M. J. (2006). Hold your horses: a dynamic computational role for the subthalamic nucleus in decision making. *Neural Networks*, 19(8), 1120–1136.
- Frémaux, N., Sprekeler, H., & Gerstner, W. (2010). Functional requirements for reward-modulated spike-timing-dependent plasticity. *Journal of Neuroscience*, 30(40), 13326–13337.
- Frémaux, N., Sprekeler, H., & Gerstner, W. (2013). Reinforcement learning using a continuous time actor-critic framework with spiking neurons. *PLoS Computational Biology*, 9(4), e1003024.
- Gabernet, L., Jadhav, S. P., Feldman, D. E., Carandini, M., & Scanziani, M. (2005). Somatosensory integration controlled by dynamic thalamocortical feed-forward inhibition. *Neuron*, 48(2), 315–327.
- Gershman, S. J., & Niv, Y. (2010). Learning latent structure: carving nature at its joints. *Current Opinion in Neurobiology*, 20(2), 251–256.
- Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D., & Brea, J. (2018). Eligibility traces and plasticity on behavioral time scales: experimental support of neoHebbian three-factor learning rules. *Frontiers in Neural Circuits*, 12, 53.
- Gurney, K. N., Humphries, M. D., & Redgrave, P. (2015). A new framework for cortico-striatal plasticity: behavioural theory meets in vitro data at the reinforcement-action interface. *PLoS Biology*, 13(1), e1002034.
- Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic algorithms and applications. *arXiv:1812.05905*.
- Hamid, A. A., Frank, M. J., & Moore, C. I. (2021). Wave-like dopamine dynamics as a mechanism for spatiotemporal credit assignment. *Cell*, 184(10), 2733–2749.
- Henneberger, C., Papouin, T., Oliet, S. H. R., & Rusakov, D. A. (2010). Long-term potentiation depends on release of D-serine from astrocytes. *Nature*, 463, 232–236.
- Kempter, R., Gerstner, W., & van Hemmen, J. L. (1999). Hebbian learning and spiking neurons. *Physical Review E*, 59(4), 4498–4514.
- Lambiotte, R., & Rosvall, M. (2012). Ranking and clustering of nodes in networks with smart teleportation. *Physical Review E*, 85, 056107.
- Lansink, C. S., Goltstein, P. M., Lankelma, J. V., McNaughton, B. L., & Pennartz, C. M. A. (2009). Hippocampus leads ventral striatum in replay of place-reward information. *PLoS Biology*, 7(8), e1000173.
- Li, H. L., & van Rossum, M. C. W. (2020). Energy efficient synaptic plasticity. *eLife*, 9, e50804.
- Litwin-Kumar, A., & Doiron, B. (2012). Slow dynamics and high variability in balanced cortical networks with clustered connections. *Nature Neuroscience*, 15(11), 1498–1505.
- Lo, C.-C., & Wang, X.-J. (2006). Cortico-basal ganglia circuit mechanism for a decision threshold in reaction time tasks. *Nature Neuroscience*, 9(7), 956–963.
- Mazzucato, L., Fontanini, A., & La Camera, G. (2015). Dynamics of multistable states during ongoing and evoked cortical activity. *Journal of Neuroscience*, 35(21), 8214–8231.
- Montbrió, E., Pazó, D., & Roxin, A. (2015). Macroscopic description for networks of spiking neurons. *Physical Review X*, 5(2), 021028.
- O'Doherty, J., Dayan, P., Schultz, J., Deichmann, R., Friston, K., & Dolan, R. J. (2004). Dissociable roles of ventral and dorsal striatum in instrumental conditioning. *Science*, 304(5669), 452–454.
- Ozeki, H., Finn, I. M., Schaffer, E. S., Miller, K. D., & Ferster, D. (2009). Inhibitory stabilization of the cortical network underlies visual surround suppression. *Neuron*, 62(4), 578–592.
- Pernice, V., Staude, B., Cardanobile, S., & Rotter, S. (2011). How structure determines correlations in neuronal networks. *PLoS Computational Biology*, 7(5), e1002059.
- Petreanu, L., Mao, T., Sternson, S. M., & Svoboda, K. (2009). The subcellular organization of neocortical excitatory connections. *Nature*, 457, 1142–1145.
- Pfister, J.-P., Toyoizumi, T., Barber, D., & Gerstner, W. (2006). Optimal spike-timing-dependent plasticity for precise action potential firing in supervised learning. *Neural Computation*, 18(6), 1318–1348.
- Plaçais, P.-Y., & Preat, T. (2013). To favor survival under food shortage, the brain disables costly memory. *Science*, 339(6118), 440–442.
- Redgrave, P., Prescott, T. J., & Gurney, K. (1999). The basal ganglia: a vertebrate solution to the selection problem? *Neuroscience*, 89(4), 1009–1023.
- Rosvall, M., & Bergstrom, C. T. (2008). Maps of random walks on complex networks reveal community structure. *PNAS*, 105(4), 1118–1123.
- Rosvall, M., Esquivel, A. V., Lancichinetti, A., West, J. D., & Lambiotte, R. (2014). Memory in network flows and its effects on spreading dynamics and community detection. *Nature Communications*, 5, 4630.
- Royer, S., & Paré, D. (2003). Conservation of total synaptic weight through balanced synaptic depression and potentiation. *Nature*, 422(6931), 518–522.
- Sejnowski, T. J. (1977). Storing covariance with nonlinearly interacting neurons. *Journal of Mathematical Biology*, 4(4), 303–321.
- Siegert, A. J. F. (1951). On the first passage time probability problem. *Physical Review*, 81(4), 617–623.
- Tervo, D. G. R., Proskurin, M., Manakov, M., Kabra, M., Vollmer, A., Branson, K., & Karpova, A. Y. (2014). Behavioral variability through stochastic choice and its gating by anterior cingulate cortex. *Cell*, 159(1), 21–32.
- Tobler, P. N., Fiorillo, C. D., & Schultz, W. (2005). Adaptive coding of reward value by dopamine neurons. *Science*, 307(5715), 1642–1645.
- Trousdale, J., Hu, Y., Shea-Brown, E., & Josić, K. (2012). Impact of network structure and cellular response on spike time correlations. *PLoS Computational Biology*, 8(3), e1002408.
- Tsodyks, M. V., Skaggs, W. E., Sejnowski, T. J., & McNaughton, B. L. (1997). Paradoxical effects of external modulation of inhibitory interneurons. *Journal of Neuroscience*, 17(11), 4382–4388.
- Turrigiano, G. G., Leslie, K. R., Desai, N. S., Rutherford, L. C., & Nelson, S. B. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. *Nature*, 391, 892–896.
- Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: the leaky, competing accumulator model. *Psychological Review*, 108(3), 550–592.
- Vogels, T. P., Sprekeler, H., Zenke, F., Clopath, C., & Gerstner, W. (2011). Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks. *Science*, 334(6062), 1569–1573.
- Wald, A. (1947). *Sequential Analysis*. Wiley.
- Wang, Z., Joshi, S., Savel'ev, S. E., et al. (2017). Memristors with diffusive dynamics as synaptic emulators for neuromorphic computing. *Nature Materials*, 16, 101–108.
- Wilson, R. C., Takahashi, Y. K., Schoenbaum, G., & Niv, Y. (2014). Orbitofrontal cortex as a cognitive map of task space. *Neuron*, 81(2), 267–279.
- Yagishita, S., Hayashi-Takagi, A., Ellis-Davies, G. C. R., Urakubo, H., Ishii, S., & Kasai, H. (2014). A critical time window for dopamine actions on the structural plasticity of dendritic spines. *Science*, 345(6204), 1616–1620.
- Zenke, F., & Gerstner, W. (2017). Hebbian plasticity requires compensatory processes on multiple timescales. *Philosophical Transactions of the Royal Society B*, 372(1715), 20160259.
