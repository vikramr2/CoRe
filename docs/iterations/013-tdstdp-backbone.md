# Iteration 013: TD-STDP backbone on the branching engine (step 1)

**Date:** 2026-10-23.

- **Model:** none. This is a baseline, not Map-STDP; `docs/model.md` and `docs/derivation.md` are unchanged.
- **Plan:** `jiggly-chasing-sparkle` (user-approved), step 1 of 4: reproduce the feed-forward TD-STDP actor-critic of Chung & Kozma 2020 [CK] (arXiv:2008.13044) on our engine. The later steps are removing Adam, action modules with a race readout, and QIF.
- **Compute:** local Linux workstation (72 cores, at most 40 workers, `OMP_NUM_THREADS=1`), no SLURM. Every number below comes from `python -m experiments.tdstdp --summary`.
- **Agents:** `snn-expert` (implementation, runs). The neuroscientist filter is deferred to step 3, because this step deliberately reproduces off-hardware parts: softmax, Adam, batch, signed weights and the per-synapse trace pair.
- **Reference:** the authors' code, `reference/td_stdp` (commit `a5cb7f4`, BindsNET 0.2.7; gitignored), read but not run. Where it disagrees with the paper, the code is the reference (§3.2).
- **Status:**
  - The pre-audit E1 failure was a reproduction error (§8).
  - **E1′:** passes its gate with BindsNET's LIF (20/20 solved, median $T_s$ 168) and fails it on the branching engine (4/20).
  - **E3′:** passes its gate (7/10 solved). $T_f$ matches CK, but $T_s$ is about 2.6× CK's.
  - **E4′:** without Adam it solves 0/10.
  - **Speed-ups (§10–§11):** the no-Adam block sum passes its gate (7× faster), and `--fast` is 1.8× faster with identical returns.
  - **Flow limit:** solves CartPole 20/20, so escape noise is what fails branching's strict $T_s$.
  - **Reported baseline:** `reports/tdstdp-baseline/` (§12). The numbers in §5 and §11 are development results.

## 1. Goal

Find out whether CK's backbone learns on the SuperNeuroMAT branching engine before building modules on it. The gate is CK's CartPole configuration without Adam: at least 18/20 solved runs and a median $T_s$ of at most 340 (2× CK).

- The first implementation (E1) followed the paper's equations and failed the gate. An audit against the authors' code (§8) found that the failure was a reproduction error.
- The code now follows the authors' code. The gate was re-run on fresh seeds as E1′ (§5.1), and LunarLander as E3′ and E4′ (§5.2), each pre-registered in §9.

## 2. What was built

- `mapstdp/tdstdp.py`:
  - `Layout`: the post populations, ordered critic E, critic I, then actor E and I per action.
  - `simulate`: the `spiking.run_heads` branching step without presynaptic normalisation, $P(\text{spike}) = \mathrm{clip}(W x(t-1))$ with $2\theta = 1$.
  - `simulate_snm`: the same step on SuperNeuroMAT.
  - `simulate_bindsnet`: BindsNET 0.2.7's `LIFNodes` as the authors configure them. A spike through w adds w mV, the threshold is 13 mV above rest, the reset is to rest, τ = 100 ms, there is no refractory period, and the neuron is deterministic.
  - `block_learn_ck`: the authors' `FBTDSTDP` rule in their per-ms order and trace scale (§3.1), summed in closed form over a block of L ms with W frozen. L = 1 is their per-ms rule exactly.
  - `block_learn_ck_fast` (`--fast`, §10 S2): the same update summed over copies in one matrix product. It equals `block_learn_ck` up to floating-point summation order. It optionally takes a constant per-ms z increment, which the flow engine uses.
  - `flow_step` (`--engine flow`, §10 S3): the branching step's noise-free limit. It uses input rates instead of spikes and spike probabilities $\mathrm{clip}(W f)$, and adds the input-autocorrelation term $W_{ji} a_p f_i (1 - f_i)$ to the z increment.
  - `Adam`: block form with per-ms-matched moments (§3.1 item 7).
  - Pre-audit pieces, kept for `--legacy`: `block_learn` (the paper's Eqs. 72–78 without the trace scale), and `simulate_lif` (CK Eq. 1, or SuperNeuroMAT's subtractive leak).
  - `pre_trace`, `rates`, `softmax`, `entropy_coef` (CK Eq. 25).
  - The `__main__` self-check, registered in `tests/test_regression.py`.
- `experiments/tdstdp.py`:
  - `PRESETS`: the authors' `config_cp.ini` and `config_ll.ini`, with η in mV.
  - `CKRunner`: a transcription of `main.py`, `Snn_actor`, `Reward_fn` and `util.batch_envs`.
  - `ck_encoder`: `util.State_to_spike_Fourier` with the authors' bounds. CartPole has 81 features and 1 input neuron each; LunarLander has 17 features and 16 input neurons each, so 272 inputs.
  - `calibrate_ck`: the E4′ η rule.
  - Checkpoints every 50 pooled episodes (`*.ckpt.npz`, gitignored, deleted on completion), and a `--summary` that keeps the two code families apart.
  - Results go to `experiments/results/tdstdp_<task>_ck[_<tokens>]_s<seed>.json`.
  - `--legacy` runs the pre-audit `Runner`, `calibrate` and `LEGACY` presets. It reproduces the committed E1/E2 results bit for bit (checked on the first 25 episodes of E1 seed 0 and E2 seed 1), under their original names `tdstdp_<task>[_<tokens>]_s<seed>.json`.
- Golden is bit-identical. `pytest -q tests` passes (8 tests) and `ruff` is clean.

## 3. Decisions and deviations

### 3.1 The authors' code, as implemented (E1′, E3′, E4′)

1. **Weight units.** BindsNET adds w mV to the membrane per presynaptic spike, against a 13 mV gap from rest to threshold. A branching weight is therefore $w_{mV}/13$ (`MV` = 13). The authors' η and Adam step are in mV, so they are divided by 13 too. `--engine lif` keeps W in mV.
2. **Trace scale.** The code's traces carry $1/\tau$ factors that CK Eqs. 72 and 77 omit: $z \mathrel{+}= P X/\tau_z$ and $q \mathrel{+}= (A - p) z/\tau_q$. η is folded into the pre trace, $P \mathrel{+}= \nu x$; for SGD this is equivalent to multiplying the update by ν.
3. **η and readout gain** come from the configs: CartPole $\nu_c$ = 0.005, $\nu_a$ = 0.01 and $\alpha_a$ = 50; LunarLander matches Table 1.
4. **Timing.** These are the corrected facts; the pre-audit reading is in §3.3.
   - Each env step is a 20 ms window. Window k presents $s_k$ together with $(a_{k-1}, p_{k-1}, r_{k-1})$.
   - $a_k$ is sampled from the softmax at the end of window k, so it is the response to $s_k$. The feedback is $A - p$, with p fixed at the probability $a_k$ was sampled with.
   - The update at ms t is $\delta_t z(t-1)$ for the critic and $\delta_t q(t) + c_e g z(t-1)$ for the actor. Here $q(t)$ is built from $z(t-1)$, and $\delta_t = e^{-1/\tau} V_t + e^{-1/2\tau} r - V_{t-1}$.
   - The target is $V := 0$ in the last 2 ms only after a true termination. At the time limit (CartPole 500, LunarLander 1000) the code bootstraps.
   - Each episode presents $s_0$ for six windows: one reset window (traces, rates and membranes cleared) and five warm-up windows. All six have δ = 0. The entropy term still applies, and q still accumulates with the stale action.
   - LunarLander resamples the action every 40 ms. Its timer is not reset at episode ends.
5. **Batch.** The H copies run in lockstep, and the update is the mean over all H copies, including those in a reset or warm-up window (BindsNET's batch reduction). Episodes are pooled over copies in the order they end, as in `main.py`. $T_f$ and $T_s$ are counted on the pooled sequence. A run stops when it is solved with at least `episodes` (3000 for LunarLander) pooled episodes, or at `max_eps` (10000).
6. **Block form.** CartPole runs at B = 1, which is the authors' rule exactly. LunarLander runs at B = 20, freezing W for one env step (a deviation, pre-registered in §9). On this machine B = 1 costs 45 ms per copy-frame against 2.9 ms at B = 20.
7. **Adam.** The code's Adam receives the ν-scaled gradient with ε = 1e-9. On our unit-trace gradient that is a step of ν/13 per ms and $\varepsilon = 10^{-9}/\nu$. At B = 20 the block Adam with per-ms-matched moments steps once per block (its acceptance gate is in §3.3 item 6).
8. **E4′ η rule.** Fixed before the run (§9): $\eta = \nu_{Adam}/\mathrm{RMS}(g)$ per population. For seed 0 this gave 1.1e-3 mV for the critic and 1.7e-3 for the actor (RMS 0.110 and 0.037).
9. **Initial weights and encoder.** W starts at U(0, 0.1) mV for CartPole and U(0, 0.125) mV for LunarLander. The Fourier bounds are `main.py`'s `state_lim`: CartPole velocities ±3.2 and angle ±π/12; LunarLander (−1, −0.2, −1, −1, −1, −1, 0, 0) to (1, 2, 1, 1, 1, 1, 1, 1).
10. **Reproduced oddities:**
    - The first window of a run is live, with a random action and $p = 1/K$, as in `main.py`.
    - A reset clears every copy's previous input spikes, because BindsNET's `Input` reset ignores the index.
11. **Not reproduced:**
    - LunarLander-v3 instead of v2.
    - Gymnasium seeding instead of global seeding.
    - Float64 instead of float32.

### 3.2 Paper against the authors' code

The code is the reference wherever the two disagree. The adopted values are in the presets.

| # | Item | Paper | Authors' code (adopted) |
| --- | --- | --- | --- |
| 1 | Synaptic input | Eq. 1 with Eq. 44: a spike moves the membrane by $Rw/\tau$ | A spike adds w mV, so the conversion is ÷13, not ÷1300 |
| 2 | Trace scale | Eqs. 72 and 77, no $1/\tau$ | $z \mathrel{+}= PX/\tau_z$ and $q \mathrel{+}= (A-p)z/\tau_q$ |
| 3 | CartPole critic η | 2.5e-3 (Table 1) | 0.005 (`critic_lr`) |
| 4 | CartPole $\alpha_a$ | 25 (Table 1) | 50 (`actor_m`) |
| 5 | Weight decay (LunarLander) | $c_w$ = 1e-8 (Table 1) | `weight_reg` is never passed to the rule, so there is no decay |
| 6 | Target $V' := 0$ | "in the last 2ms of an episode" | only on true termination; bootstrapped at the time limit |
| 7 | Action–feedback timing | not specified | sampled from the response to the current state; p fixed at the sampling probability |
| 8 | Warm-up | 100 ms, no learning | 6 windows (120 ms), no TD learning, but the entropy term and the q accumulation run |
| 9 | Batch of 16 | "batch size of 16" | 16 lockstep copies, with episodes pooled over copies for $T_f$, $T_s$ and the 3000/10000 caps |
| 10 | Encoder bounds and weight init | not given | `state_lim` and U(0, `init_scale`) mV |
| 11 | Learning-rule factor A− | 0 | `nu[0]` = 0, so it agrees |

### 3.3 Pre-audit choices (E1, E2 and the stopped first E3/E4; `--legacy`)

These are what produced the pre-fix results in §5.3. Items 1, 3 and 4 are superseded by §3.1.

1. **Weight units (superseded).**
   - We read CK Eqs. 1 and 44 as a membrane step of $Rw/\tau$ and divided η, the Adam step and $1/c_w$ by 1300.
   - A debug run with 13 destroyed the network within one episode. That run lacked the trace scale (§3.2 row 2), so its steps were about 10× (critic) and 800× (actor) the authors'.
   - With the trace scale, 13 is correct (§8).
2. **Weight init:** $W \sim U(0, w_0)$, with every post at a 0.2 per ms spike probability for the mean input over 5000 random-policy states.
3. **Batch (superseded):** asynchronous copies with 3000 episodes each. The update was averaged over the learning copies, and the returns over copies per episode index.
4. **Timing (superseded):**
   - The action was sampled from the softmax at the end of the previous window, so it was the response to $o_{k-1}$, a 20 ms observation delay.
   - The feedback was $A - s(t)$, with the drifting softmax.
   - $V := 0$ in the last 2 ms on termination and on truncation. The record's earlier "as CK do" was wrong for truncation.
   - Five warm-up windows.
5. **Block form:** CartPole at B = 1 and LunarLander at B = 20.
6. **Block Adam** (kept in §3.1):
   - The first form fed the block sum to Adam with $\beta^{20}$. It collapsed 3/10 CartPole seeds at B = 20, against 0/10 for per-ms Adam.
   - The adopted form feeds $G/L$ to m and $G^2/L$ to v with weights $1-\beta^L$, and steps $L \cdot lr \cdot \hat{m}/\sqrt{\hat{v}}$. It is identical to Adam at L = 1.
   - Its gate was B = 20 against E2 on seeds 0–9: MW p 0.14, no collapses. It passed.
7. **E4 rule:** the Adam-equivalent step at init, measured on the pre-audit loop.
8. **Encoder:** `tasks.critic_layer` (`f2c4hb`, `f1c1b`) with `tasks.FOURIER_BOUNDS`.

## 4. Self-check (`python -m mapstdp.tdstdp`)

| Check | Result |
| --- | --- |
| (a) numpy branching step = SuperNeuroMAT raster (7 inputs, 11 posts, 300 ms) | identical, rate 0.394 |
| (a′) `lif-snm` = SuperNeuroMAT with a finite leak (2000 ms) | identical, rate 0.127 |
| (b) `block_learn` closed form = per-ms loop, L = 1 and L = 20 | max relative error $4 \times 10^{-16}$ |
| (b′) `block_learn_ck` closed form = a literal per-ms transcription of `FBTDSTDP` (trace scale, q from $z(t-1)$), L = 1 and L = 20 | max relative error $6 \times 10^{-16}$ |
| (c) critic STE: Monte Carlo $E[z]$ against finite-difference $\partial E[V]/\partial W$ | cosine 0.9975 global, 1.0000 per post row |
| (d) actor rule (pre-audit order) on a 2-armed bandit: $E[r(A) q]$ against finite-difference $\partial E[r]/\partial W$ | cosine 0.9685 global, 1.0000 per row |
| (d′) actor rule in the authors' order (p fixed, action from the window's end, `block_learn_ck`), same bandit | cosine 0.9840 global, 1.0000 per row |
| (b″) `block_learn_ck_fast` = the copy sum of `block_learn_ck`, and with a constant z increment = the per-ms loop, L = 1 and L = 20 | max relative error $6 \times 10^{-16}$ |
| (e) flow limit: Monte Carlo mean of the sampled rule's update (W, f, δ and feedback fixed; 5000 blocks) against the flow update | cosine 0.99997, norm ratio 0.999 |
| closed loop, `--engine snm` against numpy (CartPole, 3 episodes, pre-audit loop) | identical returns and W |

In branching mode the STE (CK Theorem 1) is exact up to a per-postsynaptic gain $p_j$: $E[z_{ij}] \propto p_j f_i$, while the exact gradient is proportional to $f_i$. Rows are therefore exact, and the global cosine drops only as far as the post rates differ.

## 5. Results

The columns are as follows; all of them come from `--summary`.

- **Solved:** CK's criterion. For CartPole this is 100 consecutive episodes at 500; for LunarLander, a trailing-100 mean of at least 200.
- **$T_f$:** the first episode at the perfect score, mean ± sd over the runs that reach it [count].
- **$T_s$:** the last episode of the first solved window. The median counts unsolved runs as infinite.
- **$T_{475}$:** secondary and not pre-registered for the pre-audit runs. It is the first episode whose trailing-100 mean reaches 475 (Gymnasium's threshold).
- **Return:** the mean of the last 100 episodes, mean ± sd over seeds.
- **≥ 475:** the number of runs whose last-100 return is at least 475.
- **Collapsed:** a post hoc descriptor, last-50 mean below 15. These runs end on a constant action, return 9.4.
- **ms/frame:** wall time per env step per copy (20 ms simulated) with about 30–40 jobs on the machine.
- **Episode counting:** authors'-code runs count episodes pooled over copies. Pre-audit LunarLander counted per copy.

**References.** CK, 10 runs: CartPole $T_f$ 57.0 ± 24.0 and $T_s$ 169.5 ± 23.5, all solved; LunarLander $T_f$ 383.8 ± 71.5 and $T_s$ 2575.2 ± 666.5. Model v4 on CartPole: flow 486.7 ± 56.3 (n = 40); spiking `h16 dbc` 153.9 ± 45.8 (n = 5).

### 5.1 E1′ (gate): CartPole, the authors' config, no Adam, B = 1, seeds 20–39

| Run | Solved | $T_f$ | $T_s$ (median) | $T_{475}$ | Return | ≥ 475 | Collapsed | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BindsNET LIF (`--engine lif`) | **20/20** | 64.0 ± 8.4 [20] | 174.8 ± 18.5 (168) | 158.5 ± 8.9 [20] | 492.1 ± 12.7 | 18/20 | 0/20 | 6.26 |
| branching | 4/20 | 62.6 ± 16.3 [20] | 338.0 ± 70.2 [4] (inf) | 218.7 ± 71.8 [16] | 306.9 ± 214.7 | 9/20 | 7/20 | 6.41 |

- **The LIF arm passes the gate:** 20/20 solved and median $T_s$ 168 against CK's 169.5 ± 23.5, with $T_f$ 64.0 against 57.0. The authors' configuration reproduces CK's CartPole result.
- **The branching arm fails the gate:** 4/20 solved and median $T_s$ infinite. Every run reaches 500 by episode 98, and 16/20 reach a trailing-100 mean of 475.
  - 7/20 then collapse into the constant-action state.
  - Six of the seven collapse after 54–283 perfect episodes, with collapse onsets at episodes 129–345 (20-episode mean below 15). Four of those six had reached a trailing-100 mean of 475.
  - Seed 21 never reaches a trailing-100 mean of 475 before collapsing.
- **Branching against LIF** (Mann–Whitney, pre-registered):
  - $T_f$: p 0.46, so learning onset is the same.
  - Last-100 return: p $6.8 \times 10^{-4}$.
  - $T_s$ (not pre-registered): p $2.8 \times 10^{-8}$.
- **Audit seeds 0–4** (§8.2, not part of E1′) agree: LIF 5/5 solved; branching 1/5 solved, with 0/5 collapsed. Against E1′'s 7/20, 0/5 collapsed is not unusual (Fisher one-sided p about 0.16).

### 5.2 E3′ and E4′: LunarLander, the authors' config, seeds 0–9

Episodes are pooled over the 16 copies, and runs stop when solved with at least 3000 episodes, or at 10000. $T_f$ and $T_s$ are pooled episodes.

| Run | Seeds | Solved | $T_f$ | $T_s$ (median) | Return | Max trailing-100 | Episodes run | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CK 2020 | 10 runs | 10/10 | 383.8 ± 71.5 | 2575.2 ± 666.5 | – | – | ≥ 3000 | – |
| **E3′** Adam, branching | 0–9 | **7/10** | 390.4 ± 99.8 [10] | 6774.4 ± 1895.6 [7] (7808.5) | 173.1 ± 46.8 | 193.0 ± 15.4 | 3359–10000 | 6.34 |
| **E4′** SGD, branching | 0–9 | 0/10 | 806.0 ± 133.7 [10] | – (inf) | −0.9 ± 45.8 | 72.3 ± 26.2 | 10000 | 5.45 |
| LIF arm, Adam | 0–3 | 3/4 | 323.0 ± 43.7 [4] | 4085.7 ± 2978.7 [3] (5435.5) | 154.1 ± 59.1 | 204.4 ± 7.4 | 3000–10000 | 6.44 |
| LIF arm, seed 4, truncated | 4 | 0/1 at 5600 | 5015 | – | 5.7 | 15.1 | 5600 of 10000 | 5.93 |

- **E3′ passes its gate** (at least 6/10). Solved seeds and $T_s$: 0 (5326), 1 (9760), 2 (7892), 5 (7071), 6 (3359), 7 (6288) and 8 (7725). Seeds 3, 4 and 9 reached 10000 with last-100 returns of 83, 161 and 82.
  - **$T_f$ matches CK** (390.4 against 383.8).
  - **$T_s$ is 2.6× CK's** (6774 against 2575), with 3/10 unsolved against CK's 10/10.
  - Candidate causes, none tested: B = 20 rather than the authors' per-ms Adam (§3.1 item 6); LunarLander-v3 instead of v2; the branching escape noise (CartPole: §11.3).
- **E4′ against E3′** (Mann–Whitney):
  - $T_f$: p $1.8 \times 10^{-4}$;
  - $T_s$ (unsolved = 10001): p $2.2 \times 10^{-3}$;
  - last-100 return: p $3.3 \times 10^{-4}$.
  - Its η came from the pre-registered rule: 1.0–1.4e-3 mV for the critic and 1.5–2.0e-3 for the actor, per seed. Without Adam, and with that η, the rule learns to about 0 return and plateaus.
- **LIF arm** (reported only):
  - Seeds 0–3 solved 3/4, with $T_s$ 2635, 1386 and 8236; seed 1 reached 10000 unsolved.
  - Seed 4 learned slowly (first episode at 200 only at 5015). It was stopped at 22:05 EDT at 5600 of 10000 episodes, because it would have needed about 8 more hours. Its partial and checkpoint stay in `experiments/results/`, so rerunning the same command resumes it. It is not in `--summary`, which reads only completed runs.
- **Interruption (all 25 runs).**
  - At about 15:15 EDT a coordinating agent mistook these processes for the stopped pre-audit runs and killed them.
  - They were resumed at 15:25 from their checkpoints of 15:07–15:11. Those were at 1300–3050 pooled episodes; the counts are in the scratchpad (`ckfix/ckpt_backup/`).
  - **Exact:** W, the Adam moments, the pooled returns and the rng state, all checked on seed 0. E4′'s recalibrated η was identical (checked on seeds 0 and 5).
  - **Not exact:**
    - every copy started a fresh episode with cleared traces;
    - episodes that ended between a run's last checkpoint and the kill (fewer than 50 per run, 4–8 minutes of run time) were lost;
    - the episodes in progress at the checkpoint were dropped.
  - Each result records `resumes` = 1. The runs are therefore not bit-reproducible from the command line, but the protocol is otherwise unchanged.

### 5.3 Pre-fix results (pre-audit code, `--legacy`)

These runs used the pre-audit choices of §3.3. They are what was run, and they are kept for the record. The E1 failure was a reproduction error (§8), not a property of CK's rule.

#### E1 (the original gate: CK's CartPole, no Adam) and the true-LIF control

| Run | Seeds | Solved | $T_f$ | $T_s$ | Return | ms/frame |
| --- | --- | --- | --- | --- | --- | --- |
| E1, branching | 0–19 | 0/20 | 214.0 ± 121.0 [6] | – | 15.4 ± 16.4 | 6.95 |
| true LIF (`lif`, CK Eq. 1, Table 1 verbatim) | 0–4 | 0/5 | 229.0 ± 137.0 [2] | – | 18.4 ± 18.0 | 6.14 |
| true LIF (`lif-snm`, SuperNeuroMAT leak) | 0–4 | 0/5 | – [0] | – | 19.0 ± 14.0 | 6.47 |

**E1 fails the gate** (0/20 solved, median $T_s$ infinite), and the pre-audit LIF control fails the same way. That control used the paper's Eq. 1 scale with the same pre-audit η, trace scale and loop, so the failure is not specific to branching mode. The failure looks like this:
- Six seeds reach 500 once, and most end on a constant action (return 9.4).
- The policy entropy falls to 0.00–0.05 nats out of 0.69.
- At an episode end, $\delta \approx -V \approx -0.2$ for 2 ms and $q \approx \tau_q \lvert A_k - s_k \rvert z \approx 800$. With Table 1's actor η, one terminal step moves the actor weights by 25–35%.
- The policy therefore flips between the two actions from episode to episode until the softmax saturates. Once saturated, $A_k - s_k \approx 0$ for both actions and learning stops: the absorbing state.

#### E2 (Adam, per ms) and the block-form gate

| Run | Seeds | Solved | $T_f$ | $T_s$ | $T_{475}$ | Return | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2, branching + Adam (B = 1) | 0–19 | 1/20 | 123.0 ± 30.2 [20] | 322 [1] | 258.6 ± 33.4 [13] | 377.8 ± 83.2 | 8.36 |
| branching + Adam, B = 20 (block-Adam gate) | 0–9 | 0/10 | 134.5 ± 38.3 [10] | – | 248.6 ± 33.4 [9] | 435.5 ± 42.9 | 0.76 |

- **E2 against E1:**
  - On $T_s$ (unsolved counted as 401) there is no difference, MW p 0.34, because both groups are almost all censored.
  - On return, MW p $6.7 \times 10^{-8}$.
  - All 20 E2 seeds reach 500, and none collapses.
  - E2 is about 2× slower than CK on $T_f$ (123 against 57), and only 1/20 meets CK's strict $T_s$.
- **Block-Adam gate:** B = 20 against E2 on seeds 0–9 (369.7 ± 83.9) gives MW p 0.14 with no collapses, so it passes. The rejected first form scored 236.1 on the same seeds with 3/10 collapsed (MW p 0.064, session scratchpad, §3 item 6).

#### Post hoc (not pre-registered)

| Run | Seeds | Solved | $T_f$ | $T_s$ | $T_{475}$ | Return | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- |
| true LIF + Adam (`--adam --engine lif`) | 0–9 | 2/10 | 131.7 ± 25.8 [10] | 316.5 ± 19.5 [2] | 295.4 ± 45.8 [8] | 464.8 ± 31.4 | 7.59 |
| SGD, η from the E4 rule (`--sgd`) | 0–9 | 0/10 | 144.0 ± 48.3 [10] | – | – [0] | 88.7 ± 130.8 | 5.72 |
| SGD, B = 20 | 0–9 | 0/10 | 137.3 ± 32.6 [10] | – | – [0] | 59.5 ± 87.2 | 0.77 |
| E1 with init rate 0.1, V(init) ≈ 0 (`--rate0 0.1`) | 0–4 | 0/5 | 288.2 ± 60.2 [4] | – | – | 17.6 ± 11.9 | 7.04 |
| same, true LIF | 0–4 | 0/5 | 231.0 [1] | – | – | 20.9 ± 14.2 | 7.14 |

- **Branching against CK's neuron under Adam (seeds 0–9):** the learning onset is the same ($T_f$ 126.0 against 131.7, MW p 0.62). The final return is lower on the branching engine (369.7 ± 83.9 against 464.8 ± 31.4, MW p 0.006). Branching mode carries independent escape noise on every post neuron, while the LIF is deterministic given its inputs.
- **SGD with the Adam-equivalent η:**
  - The calibrated η is 3.1e-3 for the critic (Table 1: 2.5e-3) and 1.1e-3 for the actor (Table 1: 1e-2, so about 9× smaller).
  - With it every seed reaches 500 once, so the converted Table 1 actor step accounts for much of the E1 failure.
  - Without Adam, 6/10 seeds at B = 1 and 7/10 at B = 20 still end in the absorbing state.
  - B = 1 against B = 20 gives MW p 0.62: the 20 ms weight freeze does not change the SGD outcome detectably.
- **Init rate 0.1:** this matches CK's Fig. 3 rates and removes the initial terminal TD error, but it does not rescue E1.

#### First E3 and E4 (LunarLander): stopped at the wrong settings

- **Settings:** started 2026-10-23 at 11:13 EDT. E3 was `--task LunarLander --seed S` with the pre-audit preset: Adam with block moments, 16 asynchronous copies, entropy and decay. E4 was the same with `--sgd`. Seeds 0–9 each.
- **Stopped** at the user's direction after the audit (§8), because they ran at the wrong settings:
  - the Adam step was 100× too small (§8.1 row 10);
  - episodes were counted per copy (row 11);
  - the entropy term was 20× too weak, and a weight decay was applied that the authors never apply (row 12);
  - they used the pre-audit loop order.
- **State at the stop** (copy-mean returns, after about 1.5 h): E3 had 450–500 episodes per copy, with last-50 means from −171.0 to −161.6. E4 had 550–600, with last-50 means from −195.2 to −183.3.
- **E4's calibrated η** (pre-audit units, ÷1300) was 1.1–1.3e-4 for the critic and 0.92–1.02e-5 for the actor.
- **Where the partials went:** `experiments/results/tdstdp_LunarLander[_sgd]_s<seed>.partial` were moved to the session scratchpad (`stopped_e3e4/`). They are not in `--summary`.

### 5.4 Cost

- CartPole: 6.3–6.4 ms per frame for per-ms learning (B = 1) on the authors' path, under load. The pre-audit path took 7–8 ms at B = 1 and 0.76 ms at B = 20.
- LunarLander (16 copies, 272 inputs, 512 posts): 2.9 ms per copy-frame at B = 20 and 45 ms at B = 1 on a lightly loaded machine; BindsNET LIF at B = 20 costs 2.7 ms.
  - Under the full load of the E3′/E4′ runs: 5.5–6.4 ms per copy-frame.
  - Wall time per run: 3.0–7.5 h (E3′) and 7.7–8.3 h (E4′, 10000 episodes each).
- Speed-ups: see §11. CartPole B = 20 without Adam is 0.84–0.93 ms per frame, 7× faster; `--fast` makes LunarLander 1.8× faster.
- SuperNeuroMAT (`--engine snm`) costs 98 ms per frame against 10 ms for numpy, with the same raster.

## 6. Reading

- **CK's CartPole result reproduces when the authors' code is followed.** With their η, trace scale, loop order, truncation bootstrap and deterministic LIF, and no Adam, the run gives 20/20 solved, $T_f$ 64.0 against 57.0, and $T_s$ median 168 against 169.5.
- **The original E1 failure was a reproduction error, not a property of the rule.** Its causes are §8.1 rows 1–5, together with the pre-audit unit reading (÷1300).
  - The earlier reading that "Adam's per-synapse normalisation is load-bearing" does not hold. On the authors' path, CartPole needs no Adam.
  - What the pre-audit runs show is that Adam rescued a mis-scaled step.
- **The branching engine learns as fast as the LIF but holds the solution less well.**
  - $T_f$ is the same (p 0.46), and 16/20 runs reach a trailing-100 mean of 475.
  - However, only 4/20 meet CK's strict 100 × 500 criterion, and 7/20 collapse into the constant-action state after a period of near-perfect play.
  - The difference from the LIF is the neuron. Branching mode has independent escape noise on every post neuron each ms, while the BindsNET LIF is deterministic given its inputs.
  - **Evidence (§11.3):** the escape noise is the cause. The flow limit removes only the noise and solves 20/20.
  - **Mechanism (speculation, untested):** the noise keeps $(A - p) z$ and δ fluctuating after the task is solved, and with $\alpha_a$ = 50 a run of same-sign actor updates can saturate the softmax. Once it saturates, $A - p \approx 0$ and learning stops, as in the pre-audit E1.
  - **Candidate fixes** for step 2: a smaller $\alpha_a$ or actor η, more actor neurons per action (to average the escape noise), and an entropy term. Each trades against CK fidelity, and none has been run.
- **LunarLander learns on the branching engine with Adam, but more slowly than CK.** $T_f$ matches (390 against 384), and 7/10 runs solve, but at a $T_s$ 2.6× CK's (§5.2).
  - **Adam is load-bearing here.** Without it, with η set by the Adam-equivalent rule, no run solves and returns plateau near 0 (all three Mann–Whitney p ≤ 0.002).
  - This is the opposite of CartPole, where the authors' configuration needs no Adam.
- **For the plan:** step 2 (removing Adam) has two parts:
  - the branching-noise stability problem above, on CartPole;
  - a replacement for Adam's per-synapse step normalisation on LunarLander. A crossbar-native candidate would be a per-module, bounded or normalised step on the TD error. This is a Proposal, untested.
- **Tooling for step 2:** the no-Adam block sum (B = 20, 7× faster, gate passed) and `--fast` (1.8× on 16 copies, identical returns) can be used for new runs (§11).

## 7. Commands

```bash
python -m mapstdp.tdstdp                                                  # self-check
python -m experiments.tdstdp --task CartPole --seed S --engine lif        # E1′ LIF arm, S = 20..39
python -m experiments.tdstdp --task CartPole --seed S                     # E1′ branching arm, S = 20..39
python -m experiments.tdstdp --task LunarLander --seed S                  # E3′, S = 0..9
python -m experiments.tdstdp --task LunarLander --seed S --sgd            # E4′, S = 0..9
python -m experiments.tdstdp --task LunarLander --seed S --engine lif     # E3′ LIF arm, S = 0..4
python -m experiments.tdstdp --task CartPole --seed S [--engine lif] [--block 20]    # S1 (§10), S = 40..59
python -m experiments.tdstdp --task LunarLander --seed S --episodes 400 [--fast]     # S2 (§10), S = 100..104
python -m experiments.tdstdp --task CartPole --seed S --engine flow                 # S3 (§10), S = 60..79
python -m experiments.tdstdp --summary
python -m experiments.report run tdstdp-baseline --workers 40   # the reported baseline (§12) -> reports/tdstdp-baseline/
# pre-audit runs (§5.3), reproducible bit for bit
python -m experiments.tdstdp --legacy --task CartPole --seed S                   # E1, S = 0..19
python -m experiments.tdstdp --legacy --task CartPole --seed S --engine lif      # true-LIF control (also lif-snm), S = 0..4
python -m experiments.tdstdp --legacy --task CartPole --seed S --adam            # E2, S = 0..19
python -m experiments.tdstdp --legacy --task CartPole --seed S --adam --block 20 # block-Adam gate, S = 0..9
python -m experiments.tdstdp --legacy --task CartPole --seed S --sgd [--block 20]          # post hoc, S = 0..9
python -m experiments.tdstdp --legacy --task CartPole --seed S --rate0 0.1 [--engine lif]  # post hoc, S = 0..4
python -m experiments.tdstdp --legacy --task CartPole --seed S --adam --engine lif         # post hoc, S = 0..9
```

The E1′, E3′, E4′ and S1–S3 runs were launched locally from the session scratchpad scripts `ckfix/launch.sh` and `ckfix/sched.sh`, which ran these commands with at most 40 workers. Their logs are in `ckfix/logs/`.

## 8. Comparison with the authors' code (post hoc)

- **Source:** `reference/td_stdp` (commit `a5cb7f4`, which pins BindsNET 0.2.7), read against the BindsNET 0.2.7 tag. Their code was not run.
- **Scope:** every test below is a post hoc diagnostic, not pre-registered. Each one is 5 seeds × 400 CartPole episodes, with no Adam and B = 1.
- **Where things are:** scripts and outputs are in the session scratchpad (`audit/`). The audit itself changed no repo code; the fixes were implemented afterwards as the default path (§2, §3.1).

### 8.1 Differences

| # | Item | Authors' code | Ours (pre-audit) | Paper | Impact |
| --- | --- | --- | --- | --- | --- |
| 1 | Trace scale | `bindsnet_add.py:213`: $P \mathrel{+}= \nu x$, with η folded into the pre trace. `:224`: $z \mathrel{+}= P X/\tau_z$. `:185`: $q \mathrel{+}= (A-p) z/\tau_q$ | `tdstdp.py:143-148, 219-220`: $z \mathrel{+}= PX$ and $q \mathrel{+}= cz$, with η applied to the update | Eqs. 72 and 77 have no $1/\tau$ | **High.** For a given η the code's step is $1/\tau_z$ (critic) and $1/(\tau_z \tau_q)$ (actor) of the equations. The ratios check exactly on identical spikes. |
| 2 | Weight units | BindsNET `nodes.py:504-519`: a spike adds w mV with no $R/\tau$. The neuron spikes at $v \ge -52$, resets to −65, has τ = 100 and refractory period 0 (`main.py:127`) | MV = 1300 from Eqs. 1 and 44 (`experiments/tdstdp.py:37`) | Eq. 1 implies $w/\tau$ | **High.** The conversion is ÷13, not ÷1300. Their U(0, 0.1) init fires 0.141 per ms on a transcription of their LIF; U(0, 0.1/13) on the branching step fires 0.156. |
| 3 | η | `config_cp.ini`: critic 0.005, actor 0.01 | Table 1: 2.5e-3 and 1e-2 | Table 1 | **High,** combined with rows 1 and 2. The authors' step per unit of our z or q is 1.9e-5 for the critic and 9.6e-7 for the actor, in branching units. Ours is 1.9e-6 and 7.7e-6, so the critic is 10× too small and the actor 8× too large. The E4 calibration gave an actor of 8.5e-7. |
| 4 | Loop order | `bindsnet_add.py:640-665`: present $s_k$, then sample $a_k$ from that response. The next window carries $(a_k, p_k, r_k)$, with $p$ fixed at the sampling probability | `experiments/tdstdp.py:173-176, 210`: $a_k$ is sampled from the response to $o_{k-1}$, a 20 ms observation delay. The feedback is $A - s(t)$ | not specified | **High.** Removing the delay removes the collapses (§8.2). |
| 5 | Truncation | `bindsnet_add.py:527`: $V'$ is bootstrapped at the 500-step limit | $V' := 0$ on termination or truncation (`:176, 203`). the pre-audit record's "as CK do" was wrong | "the last 2 ms of an episode" | **Medium–high.** Every perfect episode ends on $\delta \approx -1$. |
| 6 | $\alpha_a$ | `actor_m` = 50 | 25 | Table 1: 25 | Medium |
| 7 | Neuron | Deterministic LIF | Escape noise | LIF | **Medium.** It decides the strict $T_s$ (§8.2). |
| 8 | Encoder bounds | Velocities ±3.2, angle ±π/12 (`main.py:93`) | ±2.0, ±0.21, ±2.5 (`tasks.py:99`) | not given | Low |
| 9 | Init, warm-up | U(0, 0.1) mV gives 0.14 per ms. Six presentations of $s_0$ | Rate 0.2. Five presentations | not given; 100 ms | Low |
| 10 | Adam (LunarLander) | `bindsnet_add.py:204` steps ν mV per ms, which is 9.6e-6 and 4.8e-6 in branching units | ÷1300, so 100× smaller | Adam | **High for E3.** |
| 11 | Episodes (LunarLander) | Pooled over the 16 copies (`main.py:233, 263`), so $T_s$ 2575 is about 161 episodes per copy | 3000 per copy | "batch size 16" | **High** for E3's cost and comparability |
| 12 | Entropy, decay (LunarLander) | Entropy enters as $c_e g z/\tau_z$ against $\delta q/(\tau_z\tau_q)$. `weight_reg` is never passed to the rule (`main.py:133-156`, `bindsnet_add.py:129`), so there is no decay | Entropy is 20× weaker than theirs, relative to the policy term. Decay is on | $c_w$ = 1e-8 | Medium |

Every other part matches. These are:

- the 1 ms input delay (`network.py:208-236`);
- δ paired with the trace of the previous ms;
- the reward handling ($0.02 r/20$ with $e^{-1/2\tau}$);
- the value $V = -0.2 + 2 \bar{\rho}$;
- Bernoulli inputs;
- an action resampled every step (`freeze_action` = `env_dt`);
- signed, unclipped weights;
- batch-mean reduction (BindsNET 0.2.7 `LearningRule`);
- trace resets at each episode.

### 8.2 Tests

The variants are cumulative. In the "scale" variant, $\eta_c$ = 0.025 and $\eta_a$ = 1.25e-3 in our MV units. These are the code's η with rows 1–3 applied.

| Variant | Solved | $T_f$ | $T_s$ | $T_{475}$ | Return | Collapsed |
| --- | --- | --- | --- | --- | --- | --- |
| E1, seeds 0–4 | 0/5 | 363.0 ± 22.0 [2] | – | – | 33.3 ± 25.4 | 2/5 |
| authors' code except E1's η | 0/5 | – | – | – | 9.4 ± 0.1 | 5/5 |
| scale | 0/5 | 72.4 ± 9.9 [5] | – | – | 68.7 ± 118.6 | 4/5 |
| scale + truncation bootstrap | 1/5 | 72.4 ± 9.9 [5] | 276 [1] | 302.0 ± 40.4 [3] | 253.3 ± 210.2 | 2/5 |
| scale + loop order | 0/5 | 68.2 ± 8.0 [5] | – | 373 [1] | 310.1 ± 92.6 | 0/5 |
| scale + both | 2/5 | 68.2 ± 8.0 [5] | 291.5 ± 33.5 [2] | 233.2 ± 49.2 [4] | 389.4 ± 190.6 | 1/5 |
| full: + $\alpha_a$ 50, bounds, init, warm-up | 1/5 | 73.4 ± 11.4 [5] | 389 [1] | 216.2 ± 40.8 [5] | 490.4 ± 9.8 | 0/5 |
| full + BindsNET LIF | **5/5** | 65.0 ± 7.4 [5] | 213.8 ± 74.5, median 185 | 163.4 ± 11.0 [5] | 500.0 ± 0.0 | 0/5 |

- **CK is reproduced** with the code's η and trace scale, their loop order, truncation bootstrap and their deterministic LIF: 5/5 solved, $T_f$ 65 against 57, and $T_s$ median 185 against 169.5.
- **The scale is necessary.** It alone moves $T_f$ to CK's level, and without it everything else collapses.
- **The loop order and the truncation bootstrap stop the collapse** that follows the first perfect episodes.
- **The branching engine** with all fixes reaches 490 but rarely holds 100 consecutive 500s.
- **The E1 failure was a reproduction error, not a property of CK's rule.** These are rows 1–5. §3.1 states the corrected facts, and §3.3 the pre-audit choices they replace.
- **The first E3 and E4 were affected** by rows 10–12. They were stopped (§5.3) and rerun as E3′ and E4′ (§5.2).

## 9. Authors'-code path: pre-registration of E1′, E3′ and E4′

Written on 2026-10-23 before any E1′, E3′ or E4′ run. The sanity runs used to test the code (CartPole seeds 0–1, both engines; a 60-tick LunarLander timing; a calibration smoke test) are in the session scratchpad (`ckfix/`).

**Code.** `python -m experiments.tdstdp` now defaults to the authors' code (result token `ck`; `--legacy` reproduces E1/E2 bit for bit). `PRESETS` are `config_cp.ini` and `config_ll.ini`. `CKRunner` transcribes `main.py`, `Snn_actor`, `Reward_fn` and `batch_envs`, and `tdstdp.block_learn_ck` transcribes `FBTDSTDP`. Weights convert by ÷13, and `--engine lif` is BindsNET's LIF with W in mV.

**E1′ (gate): CartPole, the authors' config, no Adam, B = 1, 400 episodes, seeds 20–39.**

- Commands: `--task CartPole --seed S --engine lif` and `--task CartPole --seed S`.
- **Gate (LIF):** at least 18/20 solved (CK's strict criterion, 100 consecutive episodes at 500), and median $T_s$ at most 340 (unsolved counted as infinite).
- **Branching:** the same gate is reported. Also reported: the count of runs whose trailing-100 mean reaches 475, and the count whose last-100 return is at least 475.
- **Comparison:** two-sided Mann–Whitney, branching against LIF, on $T_f$ (never reached = 401) and on the last-100 return.

**E3′: LunarLander, the authors' config, seeds 0–9, branching engine.**

- Adam per `config_ll.ini`: step ν per ms in mV (÷13 in branching units), β = (0.995, 0.99995), ε = 1e-9 on the code's ν-scaled gradient.
- 16 copies, with the update averaged over all 16.
- Entropy $c_e$ = 1e-4 at the code's scale; no weight decay (the code never applies it).
- Episodes are pooled over copies. The run stops once solved with at least 3000 pooled episodes, or at 10000.
- **Deviations** fixed here:
  - LunarLander-v3 instead of v2.
  - **B = 20** (W frozen for one env step, block Adam with per-ms-matched moments, §3 item 6). B = 1 costs 45 ms per copy-frame against 2.9 ms, which would be days per seed.
- Command: `--task LunarLander --seed S`.
- **Gate:** at least 6/10 solved, meaning a trailing-100 pooled mean of at least 200 within the run. $T_f$ and $T_s$ (pooled) are reported against CK's 383.8 ± 71.5 and 2575.2 ± 666.5.
- **LIF arm, if the budget allows:** the same with `--engine lif`, seeds 0–4, reported only.

**E4′: as E3′, without Adam (`--sgd`), seeds 0–9.**

- **η rule (fixed now):** the E4 rule on the authors' loop (`calibrate_ck`). Per population (critic, actor), $\eta = \nu_{Adam} / \mathrm{RMS}(g)$, where g is the per-ms update averaged over the 16 copies at the code's scale.
- The RMS is measured with the initial W frozen, per ms (B = 1), over the 200 ticks that follow the first 6 (no copy has stepped yet), on its own rng stream (seed, 1303).
- An RMS-sized update then moves W by the step per-ms Adam takes at init. Training then runs at B = 20, like E3′.
- Reported against E3′: solved count, and Mann–Whitney on $T_f$, $T_s$ (censored at 10001) and the last-100 pooled return.

**Run conditions.** All runs are local, with `OMP_NUM_THREADS=1` and at most 40 workers. LunarLander checkpoints every 50 pooled episodes (`*.ckpt.npz`). A resumed run restarts every copy's episode and is not bit-identical; any resume is recorded in the result (`resumes`).

## 10. Speed-ups and the flow limit: pre-registration

Written on 2026-10-23 at about 16:10 EDT, before any of these runs. The resumed LunarLander runs (§5.2) are not affected: each new feature sits behind a flag whose default leaves the existing path bit-identical.

**S1: block sum for the no-Adam rule (gate).**

- **Runs:** CartPole, the authors' config, no Adam, 400 episodes, fresh seeds 40–59. Each seed runs at B = 1 and at B = 20 (`--block 20`), on both `--engine lif` and the branching engine: 80 runs.
- **Mechanism:** at B = 20 the per-ms updates of one env step are summed in closed form with W frozen for 20 ms (`block_learn_ck`, an existing code path).
- **Gate,** per engine, B = 20 against B = 1 (paired on seeds). B = 20 passes if:
  - its solved count is within 2 of B = 1's (CK's strict criterion);
  - two-sided Mann–Whitney p > 0.05 on $T_s$ (unsolved = 401);
  - two-sided Mann–Whitney p > 0.05 on the last-100 return.
- **Also reported:** the ms/frame speed-up.
- **Use:** if both engines pass, later LunarLander reruns (not the resumed ones) may use the block sum without Adam.

**S2: faster trace algebra for many copies (`--fast`).**

- **Profile:** at B = 20 the 16 LunarLander copies already share one batched network step, and env stepping is about 1% of a tick. Of a 108 ms tick under load, 89 ms is `zsum` and `block_learn_ck`.
- **Change:** `--fast` sums the update over copies inside one matrix product, and computes only the per-copy end traces per copy.
- **Rng order is unchanged.** The result equals the default up to floating-point summation order, so it is not bit-identical, and trajectories diverge after the first rounding difference.
- **Equivalence checks:**
  - one block's mean update against the default to relative error at most $10^{-12}$ (self-check);
  - LunarLander, the authors' config, 400 pooled episodes, seeds 100–104, `--fast` against the default: two-sided Mann–Whitney p > 0.05 on the last-100 pooled return and on the mean return over all episodes.
- **Also reported:** ms per copy-frame before and after.

**S3: the flow limit of the branching engine (`--engine flow`, exploratory).**

- **Mechanism:** the escape noise is replaced by its expectation.
  - Inputs are their rates f, the pre trace is the filter of f, and a post's spike is its probability $p_j = \mathrm{clip}((W f)_j)$.
  - The z increment is $E[X_j(t) P_i(t)] = p_j \bar{P}_i + W_{ji} a_p f_i (1 - f_i)$, where the second term is the input autocorrelation through $x(t-1)$ (exact in the linear regime).
  - Rates, V, δ and p follow deterministically. The action is still sampled.
- **Self-check:** with W, f, δ and the feedback fixed, the Monte Carlo average of the sampled rule's update must match the flow update with cosine at least 0.99.
- **Runs:** CartPole, the authors' config, no Adam, B = 1, 400 episodes, seeds 60–79.
- **Reported:** solved count, $T_s$, $T_f$, last-100 return and ms/frame, against E1′ branching and E1′ LIF (seeds 20–39). Two-sided Mann–Whitney, flow against branching, on $T_s$ (unsolved = 401) and on the last-100 return.
- **Reading:** a diagnostic only, with no gate. If flow solves like the LIF, escape noise is what fails branching's strict $T_s$.

## 11. Speed-ups and the flow limit: results

All runs are from §10's pre-registration and are in `--summary`. Columns are as in §5. `--summary` groups by configuration, so its `tdstdp_CartPole_ck` and `_ck_lif` rows pool E1′ (seeds 20–39) with S1's B = 1 arms (seeds 40–59). The tables here split them by seed range.

### 11.1 S1: block sum without Adam (CartPole, seeds 40–59)

| Engine | B | Solved | $T_f$ | $T_s$ (median) | $T_{475}$ count | Return | ≥ 475 | Collapsed | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BindsNET LIF | 1 | 19/20 | 65.0 ± 11.6 [19] | 184.5 ± 27.6 (178.5) | 19 | 470.8 ± 106.3 | 18 | 1 | 6.25 |
| BindsNET LIF | 20 | 20/20 | 69.8 ± 12.3 [20] | 203.5 ± 44.0 (185) | 20 | 492.7 ± 11.0 | 19 | 0 | 0.93 |
| branching | 1 | 7/20 | 72.0 ± 32.7 [20] | 288.4 ± 77.6 [7] (inf) | 19 | 396.2 ± 157.4 | 9 | 3 | 5.92 |
| branching | 20 | 6/20 | 66.0 ± 14.1 [20] | 225.8 ± 83.8 [6] (inf) | 15 | 383.6 ± 156.3 | 8 | 3 | 0.84 |

- **The gate passes on both engines.**
  - LIF: solved 20 against 19; Mann–Whitney p 0.27 on $T_s$ and 0.54 on return.
  - Branching: solved 6 against 7; p 0.99 on $T_s$ and 0.39 on return.
- **Speed-up:** 6.7× (LIF) and 7.1× (branching) in ms/frame.
- **Replication:** the B = 1 arms replicate E1′ on fresh seeds. LIF solves 19/20 against 20/20. Branching solves 7/20 against 4/20 and again collapses in some runs (3/20).

### 11.2 S2: `--fast` (LunarLander, the authors' config, 400 pooled episodes, seeds 100–104)

- **Equivalence:** the pooled return sequences of `--fast` and the default are **identical** on all five seeds, all 400 episodes. The final W agree to the last bit or within one ulp. The pre-registered Mann–Whitney tests are therefore trivially p = 1.0.
- **Why so close:** over these runs the summation-order differences, about $10^{-16}$ relative, never flipped a spike or an action draw.
- **Speed:**
  - Same machine and load, 100 ticks after a 10-tick warm-up: 8.26 ms per copy-frame (default) against 4.48 (`--fast`), a 1.8× speed-up.
  - The runs' own wall clock (noisier under load): 9.0 against 6.2 ms per copy-frame.
- **What remains:** about 1% of a tick is env stepping. The rest is the per-copy end traces (16 × 512 × 272 per trace), which every copy needs.

### 11.3 S3: the flow limit (CartPole, seeds 60–79) against E1′

| Run | Seeds | Solved | $T_f$ | $T_s$ (median) | Return | ≥ 475 | Collapsed | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flow (`--engine flow`) | 60–79 | **20/20** | 58.6 ± 26.8 [20] | 212.7 ± 51.7 (219) | 467.7 ± 107.8 | 18 | 1 | 6.79 |
| branching (E1′) | 20–39 | 4/20 | 62.6 ± 16.3 [20] | 338.0 ± 70.2 [4] (inf) | 306.9 ± 214.7 | 9 | 7 | 6.41 |
| BindsNET LIF (E1′) | 20–39 | 20/20 | 64.0 ± 8.4 [20] | 174.8 ± 18.5 (168) | 492.1 ± 12.7 | 18 | 0 | 6.26 |

- **Flow against branching** (pre-registered): Mann–Whitney p $7.4 \times 10^{-8}$ on $T_s$ (unsolved = 401) and $5.4 \times 10^{-4}$ on the last-100 return. $T_f$ is not different (p 0.13).
- **Flow against LIF** (not pre-registered): p 0.034 on $T_s$, so flow is somewhat slower to the strict criterion. On return, p 0.52.
- **Reading:** removing the escape noise, and nothing else, turns 4/20 solved into 20/20. Escape noise is what fails branching's strict $T_s$ and causes most of its collapses.
  - One flow run (seed 75) still collapsed after solving, so the absorbing state is reachable without neuron noise too: the action sampling remains stochastic.

## 12. Reported baseline: `reports/tdstdp-baseline/`

- **The reported TD-STDP baseline is the study in `reports/tdstdp-baseline/`.**
  - Its design and gates were pre-registered in its `README.md` before launch (2026-10-23, 22:37 EDT).
  - Its numbers are in `summary.md` and `summary.csv`, built by `python -m experiments.report summarize tdstdp-baseline` from the raw runs there only.
  - The E1′, E3′, E4′ and S1–S3 numbers above are development results. They stay in `experiments/results/`, and the paper does not cite them.
- **Design:**
  - The authors'-code path with B = 20 and `--fast` (both passed in §11), on fresh seeds (CartPole 1000–1019, LunarLander 1000–1009).
  - CartPole crosses {no Adam (paper config), Adam} with {`lif`, branching}.
  - LunarLander crosses {Adam (paper config), no Adam with the E4′ η rule} with {`lif`, branching}.
  - The Mann–Whitney reference is the paper config on `lif` for each task.
  - Gates: CartPole no-Adam `lif` at least 18/20 solved, and LunarLander Adam `lif` at least 6/10.
- **Code state:** one, recorded in its `manifest.json` (HEAD `675e49e` plus `code.patch`: `--out`, `tdstdp.group_stats`, `experiments/report.py`).
- **Run:** 120 runs, local, 40 workers, 2026-10-23 22:38 to 2026-10-24 05:57 EDT (7.3 h), 0 failures, 0 resumes.
- **Gate outcomes** (from `summary.md`):
  - G1 (CartPole no-Adam `lif`, at least 18/20): **pass**, 20/20.
  - G2 (LunarLander Adam `lif`, at least 6/10): **pass**, 8/10.
- **Note on the CartPole Adam arm** (post hoc, a reading of the declared config, not a code fault):
  - It solves 0/20 on both engines, with returns of 11–15.
  - The authors' Adam steps ν mV per ms, so with CartPole's SGD η (0.005 and 0.01 mV) the step is 40–160× LunarLander's.
  - A scratchpad check at B = 1 without `--fast` (seeds 1000–1001, 60 episodes) gives the same constant-action returns of 9–10 from the first episodes. B = 20 and `--fast` are therefore not the cause.
  - **Dropped (2026-10-24, user's decision):** CK report no CartPole Adam configuration and CartPole needs no Adam, so both CartPole Adam groups are left out of the reported baseline. Their runs stay in `runs/` (study README, amendment).

## 13. Diagnostic: the Map-STDP walk on the TD-STDP LIF neuron (frozen weights)

Run 2026-10-24, scratch only (no closed loop, nothing in `reports/`). Pre-registration, scripts and text outputs are archived in `docs/iterations/013-lifdiag/` (`PREREG.md`, two amendments logged before the affected data was read). The scripts' paths point at the session scratchpad, so they need editing to rerun. The plain-language reading is in `docs/neuron-variants.md` §5.

- **Question.** Does the derivation's linear-response walk (§2.1, K1–K9) hold for model v4's network, with recurrence, simulated on the TD-STDP LIF (τ 100 ms, threshold 13 mV above rest, reset, 1 ms steps, W in mV)?
- **Setup.**
  - Recurrent weights are χ·T in mV, with χ = 10.4 mV for a nominal offspring mean of 0.8. Stimulus: Poisson input at 4 mV per spike and rate R·v, with R = 21.5 (CartPole) and 41.9 (LunarLander), giving rates of 0.12–0.22/ms.
  - Gains come from open-loop replay with common random numbers.
  - Four networks: CartPole and LunarLander, at initialisation and with trained flow-v4 weights.
  - Frames: 200 ms reset and 20 ms persistent. The branching engine is the control.
- **Results on initial weights** (reset frames; ranges over CartPole and LunarLander; ✓ = pass at the pre-registered threshold):

| Operating point | TV vs K3 (≤ 0.05) | κ vs $J^\Lambda$ corr / slope | ρ (gain CV) | rule cos (≥ 0.9) | κΦ / (4b) cos (≥ 0.9) | settle, cold / switch (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| branching | 0.02 ✓ | 0.98 / 0.87 ✓ | 0.80 (0.03) ✓ | 0.99 ✓ | 0.98 / 0.73–0.79 | 6 / 1 ✓ |
| LIF, deterministic | 0.20–0.28 | 0.57–0.76 / 0.55–0.70 | 0.65 (0.31) ✓ | 0.88–0.90 | 0.64–0.80 / 0.55–0.73 | 82–118 / 1 |
| LIF, σ = 1 mV/√ms | 0.04–0.08 | 0.66–0.84 / 0.54–0.72 | 0.65 ✓ | 0.96 ✓ | 0.70–0.83 | 20–21 / 1 |
| LIF, σ = 2 | 0.03–0.07 | 0.64–0.82 / 0.51–0.71 | 0.65 ✓ | 0.95 ✓ | 0.68–0.81 | 15–16 / 1–2 ✓ |
| LIF, σ = 4 | 0.04–0.09 | 0.65–0.81 / 0.50–0.68 | 0.60 ✓ | 0.94–0.95 ✓ | 0.67–0.80 | 4–5 / 2–4 ✓ |
| LIF, escape Δ = 2 mV | 0.13–0.14 | 0.66–0.77 / 0.58–0.72 | 0.64 ✓ | 0.92 ✓ | 0.64–0.67 | 16–18 / 1 ✓ |
| LIF, χ = 6.5 mV | 0.35–0.37 | 0.49–0.79 | 0.47 ✓ | 0.68–0.82 | 0.59–0.86 | never |
| LIF, χ = 12.35 mV | 0.24 | 0.60–0.67 / 0.64 | 0.68 ✓ | 0.85 | 0.63–0.70 | 34–37 / 1 |

- **No LIF operating point passes every check.** The best, σ = 2–4 mV/√ms, passes subcriticality, the rule direction and settling, and passes the flow check on CartPole only. It fails the pairing and task-credit checks on every network.
- **Mechanism.**
  - The cascade is weaker than nominal (about 0.64 offspring per spike, against 0.8), because of leak and reset.
  - The lag-1 pairing count carries one factor per postsynaptic neuron (its window efficiency) and one per presynaptic neuron (regularity). Together they explain 98–99.5% of its variance with noise, but make it non-proportional.
- **Trained weights** (flow-v4 weights, nearly sealed modules).
  - Every engine fails the rule and credit checks, the branching control included: rule cos 0.44–0.89, κΦ 0.04–0.47. The split-half reliability is 0.89–0.99, so this is bias, not noise.
  - Cause: on weak cross-module synapses, κ picks up whole-module co-fluctuation that does not scale with the synapse, and $M^\ast$ amplifies it.
  - Appendix A samples pairings in proportion to J, so it cannot detect this. Restricted to within-module synapses, the rule cos is 0.87–0.93.
- **Reading.**
  - The deterministic LIF does not support the walk.
  - With input noise of about 2 mV/√ms, the structural (map) term is usable, with χ raised to about 13 mV or the K3 walk taken with measured gains. The task term is then empirical-only, with credit 20–35% off.
  - The trained-weight bias on the branching engine needs a closed-loop check.

## 14. Reported flow arms: `reports/tdstdp-baseline-flow/`

- **Why:** the baseline covered `lif` and branching only. Flow (§10 S3) is the engine on which Map-STDP's derivation is exact, so the columns-plus-map-term comparison needs a TD-STDP flow baseline.
- **Study:** `tdstdp-baseline-flow`, a companion study (a study uses one code state, and the baseline ran at `675e49e`). Its README was pre-registered on 2026-10-24 at 15:20 EDT, before launch.
  - Groups: CartPole no Adam at B = 20; CartPole no Adam at B = 1 (block-sum control); LunarLander with Adam; LunarLander with the E4′ SGD rule measured on flow. All use `--fast` and the baseline's seeds.
  - Mann–Whitney tests compare against `tdstdp-baseline`'s reference groups, through the runner's new cross-study reference.
- **Development evidence for the block-1 control:**
  - On dev seeds 60–79, flow at B = 20 solves 16/20 CartPole runs. The other 4 collapse by episode 100–150 to a one-action policy (entropy about 0.06, return about 10).
  - At block 1, all 20 reach $T_s$.
  - The B = 20 gate (§11 S1) was measured on branching only.
- **Run:** 60 runs, local, 40 workers, 2026-10-24 15:19 to 22:45 EDT (7.4 h), 0 failures, 0 resumes.
- **Results** (from the study's `summary.md`; MW against the baseline's `lif` reference of each task):

| Group | Solved | $T_s$ (median) | Last-100 return | vs `lif` reference |
| --- | --- | --- | --- | --- |
| CartPole, no Adam, B = 20 | 17/20 | 207.5 ± 56.3 (219) | 415.6 ± 175.0 | $T_s$ p 0.44 |
| CartPole, no Adam, B = 1 | 18/20 | 211.9 ± 65.1 (216) | 401.1 ± 195.8 | $T_s$ p 0.62 |
| LunarLander, Adam | 3/10 | 3990.7 ± 1396.0 (inf) | 132.5 ± 46.5 | $T_s$ p 0.075 |
| LunarLander, no Adam (η calibrated) | 0/10 | – (inf) | −24.2 ± 20.0 | return p 0.00025 |

- **Reading rule 2 (block sum on flow):** 17/20 at B = 20 against 18/20 at B = 1, Fisher p = 1.0. The block sum does not change flow's outcome, so `CartPole_noadam_flow` (B = 20) is the CartPole flow entry. The dev split (16/20 against 20/20 on seeds 60–79) was chance.
- **Reading:**
  - **CartPole:** flow solves 17/20, against 20/20 for `lif` and 5/20 for branching. Removing the escape noise recovers most of what branching loses. The 3 failures are a one-action collapse within the first 100–150 episodes (return about 10), which also occurs at B = 1 (2 never learn, 2 collapse after solving).
  - **LunarLander with Adam:** 3/10 solved, against 8/10 for `lif` and 6/10 for branching. Three more runs peak within 5 of the threshold (trailing-100 of 196–198) without crossing it.
  - **LunarLander without Adam:** 0/10, as on `lif` and branching. The trailing-100 return peaks at 67–103 (episodes 3700–6700), then falls to −24 ± 20 by 10000. A fixed η is unstable late in training on flow, and the noise-free engine does not remove the need for Adam.
- **Naming:** the no-Adam LunarLander groups were run as `LunarLander_sgd_<engine>` (`--sgd`). From 2026-10-24 they are shown as `LunarLander_noadam_<engine>`, and the flag is `--eta-cal` (`--sgd` is a hidden alias). "SGD" suggested backprop; the arm is the local rule with a calibrated fixed η. Earlier sections of this record keep the names that ran.
