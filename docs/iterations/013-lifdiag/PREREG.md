# LIF frozen-weight diagnostic: pre-registration (written 2026-10-24, before any measurement)

Question: does the linear-response walk of derivation §2.1/§5 (K1-K9) hold for the v4 map network
(controller, 2 latents x16, action modules x16; p_cc = 0; SBM init from core.build_network via m2.Agent with the
v4 flags; trained W = flow v4 runs in scratch qif/trained/{task}_s0.pkl, default encoders) when simulated with
TD-STDP's BindsNET LIF (tau 100 ms, threshold 13 mV above rest, reset to rest, refractory 0, 1 ms steps,
delay-1 synapses, W in mV)?

Network in mV: recurrent W_mV = chi * T (presynaptic normalisation as in spiking.set_weights), chi in mV.
Nominal offspring mean chi / 13. Stimulus: each neuron i gets Bernoulli external spikes with p_i = R v_i per ms,
w_in = 4 mV each. R calibrated per (task, chi) on init W, deterministic, so the mean rate over all neurons is
0.15 / ms (reported). Noise variants reuse R.

Operating points (LIF): det chi=10.4 (nominal nu 0.8, default); input noise sigma = 1, 2, 4 mV per sqrt(ms) at
chi=10.4; exponential escape noise Delta = 2 mV (hazard exp((v-13)/Delta) per ms) at chi=10.4; det chi = 6.5, 12.35
(nu 0.5, 0.95). Frames: (P) 20 ms persistent state, stimulus switched i.i.d. among S held-out stimuli every frame;
(R) cold-start reset frame of 200 ms per stimulus. Frame statistics use the whole frame (as the engine does).
Control: branching engine (spiking.run_heads semantics, gain 0.8, rate 1): native 100-step cold-start frame, and
20-step persistent frames.

Measurements (per network x op x stimulus, S = 4 held stimuli, networks = CartPole/LunarLander x init/trained):
- r: stationary rates (long persistent run). G_i by open-loop replay: each neuron re-simulated with its recorded
  input (+ delta constant current, common random numbers): G = d r_i / d x_i, x in mV/ms. Also the kick gain
  (extra spikes per extra recurrent-size kick / kick size). Lambda_model = diag(G) B, B = W_mV.
- R = d r / d I_ext by network perturbation (each neuron's input +delta, CRN): the model-free response; K2 test:
  rel. error of R vs (I - Lambda_model)^-1 diag(G).

Checks and pass thresholds:
1. pi: frame-averaged spike shares vs pi_K3 = normalise((I - Lambda_model)^-1 (G * I_ext)) (stimulus-driven
   immigrants, zero intercept). PASS iff TV <= 0.05 AND RMS module-level bias <= RMS single-frame module-level
   sampling SD. Also reported: TV vs pi of Eq. (1) (core.flow, alpha 0.2), and mu = r - Lambda r (negative share).
2. kappa vs J^Lambda = Lambda_model diag(pi) on existing synapses, frame-averaged covariance count. zeta = sum kappa /
   sum J^Lambda; slope = OLS slope (with intercept) of kappa / zeta on J^Lambda. PASS iff corr >= 0.9 and slope in
   [0.8, 1.25]. Primary window: lag 1 (implemented). Secondary windows reported: flat lags 1-5 and 1-20.
3. rho(Lambda_model) < 1 PASS; gain CV (over neurons with r > 0.005/ms) reported.
4. Rule average: cosine(mean over frames of Delta W = -(g - gbar^eff) kappa [Theorem K, g from the map equation on
   J^eff with the routing pairs excluded from q (exempt, exact-gradient form), G measured], -nu_j W * dD/dW) with
   dD/dW by central finite differences of D(J^eff(W)) at fixed pi, G. PASS iff >= 0.9. Also reported: the same
   against FD of the full linear-response walk (pi re-solved from (I - diag(G) B(W))^-1 G I_ext, G fixed).
5. Score: exact score = model-free W * dlogP/dW = chi T_ij r_j (psi_i - psibar_j), psi = R^T s (greedy action,
   beta 1). Candidates with measured kappa (primary window): (4b) as implemented; kappa * Phi (Phi = psi / G);
   kappa * Phi^K (module-level, 1 table). Cosines after removing column rescaling. PASS iff >= 0.9 (each reported).
6. Relaxation: ensemble-averaged pi(t) (1 ms bins, 3-bin smoothing) after (a) switch from stationary stimulus A to B,
   (b) cold start; t95 = first time TV(pi(t), pi_B) <= 0.05 (above floor). PASS iff t95 <= 20 ms (env step).
   Also the TV of the first 20 ms frame average after a switch.

Overall: an op passes iff checks 1-6 pass on every network (worst case over stimuli for cosines: mean over stimuli
reported, pass on the mean). Branching control is expected to pass 1-5 (Appendix A); its numbers calibrate.

## Amendment 1 (after the smoke test, before any full run)
The first 4 held stimuli are consecutive states of one random-policy episode (pairwise stationary-pi TV 0.01-0.02),
which makes the switching checks vacuous. Stimuli are instead the 4 held stimuli chosen by farthest-point selection
on TV(v) (pairwise pi TV 0.12-0.27). Consecutive real frames differ by pi TV ~0.01-0.02; this is reported as context
for the persistent-frame results. Smoke-test numbers (CartPole init, quick mode) are not used.

## Amendment 2 (after the branching control's full run, before any LIF full-run analysis)
Calibration from the control: the A->B pi difference is mostly controller mass, which external spikes set at once,
so full-network t95 is ~1 step even for branching. Secondary check 6b added: t95 of TV over non-controller neurons
(renormalised), same 0.05 threshold. Check 5 adds a diagnostic variant (4b) with stationary h (kappa error only).
