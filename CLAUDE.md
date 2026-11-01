# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

The project is research, and the docs lead the code. M1 (flow reference against SuperNeuroMAT) and M2 (CartPole closed loop, model v3) are done; M3 (MountainCar, Acrobot, LunarLander, plus parallel spiking heads) is in progress (iterations 005–012; model v4 is a flag configuration of the M3 runs, and the code defaults still reproduce v3), and M4 (partial observability, carry-over) is next (SPEC §11). The unmerged branch `m4-tmaze` holds an M4 prototype on a T-maze: carry-over with a module-level winner-take-all, Eqs. 1f/4f/4g, `experiments/m4.py` and `docs/iterations/004-m4.md`. Its iteration number collides with `main`'s `004-map-term-ablation.md`, so renumber it on merge. The earlier PyTorch prototype was deleted on purpose; it is in git history at `fcedb0a` if you need it for reference. Do not restore it unless asked.

Research question: can a spiking network's communities serve as its states, giving it a state-space abstraction that solves RL tasks? **Map-STDP** inverts minimum description length: community detection finds the partition that minimises a walk's description length (the map equation) for a given graph, while Map-STDP fixes the partition and learns the weights. It is a three-factor rule. TD-modulated STDP with an action-gated eligibility trace learns the task. Cost-modulated STDP, whose per-module neuromodulator broadcasts the marginal description cost of each transition, makes the given partition the minimum-description-length one. The communities are a central controller (receives the stimulus), one action community per action, and latent communities. The coarse-grained module chain $T^K(\mathbf o)$ is the extracted, stimulus-conditioned Markov model. Each environment frame is one implicit random walk. Side hypothesis, secondary: description length tracks metabolic/thermodynamic cost.

## Docs

- `docs/derivation.md` is the **canonical** derivation. It states the current model only: the walk observed from spikes (a step j → i is a spike of i within a window after a spike of j; the flow engine is the special case without chance or indirect terms), the map-equation rule with its two statistics (pair counts κ for the baseline, the signed counter c for the step; Lemma E, Theorem E), the four engines (§5) and stability with inhibition (§7); Appendix A holds the proofs and numerical checks (`fdcheck.py --emergent`). `docs/derivation-guide-emergent.md`, `docs/derivation-details-emergent.md` and the PDF are the user's source drafts. History goes in the CHANGELOG and iteration records, not in the derivation. Put math changes in the markdown.
- `docs/neuron-variants.md` is a plain-language explainer of Map-STDP on the four engines (flow, branching, LIF, QIF): what the rule assumes, how each engine meets it, and the limitations of LIF and QIF. It explains the derivation, never overrides it.
- `docs/SPEC.md` holds the goals, the design considerations (simulator, tasks, stimulus, description length as neuromodulator, memristive hardware), open questions and milestones. Items marked **Proposal** are undecided. Do not treat them as settled.
- `docs/model.md` is the current implemented model (now v3): every component, its parameter values, and a status tag (impl / spec / Proposal / abstraction). Bump its version and history when the implemented model changes.
- `docs/iterations/NNN-*.md`: one record per experimental iteration, covering the problem, diagnosis, runs (commands, SLURM job IDs) and results. Numbers stay preliminary until they are reproduced through the repo CLI.
- `docs/CHANGELOG.md`: add a dated entry for any change to docs or code, including decisions and reversals.

## Code

`mapstdp/` is the library and `experiments/` holds the milestone scripts. Docstrings cite the derivation's equation numbers (Eq. 1a, 4b, …), so read the code and `docs/derivation.md` together.

- `mapstdp/core.py`: the numpy flow-level reference. It builds the SBM network, computes the stimulus $v(\mathbf o)$, the n-hop flow, the module tables ($p$, $q$, $J^K$), the map equation and its gradient, the modulator $g$, and the structural, homeostasis and eligibility updates. Module tables are $K \times K$, indexed `[b, a]` = (post module, pre module). Logs are base 2.
- `mapstdp/spiking.py`: the SuperNeuroMAT backend. It runs in branching mode (`leak=inf`, uniform threshold noise, so one step is one hop), and teleportation is Bernoulli input spikes. `run_frame` returns the frame's spike raster, from which `pi_hat` (spike shares), `pairings` (the covariance count $\kappa$) and `race` (the action readout) are computed.
- **`rule="emergent"`** (`m2.Config.rule`, `--rule emergent`; default `"v4"` keeps the golden tests): the observed-walk rule of `docs/derivation.md`. Each spiking engine's `Agent.counter_frame(v, rng)` returns `(S, kappa_raw, c)` (raster, raw pair counts in the engine's window, signed counter; summed over heads), and `core.observed_walk`, `structural_update_emergent`, `homeostasis_update_emergent` and `eligibility_step_emergent` apply Eqs. 8, 4, 4d and 4b. Engines: flow (bit-identical to v4), branching (`spiking.branching_counters`, exact Eq. 11a), `mapstdp/qif.py` (Eq. 11b, gain regressed on the injected noise) and `mapstdp/lif.py` (noisy LIF, τ_m 20 ms, 60 ms window, Siegert gain; `--engine lif`, emergent only). `--match-scales` RMS-matches λ and η to flow at initial W; `--save-model` writes W snapshots; runs checkpoint every 50 episodes and resume. `experiments/bound_cal.py` makes the diffbound calibration tables (`BOUND_CAL*`); `experiments/structure.py` compares saved models (block weight shares, module tables, L of W's walk; on vs off).
- `experiments/m2.py`: `Agent` is the closed-loop CartPole agent. **Both backends share one frame update** and differ only in where $\pi$, $\kappa$ and the action come from (flow: n-hop flow, $J$, Eq. 1d; spiking: spike shares, covariance count divided by the frame's spike count, race readout). Only model v3 remains in code (v1/v2 and dual routing were pruned on branch `columns-tdstdp`; their committed results still load in `--summary`). Every option is a field of the frozen `m2.Config`: `Agent(cfg)`, `run(cfg, episodes)`. Results go to `experiments/results/m2_<backend>_<routing>_lam<lam>[...]_<model>_s<seed>.json`, which are committed.
  `Config` fields include `task` (a key of `experiments/tasks.py`), `heads` (parallel spiking replicas on shared W, pooled), `engine` (`snm` or `numpy` = `spiking.run_heads`, raster-identical and much faster), `td_norm` and `ablate`; all defaults reproduce M2.
- `experiments/tasks.py`: the M3 task registry (sizes, encoder, reward, episodes). `experiments/m3.py`: per-task runs and summaries, results in `experiments/results/m3_<task>_<backend>[_h<H>]_lb<β>[_tn<s>][_<variant>]_s<seed>.json`. `experiments/ceiling.py`: the policy-class ceiling (CEM over the flow policy's actions × controller matrix).
- `experiments/viz.py` + `viz_template.html`: the walk replay of the best models (`models/`), published from the orphan branch `viz` (`index.html` + `data/`) with GitHub Pages at https://vikramr2.github.io/map-stdp/. `viz.py all --out DIR` records one frozen episode per model, capturing spikes by wrapping module functions (no RNG draw moves; `tests/test_viz.py`), and samples each spike's cause: a parent spike (a walk step), the stimulus or background, from the parent posterior of a linear (Hawkes) process with pre-set measured kernels; exact on branching, walkers on flow. Rebuild steps are in the `viz` branch README.
- `experiments/m2_viz.py` + `m2_viz_template.html`: the earlier v3 replay (`v3-replay.html` on `viz`); `replay.html` here is gitignored. Rebuild: `python -m experiments.m2_viz page --seed 4 --la-beta 0.3`.
- `mapstdp/tdstdp.py` + `experiments/tdstdp.py` (iteration 013, a baseline, not Map-STDP): the feed-forward TD-STDP actor-critic of Chung & Kozma 2020 (docstrings cite "[CK Eq. n]") on the branching step without presynaptic normalisation, plus BindsNET's LIF as a control (`--engine lif`) and the branching step's noise-free limit (`--engine flow`). The default follows the authors' code (`reference/td_stdp`, gitignored) where it disagrees with the paper: `PRESETS` are their `config_cp.ini` and `config_ll.ini` in mV, converted to branching units by `MV` = 13, and `CKRunner` and `block_learn_ck` transcribe their loop and rule. Traces are summed in closed form over blocks of frozen W (block 1 = the per-ms rule). `--legacy` reproduces the pre-audit runs (CK Table 1, `MV_LEGACY` = 1300). Results go to `experiments/results/tdstdp_<task>[_ck][_<tokens>]_s<seed>.json`.
- `experiments/ablate.py` (iteration 017): a saved model with plasticity off under ablation conditions: v4 latents lesioned, removed or the only path to the actions; TD-STDP recurrent columns off, lesioned across or within columns, or untrained. Study runner `ablate` (`summarize_ablate`: paired Wilcoxon on ablated − intact per model).
- `experiments/report.py`: the final-study runner. Studies are declared in `STUDIES` (groups → task, engine, flags, seeds; a reference group per task; gates). `run` launches the declared runs through `--out` and resumes checkpointed ones; A study's `runner` is `tdstdp` or `m3`. `summarize` rebuilds the summary from `runs/` only, with `tdstdp.group_stats` (the statistics `tdstdp --summary` uses) or `m3_stats` and the declared on/off `pairs`; a study's reference may be another study's group, `"<study>/<group>"`.
- **Results layout.**
  - `experiments/results/` is the **intermediates** store: every dev, tuning and diagnostic run, and the default output of every CLI. Existing files stay where they are, because tests and iteration records reference them.
  - `reports/<study>/` holds **only final reported outputs**, one directory per study:
    - `README.md`: the pre-registered design and gates, written before the runs;
    - `manifest.json`: git HEAD, the sha256 of the uncommitted code in `mapstdp/` and `experiments/` (saved as `code.patch`), versions, and the exact per-run commands;
    - `runs/<group>/s<seed>.json`: the raw results;
    - `summary.csv` and `summary.md`.
  - `models/` holds the best run per arm of the race studies and of `emergent-v4` (`models/README.md`); `archive/models/` (gitignored, kept on a server) holds every saved model.
  - A study uses one code state. The runner refuses to resume if the code changed. Checkpoints and partials (`*.ckpt.npz`, `*.partial`) are gitignored.
- Each `mapstdp` module has a `__main__` self-check (Eq. 1a bound, finite-difference gradient, Monte Carlo three-factor average, `W_snm` orientation). Extend it when you add a rule.

**Traps** (each has cost time before):

- **RNG order defines every trajectory.** The golden tests are bit-identical, so moving, merging or skipping an `rng` call breaks `tests/golden.json`, even if the code reads the same. Put new features behind flags whose default path draws nothing extra.
- **Never add `zip(..., strict=True)`.** `m2.Agent.evaluate` relies on truncation (20 held-out stimuli zipped with 100 policies), which is why ruff's B905 is off in `ruff.toml`.
- **Result-file prefixes are namespaces.** `m2.summary()` globs `experiments/results/m2_*.json`, and the test suite reads every match, so other outputs must not start with `m2_`. The replay files use `viz_m2_*` for this reason.
- **Silent modules in spiking.** A sealed or winner-take-all module can emit no spikes in a frame. `module_stats` then has 0/0 columns, so guard any new ratio or credit computed from spike shares.
- **Final numbers come only from `reports/`; never hand-copy results there.** Only `python -m experiments.report` writes into `reports/<study>/` (apart from its pre-registration README). Numbers from `experiments/results/` are development results.

## Commands

Run everything from the repo root, as modules:

```bash
python -m mapstdp.core                     # self-checks (likewise mapstdp.spiking, mapstdp.tdstdp)
python -m experiments.m1                   # M1 checks -> experiments/results/m1.json
python -m experiments.m2 --backend flow --routing exempt --lam 0.05 --seed 0 --episodes 1500   # one M2 run
python -m experiments.m2 --summary         # results table over seeds, plus the random-policy return
python -m experiments.m3 --task Acrobot --backend flow --td-norm 0.04 --seed 0           # one M3 run (spiking: --heads H)
python -m experiments.m3 --task LunarLander --td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5 --seed 0   # model v4 (flow)
python -m experiments.m3 --task MountainCar --td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5 --encoder g12s0.15 --critic-layer g24 --c-ent 0.01 --ent-target 0.8 --seed 0   # best MountainCar (iteration 012)
python -m experiments.m3 --task CartPole --rule emergent --td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5 --backend spiking --engine lif --heads 16 --readout diffbound --match-scales --seed 0 --save-model M.npz   # emergent rule (engines: flow, numpy = branching, qif, lif)
python -m experiments.structure --models DIR --pair ON OFF   # structure of saved models, map term on vs off
python -m experiments.m3 --summary         # per task: random, flow, spiking per H  (--ablations, --fidelity)
python -m experiments.ceiling --task CartPole --level 1 --readout 1d                     # policy-class ceiling
python -m experiments.tdstdp --task CartPole --seed 0 [--adam | --eta-cal] [--block 20] [--engine lif | flow] [--fast] [--legacy] [--out PATH]  # TD-STDP backbone (iteration 013)
python -m experiments.tdstdp --summary      # T_f, T_s, solved fraction, last-100 return vs Chung & Kozma and v4
python -m experiments.report run tdstdp-baseline --workers 40   # a final study -> reports/<study>/ (resumes; then summarizes)
python -m experiments.report summarize tdstdp-baseline          # rebuild reports/<study>/summary.csv and .md from runs/
python -m experiments.report summarize tdstdp-baseline-flow     # the flow-engine arms (companion study)
python -m experiments.ablate --family v4 --model reports/emergent-v4/models/CartPole_lif_on/s2000.npz --out X.json   # ablations of one saved model (--family ck: race models)
pytest -q tests                            # behaviour lock (see Tooling)
pytest -q tests -k "flow-exempt"           # one case, e.g. one golden fingerprint
ruff check .
```

M2 runs are per seed (5 seeds × 1500 episodes is the usual reporting unit); spiking runs are much slower than flow runs.

## Conventions

- $W_{ij}$ is the synapse from **presynaptic $j$ to postsynaptic $i$**. The random walk follows spikes forward: $T_{ij} = W_{ij}/d_j$, where $d_j$ is $j$'s total outgoing weight.
- SuperNeuroMAT's `weight_mat()` is indexed `[pre, post]`, the transpose of $W$: `W_snm = W.T`. Its built-in STDP has no third factor, so it stays off; Map-STDP is applied per frame from `ispikes`.
- The stimulus enters the flow through teleportation: $\pi = (1-\alpha)T\pi + \alpha v(\mathbf o)$.
- Flow is per frame: $\pi_f$ solves that equation with $v(\mathbf o_f)$, approximated by $n$ hops of power iteration. Frames are indexed $f$; $t$ is the simulation step. Module types are $\mathcal C$ (controller), $\mathcal A$ (actions) and $\mathcal L$ (latent); controller and action modules are pinned.
- Structural rule: $\Delta W_{ij} = -\lambda (g_{ij} - \bar{g}_j) \kappa_{ij}$, where the modulator $g_{ij} = \partial D / \partial J_{ij}$ is the **marginal** cost, never the pointwise cost, and $\bar{g}_j$ is a per-neuron baseline. Its expectation is $-\lambda W_{ij} \partial D / \partial W_{ij}$. For the map equation it averages to the gating $G(i,j) = \mathbb{I}[i \notin m(j)] - \bar{e}_j$. Controller→action/latent routing is cross-module, so the map term alone starves it (derivation §4.5). The implemented default **exempts** the routing pairs ($g = 0$ there); the dual routing term (4a) is an ablation whose multiplier diverges (flow reference in `m1.py` only), and the latent-output barrier (4e) is a **Proposal**.
- **Hardware target: memristive crossbars.** Every rule must be crossbar-native: per-module broadcast modulators times pulse-coincidence STDP. Modulators must be $K \times K$ module-pair tables. No nonlinear function of an individual synapse's conductance, and no access to the transposed $W_{ji}$. The only per-synapse state beyond $W$ is the task eligibility trace.
- Math must render in GitHub, VS Code and Markdown Preview Enhanced: put display `$$` on their own lines with blank lines around them, use `\lbrace`/`\rbrace` and `\lVert`/`\rVert` instead of `\{`/`\}` and `\|`, avoid `\,`, `\;` and `\tag` (number equations as `\qquad \text{(n)}`), and use braced arguments (`\mathbb{I}`, `\bar{e}`).
- Before changing a derivation, check the gradient claims numerically (finite differences on a small random network), and check that a new modulator's three-factor average matches $-W \odot \nabla D$ by Monte Carlo.

## Environment

The planned spiking simulator is **SuperNeuroMAT** (ORNL; see SPEC §3), with a numpy flow-level reference run first. Brian2 is only a fallback. `docs/superneuro` is the SuperNeuro reference submodule (README and tutorials). `environment.yml` lists the current stack (numpy, scipy, numba, gymnasium, superneuromat 3.5.0, Node.js); the npm tools are listed at its top. Never install packages globally without asking; use a scratchpad `--target` or a venv for experiments.

**Machines.** Work moves between machines. Paths in the hooks and MCP config resolve from the active env, so any machine needs only:

1. Clone the repo, then `git submodule update --init`.
2. Create the `map-stdp` env from `environment.yml`, plus the npm tools listed at its top.
3. Launch Claude from `conda activate map-stdp`.

The SLURM notes below apply only to the campus cluster. On a machine without SLURM, run experiments directly.

**Compute (SLURM, campus cluster only).** The login node is for editing, quick checks and short tests only. Anything longer than a few minutes goes to SLURM:

- **Submitting.** Use `sbatch [--job-name=NAME] slurm/job.sbatch <command...>` from the repo root. The template activates `map-stdp` and runs the command.
- **Partition: `secondary`** (the user's choice). Defaults are 1 node, 8 CPUs, 64 GB, 4 h.
  - **4 h is the hard limit.**
  - **Jobs can be preempted and requeued,** so long experiments must checkpoint and resume, or be split into jobs under 4 h, for example one per seed.
- **Other partitions** need the user's OK:
  - `zgdrasil` starts immediately, with 7 days and 2× A30 on 1 node;
  - `IllinoisComputes` needs `-A chackoge-ic` and allows 3 days, on 128-core nodes.
- **Logs.** They go to `logs/py_<jobid>.out` and `.err`; `logs/` is gitignored and must exist before submitting.
- **Monitoring.** Check jobs with `squeue -u $USER`, or `sacct -j <id>` after they finish.
- **Recording.** Put the job ID and the command in the iteration record in `docs/iterations/`.

**Tooling.** The `map-stdp` conda env has Python 3.11, Node.js and the packages in `environment.yml`. On the campus cluster it lives at `/scratch/vikramr2/conda/envs/map-stdp`, off `/u`, which has a 500k-file quota. On the local workstation it is `~/miniforge3/envs/map-stdp` (Miniforge in the home directory).

- On the cluster, `conda activate map-stdp` resolves to it through `envs_dirs` in `~/.condarc`.
- On the cluster, for installs, set `CONDA_PKGS_DIRS=/scratch/vikramr2/conda/pkgs`, and use `pip --no-cache-dir` and `npm --cache /scratch/vikramr2/conda/npm-cache`, so caches stay off `/u`.

- **Launched outside the env,** the KaTeX hook finds a Node without katex and errors after every doc edit, and CodeGraph does not connect. Relaunch from the env, or run `node .claude/hooks/math_check.js <files>` with the env's `bin` on `PATH`.
- **ponytail** (project-scope plugin, `.claude/settings.json`): a least-code skill. Its hooks run `node`, so run `conda activate map-stdp` before launching `claude`.
- **CodeGraph** (project-scope MCP server, `.mcp.json`; telemetry off): a code-intelligence index. The index lives in `.codegraph/`, which is gitignored by its own `.gitignore`. Run `codegraph sync` after large changes.
- **Hooks** (`.claude/hooks/`):
  - KaTeX math check after editing docs;
  - commits blocked without a CHANGELOG entry ("skip changelog" overrides);
  - installs blocked outside the env;
  - an env warning at session start.
- **Skills:** `/derivation-check` (run before any derivation math change), `/doc-edit`, `/expert-review`, `/changelog`.
- **Behaviour lock:** `pytest -q tests` runs the module self-checks and fixed-seed closed-loop runs that must match `tests/golden.json` bit for bit. Regenerate it (`python -m tests.test_regression --update`) only for an intended behaviour change, and log it in the CHANGELOG. Lint with `ruff check .` (rules pinned in `ruff.toml`; no `ruff format`).
- **Agents:** `neuroscientist`, `snn-expert`, `superneuro-expert`, and `codebase-cleaner` (behaviour-preserving cleanup, gated by the behaviour lock).
- **Plugins:** math-proof, pyright-lsp (pyright is in the env), commit-commands, hookify.
