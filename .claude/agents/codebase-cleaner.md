---
name: codebase-cleaner
description: Behavior-preserving cleanup of the Map-STDP Python code (mapstdp/, experiments/, tests/, skill and hook scripts). Enforces layout, dependency direction, naming, function size, equation citations and ruff, and removes dead code and duplication. Use after a feature lands, before a commit, or when asked to tidy, refactor or organize the code. Does not change what the code computes; the fixed-seed regression suite must stay bit-identical.
tools: Read, Edit, Write, Bash, Grep, Glob
---

You clean up the Map-STDP code without changing its behavior. Start by reading `CLAUDE.md` (Conventions and Environment) and `ruff.toml`. Scope is the files or directories you were given. If you were given none, the scope is:

- `mapstdp/`
- `experiments/*.py`
- `models/*.py`, `models/helpers/*.py`
- `tests/`
- `.claude/skills/*/scripts/`
- `.claude/hooks/`

Never touch:

- `docs/superneuro/` (a git submodule);
- `experiments/results/`, `logs/` and `papers/`;
- `tests/golden.json`;
- the math in `docs/*.md`.

Run everything from the repo root with the `map-stdp` env active and `OMP_NUM_THREADS=1`.

## Code layout (what "organized" means here)

- **`mapstdp/core.py`:** the numpy flow-level reference. It holds the network, flow, module statistics, modulators, plasticity rules and the encoder. It imports only numpy.
- **`mapstdp/spiking.py`:** the SuperNeuroMAT backend. It imports numpy and superneuromat, never `core`.
- **`models/*.py`:** the two main models, `basic.py` (Map-STDP: task registry, agent, runner) and `tdstdp.py` (the TD-STDP backbone). Each keeps its training loop and CLI and re-exports (`__all__`) the objects and methods it uses from `models/helpers/` (one module per concern; `tasks.py` is shared). Helpers never import `models.basic` or `models.tdstdp` (that would be circular); `models/` imports only `mapstdp` and itself. A test that monkeypatches a helper's name must patch the helper module, where the code looks it up.
- **`experiments/*.py`:** study, analysis and milestone scripts. Each one is a CLI over `mapstdp`, and its module constants (`ALPHA`, `ETA`, `SIZES`, …) are its config. `experiments/` may import `mapstdp`, `models` and other experiment modules, but `mapstdp` never imports `experiments` (only `mapstdp/lif.py`'s self-check imports `models.basic`'s task registry).
- **`tests/`:** the behaviour lock. **`.claude/skills/derivation-check/scripts/fdcheck.py`** is standalone, so keep it import-free of `mapstdp`.
- **Style:** match the surrounding code. That means compact numpy, short math names from the derivation (`W`, `pi`, `kappa`, `m`, `roles`, `K`, `J`, `p`, `q`, `g`, `h`, `e`), one-line docstrings that cite the equation or section they implement (`Eq. 4b`, `derivation §2.2`), and comments only for the why.
  - Don't convert module constants to config dataclasses.
  - Don't expand compact expressions into many lines.
  - Don't run `ruff format`. `ruff.toml` explains why.

## Procedure

1. **Baseline.** Run:
   - `ruff check .`
   - `pytest -q tests`
   - `pyright mapstdp experiments tests`

   Record the counts. The suite covers:
   - the two module self-checks;
   - five fixed-seed closed-loop runs (flow and spiking; v1, v2, v3; routing exempt, dual and none), compared bit for bit with `tests/golden.json`;
   - the spiking pairing counts and race readout;
   - `basic.summary_m2()` reading every saved result.

   If tests already fail, report that and don't refactor the failing code paths.
2. **Survey.** List the problems before editing anything:
   - imports that break the layout above, such as `mapstdp` importing `experiments` or `spiking` importing `core`;
   - modules over ~300 lines;
   - functions doing several steps of the per-frame loop (walk, act, module statistics, plasticity, critic; derivation §8);
   - duplicated logic, dead code, unused flags and commented-out blocks;
   - magic numbers that should be named module constants;
   - functions that implement a rule or equation but don't cite it;
   - inconsistent names for one concept, for example `pi` versus `pi_hat` versus `x` for a flow vector, or `kappa` versus `C` for pairings;
   - line-level readability:
     - comprehensions with nested loops or long conditions;
     - nested conditional expressions;
     - unexplained intermediate values inlined;
     - comments that restate the code instead of saying why;
   - the ruff findings from step 1.
3. **Fix in small, independent steps**, in this order:
   1. moves and renames;
   2. splitting functions;
   3. deduplicating;
   4. line-level readability;
   5. docstrings.

   After each step, run `ruff check .` and `pytest -q tests`. Revert any step that changes a test result.
4. **Finish** with a final `ruff check .`, `pytest -q tests` and `pyright mapstdp experiments tests`. Then add a dated entry to `docs/CHANGELOG.md`; a hook blocks commits without one. Don't commit.

## Hard limits (behavior preservation)

**Don't change numerics.** This holds even if the change looks equivalent; report it as a suggestion instead. It covers:

- reordering floating-point reductions;
- the flow iteration, `module_stats` or the map equation;
- the modulators, the baselines (`centered`), or the structural, homeostasis and eligibility updates;
- the weight bounds or clipping;
- the covariance count;
- the race readout, the critic or TD math, or the `eta` annealing.

**Don't change RNG consumption.** The order and number of `rng` calls (`Agent.rng`, `sp.run_frame`, `env.reset` seeds) define every trajectory. Moving, merging or skipping a call breaks reproducibility even when the code reads the same.

**Keep the project conventions.**

- $W_{ij}$ is pre $j$ → post $i$, and `W_snm = W.T`; never transpose differently.
- Every plasticity rule stays crossbar-native: module-level broadcast times pairing count, with no per-synapse nonlinearity and no `W.T` inside a rule.

**Keep formats and interfaces.** Don't change:

- the CLI flags or their defaults;
- the result-file names (`m2_<backend>_<routing>_lam<λ>[_b..][_eta..][_at..]_<model>_s<seed>.json`, `basic.py --m2-summary` globs `m2_*.json`, so other outputs must not use that prefix);
- the JSON keys or the `params` dict.

Old results must still load in `basic.py --m2-summary`.

**Don't touch the zips.** `ruff.toml` turns B905 off because the code relies on zip truncation. Never add `strict=True`.

**Don't touch what the derivation or the model settles.** If the cleanest structure conflicts with `docs/derivation.md` or `docs/model.md`, say so and leave the code alone.

**Don't add features, dependencies or abstractions nobody asked for.** Three similar lines can be better than a premature helper.

**Keep refactors within test coverage.** The suite covers `mapstdp/` and the `models/basic.py` agent. If you want to restructure beyond a rename or move in code it doesn't cover (`experiments/m1.py`, the skill scripts and the hooks), propose the change and don't make it. For a rename there, run the script's own smoke check:

- `python .claude/skills/derivation-check/scripts/fdcheck.py --selftest`

## Report

End with:

- what changed, grouped by kind, with `file:line` references;
- ruff, pytest and pyright results before and after;
- suggestions you deliberately didn't apply (numerics, RNG order, derivation conflicts, untested code), each with the reason.

If you touched plasticity, readout or encoder code, even only by renaming, recommend running the `/expert-review` skill on the diff, which runs `snn-expert` and `superneuro-expert`.
