"""Run a saved race-study model (models/<study>/<arm>/s<seed>.npz) in its environment with plasticity off and print the
return per episode and their mean. Usage: python models/evaluate.py <model.npz> [--episodes N] [--seed S] (run
from the repo root, env map-stdp; the policy is stochastic, so returns vary with --seed)."""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # run as a script from the repo root

from experiments import tdstdp

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--seed", type=int, default=12345, help="rng seed of the evaluation (noise, environment)")
a = ap.parse_args()
r = tdstdp.load_ck_model(a.model, rng_seed=a.seed)
while len(r.returns) < a.episodes:
    r.tick()
ret = np.array(r.returns[:a.episodes])
print(f"{a.model}: {len(ret)} episodes, return mean {ret.mean():.1f} sd {ret.std():.1f}, "
      f"min {ret.min():.1f}, max {ret.max():.1f}")
print(" ".join(f"{x:.0f}" for x in ret))
