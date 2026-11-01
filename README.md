# Map-STDP replay page

`index.html` replays one CartPole episode of a trained spiking Map-STDP agent (model v3), spike by spike: the walk's
hops between modules, the race between the action modules, and the cart. It is published with GitHub Pages at
https://vikramr2.github.io/map-stdp/.

This branch holds only the page. The code that generates it is on branch `dev`, in `experiments/m2_viz.py` and
`experiments/m2_viz_template.html`. To rebuild it from a `dev` checkout:

```bash
python -m experiments.m2_viz page --seed 4 --la-beta 0.3 --out index.html   # then copy index.html to this branch
```
