# Map-STDP walk replay

Published with GitHub Pages at https://vikramr2.github.io/map-stdp/.

`index.html` replays one episode of each of the 28 best trained models, spike by spike, next to the environment:

- Map-STDP model v4 on CartPole: the flow, branching, QIF and LIF engines, each with the map term on, off and sealed.
- The TD-STDP backbone with the race readout on CartPole, Acrobot and LunarLander: the published backbone and the
  recurrent-column arms.

Each spike is drawn with one sampled cause: an earlier spike that drove it (a walk step), the stimulus, or background
noise. The flow engine has no spikes, so it shows walkers sampled from the walk it computes. Each model's page note
states the rule for its engine and whether it is exact or an approximation.

Hover over a neuron to draw only its own outgoing synapses; click to pin it. `index.html#<model id>@<neuron>` links
to a pinned neuron.

`css/replay.css` and `js/replay.js` are the page's style and script. `data/` holds one JSON file per model plus `index.json`. The page loads them with
`fetch`, so to view it locally, serve this folder over HTTP (`python -m http.server`) rather than opening the file
directly.

`v3-replay.html` is the earlier single-model page (model v3, branching engine, CartPole), with its style in
`css/v3-replay.css`, its script in `js/v3-replay.js` and its episode in `data/v3-replay.json`.

## Rebuilding

The code is on branch `dev`: `experiments/viz.py` (the recorder), `experiments/viz_template.html`,
`experiments/viz_template.css` and `experiments/viz_template.js` (the page, its style and its script). From a `dev` checkout with this branch in `../map-stdp-viz`:

```bash
python -m experiments.viz all --out ../map-stdp-viz/data --workers 28     # about 3 min
cp experiments/viz_template.html ../map-stdp-viz/index.html
cp experiments/viz_template.css ../map-stdp-viz/css/replay.css
cp experiments/viz_template.js ../map-stdp-viz/js/replay.js
```

The v3 page is rebuilt with `python -m experiments.m2_viz page --seed 4 --la-beta 0.3 --out ../map-stdp-viz/v3-replay.html`,
which writes the page, `css/v3-replay.css`, `js/v3-replay.js` and `data/v3-replay.json`.
