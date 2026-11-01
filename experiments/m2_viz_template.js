// The v3 replay page (experiments/m2_viz.py page): its script tag's data-src names the recorded episode.
fetch(document.currentScript.dataset.src).then(r => r.json()).then(D => {
  const N = D.m.length, K = D.k, F = D.frames.length;
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const roleVar = ['--ctrl', '--lat1', '--lat2', '--actL', '--actR'];
  let C = {};
  const readColors = () => { for (const v of ['--ink','--muted','--rule','--edge','--hop','--cart','--pole','--track','--panel',...roleVar]) C[v] = css(v); };
  readColors();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readColors);
  new MutationObserver(readColors).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  const trainR = D.train_returns, last = trainR.slice(-100);
  document.getElementById('subtitle').textContent =
    `spiking agent, model v3${D.la_beta ? `, latent-output barrier β = ${D.la_beta}` : ''}, seed ${D.seed} · trained ${trainR.length} episodes (last 100: ${(last.reduce((a,b)=>a+b,0)/last.length).toFixed(0)}) · this episode: ${F} frames`;
  document.getElementById('foot').textContent =
    `Each CartPole frame is one ${K}-step random walk on the SuperNeuroMAT network. The stimulus teleports spikes into the controller grid cell nearest (θ, θ̇); spikes branch along synapses (α = ${D.alpha}). Each spike is either a stimulus input (ring) or caused by one spike a step earlier; that cause is drawn as a hop, sampled in proportion to synaptic drive, which is exact in this branching mode. Grey lines are all synapses between modules, darker where stronger. After a ${D.burn}-step burn-in, the first action module to spike wins the race and sets the push. Weights are frozen for this replay.`;

  // ---- layout (unit square, aspect 16:11) ----
  const mod = D.m, nMod = Math.max(...mod) + 1, pos = new Array(N);
  const members = Array.from({ length: nMod }, (_, k) => mod.map((x, i) => x === k ? i : -1).filter(i => i >= 0));
  const centers = [[0.2, 0.5], [0.52, 0.22], [0.52, 0.78], [0.84, 0.25], [0.84, 0.75]];
  members[0].forEach((n, c) => { const i = Math.floor(c / 6), j = c % 6;  // theta index i (x), theta_dot index j (y)
    pos[n] = [0.07 + i * 0.052, 0.24 + (5 - j) * 0.072]; });
  for (let k = 1; k < nMod; k++) members[k].forEach((n, c) => {
    const a = c * 2.39996, r = 0.022 * Math.sqrt(c + 0.5);
    pos[n] = [centers[k][0] + r * Math.cos(a) * 0.69, centers[k][1] + r * Math.sin(a)]; });
  const drawE = D.edges.filter(([post, pre]) => mod[pre] !== mod[post]).map(([post, pre, T]) => [pre, post, T]);
  const maxT = Math.max(...drawE.map(e => e[2]));
  // per frame: hops and input spikes indexed by step
  for (const fr of D.frames) {
    fr.hopAt = Array.from({ length: K }, () => []); fr.inAt = fr.X;
    for (const [tt, j, i] of fr.H) fr.hopAt[tt].push([j, i]);
  }
  const isLatent = k => D.roles[k] === 'L', isAction = k => D.roles[k] === 'A';
  const actMods = [3, 4];

  // ---- canvases ----
  const net = document.getElementById('net'), plant = document.getElementById('plant');
  const nx = net.getContext('2d'), px = plant.getContext('2d');
  function fit(cv) { const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1;
    cv.width = Math.round(r.width * d); cv.height = Math.round(r.height * d); }
  const fitAll = () => { fit(net); fit(plant); };
  new ResizeObserver(fitAll).observe(net); fitAll();

  // ---- playback state ----
  let f = 0, t = 0, acc = 0, playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  let speed = 500, lastTs = null;
  const glow = new Float32Array(N), ring = new Float32Array(N), hops = [];  // hop: [pre, post, age]
  let nHop = 0, nCross = 0, nLA = 0, nIn = 0;
  const modCount = new Float32Array(nMod);
  const scrub = document.getElementById('scrub'); scrub.max = F - 1;
  const playBtn = document.getElementById('play');
  playBtn.textContent = playing ? 'Pause' : 'Play';

  function resetFrame() { t = 0; glow.fill(0); ring.fill(0); hops.length = 0; modCount.fill(0); nHop = nCross = nLA = nIn = 0; }
  function stepOnce() {
    const S = D.frames[f].S;
    const fr = D.frames[f];
    for (let i = 0; i < N; i++) { glow[i] *= 0.72; ring[i] *= 0.72; }
    for (const h of hops) h[2]++;
    while (hops.length && hops[0][2] > 4) hops.shift();
    for (const n of S[t] || []) { glow[n] = 1; modCount[mod[n]]++; }
    for (const n of fr.inAt[t] || []) { ring[n] = 1; nIn++; }
    for (const [j, i] of fr.hopAt[t]) {
      hops.push([j, i, 0]); nHop++;
      if (mod[i] !== mod[j]) { nCross++; if (isLatent(mod[j]) && isAction(mod[i])) nLA++; }
    }
    t++;
    if (t >= K) { f++; if (f >= F) { f = 0; } resetFrame(); scrub.value = f; }
  }

  function drawNet() {
    const w = net.width, h = net.height, s = Math.min(w / 1.0, h / (11 / 16)), ox = (w - s) / 2, oy = 0;
    const X = p => ox + p[0] * s, Y = p => oy + p[1] * s * (11 / 16);
    nx.clearRect(0, 0, w, h);
    const d = devicePixelRatio || 1, fr = D.frames[f];
    // module frames and labels
    nx.font = `${11 * d}px ${css('--display')}`; nx.textAlign = 'center';
    const labels = ['controller', 'latent 1', 'latent 2', '← push left', 'push right →'];
    for (let k = 0; k < nMod; k++) {
      const won = t > fr.race_t && fr.race_t >= 0 && actMods[fr.a] === k;
      const xs = members[k].map(n => X(pos[n])), ys = members[k].map(n => Y(pos[n]));
      const x0 = Math.min(...xs) - 12 * d, x1 = Math.max(...xs) + 12 * d, y0 = Math.min(...ys) - 12 * d, y1 = Math.max(...ys) + 12 * d;
      nx.strokeStyle = won ? C[roleVar[k]] : C['--rule']; nx.lineWidth = (won ? 2.5 : 1) * d;
      nx.beginPath(); nx.roundRect(x0, y0, x1 - x0, y1 - y0, 8 * d); nx.stroke();
      nx.fillStyle = won ? C[roleVar[k]] : C['--muted'];
      nx.fillText(labels[k] + (won ? '  ✓ race' : ''), (x0 + x1) / 2, y0 - 6 * d);
      // spikes so far this frame (share of the frame's activity = p̂)
      nx.fillStyle = C['--muted']; nx.font = `${10 * d}px ${css('--mono')}`;
      nx.fillText(`${modCount[k]} spikes`, (x0 + x1) / 2, y1 + 14 * d);
      nx.font = `${11 * d}px ${css('--display')}`;
    }
    nx.fillStyle = C['--muted']; nx.font = `${10 * d}px ${css('--mono')}`;
    const g0 = pos[members[0][0]], g1 = pos[members[0][35]];
    nx.fillText('θ →', X([(g0[0] + g1[0]) / 2, 0]), Y([0, 0.24 + 5 * 0.072 + 0.06]) + 8 * d);
    nx.save(); nx.translate(X([0.035, 0]), Y([0, 0.42])); nx.rotate(-Math.PI / 2); nx.fillText('θ̇ →', 0, 0); nx.restore();
    // every cross-module synapse, opacity by strength (sqrt so weak latent->action links stay visible)
    nx.lineWidth = 0.6 * d;
    for (const [j, i, T] of drawE) { nx.globalAlpha = 0.04 + 0.4 * Math.sqrt(T / maxT);
      nx.strokeStyle = C['--edge']; nx.beginPath(); nx.moveTo(X(pos[j]), Y(pos[j])); nx.lineTo(X(pos[i]), Y(pos[i])); nx.stroke(); }
    // walk hops
    nx.strokeStyle = C['--hop'];
    for (const [j, i, age] of hops) { nx.globalAlpha = Math.max(0, 1 - age / 4); nx.lineWidth = (2.2 - age * 0.4) * d;
      nx.beginPath(); nx.moveTo(X(pos[j]), Y(pos[j])); nx.lineTo(X(pos[i]), Y(pos[i])); nx.stroke(); }
    nx.globalAlpha = 1;
    // neurons
    for (let n = 0; n < N; n++) {
      const r = (3.2 + 3.5 * glow[n]) * d, col = C[roleVar[mod[n]]];
      nx.beginPath(); nx.arc(X(pos[n]), Y(pos[n]), r, 0, 6.2832);
      nx.globalAlpha = 0.28 + 0.72 * glow[n]; nx.fillStyle = col; nx.fill();
      if (glow[n] > 0.9) { nx.globalAlpha = 0.25; nx.beginPath(); nx.arc(X(pos[n]), Y(pos[n]), r * 2.2, 0, 6.2832); nx.fill(); }
      if (ring[n] > 0.05) { nx.globalAlpha = ring[n]; nx.strokeStyle = C['--ink']; nx.lineWidth = 1.5 * d;
        nx.beginPath(); nx.arc(X(pos[n]), Y(pos[n]), r + 3 * d, 0, 6.2832); nx.stroke(); }
    }
    nx.globalAlpha = 1;
  }

  function drawPlant() {
    const w = plant.width, h = plant.height, d = devicePixelRatio || 1, fr = D.frames[f];
    const nxt = D.frames[Math.min(f + 1, F - 1)].obs, u = t / K;
    const x = fr.obs[0] + (nxt[0] - fr.obs[0]) * u, th = fr.obs[2] + (nxt[2] - fr.obs[2]) * u;
    px.clearRect(0, 0, w, h);
    const sx = w / 5.6, cx = w / 2 + x * sx, ty = h * 0.7;
    // track with the ±2.4 limits
    px.strokeStyle = C['--track']; px.lineWidth = 2 * d;
    px.beginPath(); px.moveTo(w / 2 - 2.4 * sx, ty); px.lineTo(w / 2 + 2.4 * sx, ty); px.stroke();
    px.fillStyle = C['--muted']; px.font = `${10 * d}px ${css('--mono')}`; px.textAlign = 'center';
    for (const v of [-2.4, 0, 2.4]) { px.fillRect(w / 2 + v * sx - d, ty - 5 * d, 2 * d, 10 * d); px.fillText(v.toFixed(1), w / 2 + v * sx, ty + 20 * d); }
    // ±12° failure wedge
    const L = 1.0 * sx * 1.5;
    px.globalAlpha = 0.12; px.fillStyle = C['--pole']; px.beginPath(); px.moveTo(cx, ty - 14 * d);
    px.arc(cx, ty - 14 * d, L, -Math.PI / 2 - 0.2094, -Math.PI / 2 + 0.2094); px.closePath(); px.fill(); px.globalAlpha = 1;
    // cart
    px.fillStyle = C['--cart']; px.beginPath(); px.roundRect(cx - 34 * d, ty - 24 * d, 68 * d, 20 * d, 4 * d); px.fill();
    // pole
    px.strokeStyle = C['--pole']; px.lineWidth = 7 * d; px.lineCap = 'round';
    px.beginPath(); px.moveTo(cx, ty - 14 * d); px.lineTo(cx + L * Math.sin(th), ty - 14 * d - L * Math.cos(th)); px.stroke();
    px.fillStyle = C['--panel']; px.beginPath(); px.arc(cx, ty - 14 * d, 3.5 * d, 0, 6.2832); px.fill();
    // action arrow once the race is decided
    const decided = fr.race_t >= 0 && t > fr.race_t;
    if (decided) { const dir = fr.a ? 1 : -1, ax = cx + dir * 44 * d;
      px.strokeStyle = C[fr.a ? '--actR' : '--actL']; px.fillStyle = px.strokeStyle; px.lineWidth = 3 * d;
      px.beginPath(); px.moveTo(ax, ty - 14 * d); px.lineTo(ax + dir * 26 * d, ty - 14 * d); px.stroke();
      px.beginPath(); px.moveTo(ax + dir * 34 * d, ty - 14 * d); px.lineTo(ax + dir * 24 * d, ty - 20 * d); px.lineTo(ax + dir * 24 * d, ty - 8 * d); px.fill(); }
    // readouts
    document.getElementById('rx').textContent = `${x >= 0 ? ' ' : ''}${x.toFixed(3)} m`;
    document.getElementById('rth').textContent = `${th >= 0 ? ' ' : ''}${(th * 180 / Math.PI).toFixed(2)}°`;
    document.getElementById('ra').textContent = decided ? (fr.a ? 'push right' : 'push left') : t < D.burn ? 'burn-in' : 'racing…';
    document.getElementById('rp').textContent = fr.P[1].toFixed(3);
    document.getElementById('frameLabel').textContent = `frame ${f + 1} / ${F}`;
    document.getElementById('rh').textContent = `${nHop}`;
    document.getElementById('rx2').textContent = `${nCross} (${nHop ? (100 * nCross / nHop).toFixed(0) : 0}%)`;
    document.getElementById('rla').textContent = `${nLA}`;
    document.getElementById('rin').textContent = `${nIn}`;
    document.getElementById('stepLabel').textContent = `step ${String(t).padStart(2, ' ')} / ${K}${t < D.burn ? ' · burn-in' : ''}`;
    document.getElementById('clock').textContent = `t = ${((f + u) * 0.02).toFixed(2)} s of ${(F * 0.02).toFixed(2)} s`;
  }

  function loop(ts) {
    if (lastTs === null) lastTs = ts;
    const dt = Math.min(0.1, (ts - lastTs) / 1000); lastTs = ts;
    if (playing) { acc += dt * speed; let n = Math.floor(acc); acc -= n; while (n-- > 0) stepOnce(); }
    drawNet(); drawPlant();
    requestAnimationFrame(loop);
  }

  playBtn.addEventListener('click', () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; });
  document.getElementById('speed').addEventListener('change', e => { speed = +e.target.value; hops.length = 0; });
  scrub.addEventListener('input', () => { f = +scrub.value; resetFrame(); });
  addEventListener('keydown', e => { if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); playBtn.click(); } });
  requestAnimationFrame(loop);
}).catch(() => document.body.insertAdjacentHTML('afterbegin',
  '<p style="padding:1em">Could not load the episode data: serve this folder over http (python -m http.server).</p>'));
