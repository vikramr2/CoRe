import glob, json, sys
D = "/tmp/claude-26504/-home-vr9-map-stdp/de3951c9-83d8-41b5-ab74-0f748b322531/scratchpad/lifdiag/"
rows = []
for f in sorted(glob.glob(D + "out/an_*.txt")):
    for l in open(f):
        if l.startswith("{"):
            rows.append(json.loads(l))
order = ["branch", "lif_det", "lif_s1", "lif_s2", "lif_s4", "lif_esc2", "lif_chi6.5", "lif_chi12.35"]
rows.sort(key=lambda d: (d["task"], d["W"], order.index(d["op"])))
cols = sys.argv[1].split(",")
print("task W op | " + " ".join(f"{c[:14]:>14}" for c in cols))
for d in rows:
    print(f"{d['task'][:5]} {d['W'][:4]} {d['op']:>12} | " + " ".join(f"{d.get(c, float('nan')):14.3f}" for c in cols))
