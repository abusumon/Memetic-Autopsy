"""Table 3: plain GA ('None') vs EM-only, paired Wilcoxon over 20 seeds per instance, Holm over 4 instances.
IMPORTANT: runs were resumed in chunks, so the 'run' column restarts; FILE ORDER == seed order 1..20."""
import csv, collections, statistics as st
from scipy.stats import wilcoxon
R = '../results/ablation_reproduction/'
D = collections.defaultdict(list)
for f in ['p1_eil51.csv', 'p1_kroA100.csv', 'p2_none.csv', 'p2_emonly.csv']:
    for r in csv.DictReader(open(R + f)):
        D[(r['condition'], r['instance'])].append(float(r['best']))
out = []
for inst in ['eil51', 'berlin52', 'st70', 'kroA100']:
    a, b = D[('None', inst)], D[('EM_only', inst)]
    assert len(a) == len(b) == 20, (inst, len(a), len(b))
    d = [x - y for x, y in zip(a, b)]
    out.append([inst, st.mean(a), st.mean(b), (st.mean(a) - st.mean(b)) / st.mean(a) * 100,
                sum(t > 0 for t in d), sum(t < 0 for t in d), wilcoxon(d).pvalue])
order = sorted(range(4), key=lambda i: out[i][-1]); prev = 0; ph = [0] * 4
for rank, i in enumerate(order):
    prev = max(prev, min(1, (4 - rank) * out[i][-1])); ph[i] = prev
for i, r in enumerate(out):
    print(f"{r[0]:9s} plain={r[1]:.2f} EM={r[2]:.2f} improvement={r[3]:+.2f}% W/L={r[4]}/{r[5]} p={r[6]:.3g} p_Holm={ph[i]:.3g}")
