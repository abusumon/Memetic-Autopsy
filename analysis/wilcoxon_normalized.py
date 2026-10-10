"""Table 2: aggregate Wilcoxon signed-rank tests on NORMALISED optimality gaps (Holm-corrected).
Input: ../results/ablation_master_table.csv (the original ablation output). Needs: scipy."""
import csv
from scipy.stats import wilcoxon
rows = list(csv.reader(open('../results/ablation_master_table.csv', newline='')))
data = rows[2:]
conds = ['EM_only', 'LS_only', 'EM_LS', 'EM_SE_LS', 'EM_LS_SE']   # EM_LS_SE = SE passes before AND after LS
opt = {'eil51':426,'berlin52':7542,'st70':675,'kroA100':21282,'bier127':118282,'ch150':6528,'pr152':73682,'rat195':2323,
       'kroA200':29368,'pr226':80369,'a280':2579,'pr299':48191,'lin318':42029,'pr439':107217,'rat575':6773,'u724':41910,'rat783':8806}
gap = {c: [] for c in conds}
for r in data:
    for k, c in enumerate(conds):
        avg = float(r[1 + 4 * k + 1]); gap[c].append((avg - opt[r[0]]) / opt[r[0]] * 100)
pairs = [('EM_only','EM_LS'),('LS_only','EM_LS'),('EM_LS_SE','EM_LS'),('EM_only','EM_SE_LS'),('LS_only','EM_SE_LS'),('EM_LS_SE','EM_SE_LS')]
res = []
for m, rf in pairs:
    d = [a - b for a, b in zip(gap[m], gap[rf])]
    nz = [x for x in d if abs(x) > 1e-12]
    w = sum(x < -1e-12 for x in d); l = sum(x > 1e-12 for x in d)
    res.append([m, rf, w, len(d) - w - l, l, wilcoxon(nz).pvalue if nz else 1.0])
order = sorted(range(len(res)), key=lambda i: res[i][5]); prev = 0; ph = [0] * len(res)
for rank, i in enumerate(order):
    prev = max(prev, min(1, (len(res) - rank) * res[i][5])); ph[i] = prev
for i, r in enumerate(res):
    print(f"{r[0]:9s} vs {r[1]:9s} W/T/L={r[2]}/{r[3]}/{r[4]}  p={r[5]:.3g}  p_Holm={ph[i]:.3g}")
