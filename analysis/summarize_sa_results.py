"""Summarise the final SA-pipeline runs (../results/sa_final_thesis_runs/*_proposed_runs.csv).
Excludes machine-stall runs (elapsed > 250 s with a 240 s cap; affects pr439 only), as in the paper.
Prints mean, optimality gap, elimination rate, median generations  ->  Table 7 / Figure 2."""
import csv, glob, os, statistics as st
opt = {'eil51':426,'berlin52':7542,'st70':675,'kroA100':21282,'bier127':118282,'ch150':6528,'pr152':73682,'rat195':2323,'kroA200':29368,
       'pr226':80369,'a280':2579,'pr299':48191,'lin318':42029,'pr439':107217,'rat575':6773,'u574':36905,'p654':34643,'u724':41910,'rat783':8806}
for f in sorted(glob.glob('../results/sa_final_thesis_runs/*_proposed_runs.csv')):
    k = os.path.basename(f).split('_')[0]
    rows = [r for r in csv.DictReader(open(f, encoding='utf-8', errors='ignore')) if float(r['elapsed_ms']) < 250000]
    b = [float(r['best']) for r in rows]
    print(f"{k:9s} runs={len(rows):2d} mean={st.mean(b):10.1f} gap={(st.mean(b)-opt[k])/opt[k]*100:6.2f}%  "
          f"elim={st.mean(float(r['elim_rate_pass2_pct']) for r in rows):5.2f}%  gens_med={st.median(float(r['generations_completed']) for r in rows):.0f}")
