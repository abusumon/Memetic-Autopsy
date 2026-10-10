# memetic-autopsy

**Code and data for _"The Operators That Didn't Matter: A Controlled Autopsy of Mutation Protection, Local Search, and Similarity Elimination in Memetic TSP Solvers"_**
Md Abu Bakkar Siddik Sumon · Mathematics Discipline, Khulna University · preprint, 2026

Which parts of a memetic GA for the Travelling Salesman Problem actually earn their keep? This repository ablates essential mutation (EM), local search (LS) and similarity elimination (SE) in a shared EAX-based GA on 17 TSPLIB instances (51–783 cities, 20 runs per condition) and gives everything needed to reproduce the numbers.

## Main findings

| | Finding | Evidence |
|---|---|---|
| 1 | Local search carries almost all the quality | LS only vs EM+LS: p<sub>Holm</sub> = 0.168; optimum reached in every run on 10 of 11 instances up to n = 280 |
| 2 | EM helps a GA *without* LS only at some sizes, and not once LS is present | kroA100: +4.7% (19 wins, 1 loss); no effect at n ≤ 70 |
| 3 | SE (θ = 0.5, two passes) hurts quality and speed | 0/2/15 wins/ties/losses, p<sub>Holm</sub> = 2.4×10⁻⁴; run time +29% to +181% |
| 4 | θ = 0.5 is unselective; a fixed θ means different things at different n | 96–99.99% of offspring eliminated; elimination share falls from 32% (n = 100) to 2.5% (n = 783) |

## Repository layout

| Folder | Contents |
|---|---|
| `ablation_2opt/` | EM/LS/SE ablation (EAX + 2-opt-style LS): headers, `driver.cpp`, resumable `qrun.py`, `commands.sh` |
| `sa_pipeline/` | SA-based pipeline (`proposed.cpp`), θ sweep and initialization control (`-DRANDOM_INIT`), `commands.sh` |
| `analysis/` | Python scripts for the statistics and summary tables (need `python3`, `scipy`) |
| `results/` | Raw per-run CSVs and logs, plus the original ablation master table |
| `data/` | TSPLIB instances used |
| `paper_build/`, `original_uploaded_sources/` | Paper build scripts and additional source files |

## Quick start

```bash
# build (g++ >= 9; OpenMP optional)
cd ablation_2opt && g++ -std=c++17 -O2 -fopenmp -Iinclude -o driver driver.cpp
cd ../sa_pipeline && g++ -std=c++17 -O2 -fopenmp -Iinclude -o proposed src/proposed.cpp
                     g++ -std=c++17 -O2 -fopenmp -DRANDOM_INIT -Iinclude -o proposed_rand src/proposed.cpp

# verify the ablation driver against the published numbers (expected values are in commands.sh)
cd ../ablation_2opt && ./commands.sh        # section A is the exact-reproduction check

# recompute the statistics
cd ../analysis && python3 wilcoxon_normalized.py && python3 em_vs_none.py && python3 summarize_sa_results.py
```

Condition names in the code map to the paper as follows: `None` = plain GA, `EM_only`, `LS_only`, `EM_LS`, `EM_SE_LS` = EM+SE+LS (SE pass 1 only), `EM_LS_SE_both` = EM+LS+SE (SE before and after LS). Run *r* uses seed *r*+1; the main ablation uses 1000 generations and random-shuffle initialization.

## Where each paper result comes from

| Paper item | Source |
|---|---|
| Tables 1–2, Figure 1a | `results/ablation_master_table.csv`, `analysis/wilcoxon_normalized.py` |
| Verification paragraph | `ablation_2opt/commands.sh` section A |
| Table 3 | `commands.sh` section B, `analysis/em_vs_none.py` |
| Table 6 (θ sweep), Table 8 (init control) | `sa_pipeline/commands.sh` sections F and G; `results/sa_theta_init/` |
| Table 7, Figure 2 | `results/sa_final_thesis_runs/`, `analysis/summarize_sa_results.py` |

## Notes and limitations

- The original ablation driver was lost; `driver.cpp` is a re-implementation that reproduces the reported means exactly.
- The full 17-instance ablation was not re-run (hours to days on one core). Re-running instructions are in `ablation_2opt/commands.sh`, section C.
- In the reproduction CSVs the `run` column restarts when a job was resumed; **file order is the seed order**.
- Not yet done: θ and replacement-policy sweep inside the 2-opt framework, per-run convergence tests, equal-time comparison of the two LS families, initialization control for n ≥ 300.

## Citation

```bibtex
@misc{sumon2026operators,
  author = {Sumon, Md Abu Bakkar Siddik},
  title  = {The Operators That Didn't Matter: A Controlled Autopsy of Mutation Protection, Local Search, and Similarity Elimination in Memetic TSP Solvers},
  year   = {2026},
  note   = {Preprint},
}
```

## License

MIT, see `LICENSE`. TSPLIB instances are distributed under their original terms (G. Reinelt, *ORSA J. Computing* 3(4), 1991).

## Contact

Md Abu Bakkar Siddik Sumon, abusumon1701@gmail.com
