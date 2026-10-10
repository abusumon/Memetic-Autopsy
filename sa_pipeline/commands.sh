#!/bin/bash
# Build:  g++ -std=c++17 -O2 -fopenmp -Iinclude -o proposed src/proposed.cpp
#         g++ -std=c++17 -O2 -fopenmp -DRANDOM_INIT -Iinclude -o proposed_rand src/proposed.cpp   # random-init control
# Defaults (see results/sa_final_thesis_runs/run_log.txt): pop 300, mutation 0.05, crossover 0.8, essential 0.1,
#   sa.iterations 100, stagnation limit 150, 4-minute wall-clock cap per run.
cd "$(dirname "$0")"; ln -sfn ../data data
# --- E. Final runs behind Table 7 / Figure 2 (original outputs: ../results/sa_final_thesis_runs) ---
./proposed --datasets eil51,berlin52,st70,kroA100,bier127,ch150,pr152,rat195,kroA200,pr226,a280,pr299,lin318,pr439,u574,rat575,p654,u724,rat783 \
           --runs 10 --gens 2000 --thetas 0.85 --time-budget-min 4 --data-dir data --results-dir results_final
# --- F. theta sweep behind Table 6 (SE off == theta 1.5) ---
./proposed --datasets st70,kroA100 --runs 5 --gens 2000 --thetas 0.5,0.7,0.85,1.5 --data-dir data --results-dir res_rnn
./proposed --datasets ch150        --runs 5 --gens 2000 --thetas 0.5,0.7,0.85,1.5 --data-dir data --results-dir res_a
./proposed --datasets kroA200      --runs 3 --gens 2000 --thetas 0.5,0.85,1.5     --data-dir data --results-dir res_b
# --- G. initialisation control behind Table 8 (random shuffle instead of RNN, also in stagnation restarts) ---
./proposed_rand --datasets kroA100,ch150 --runs 3 --gens 2000 --thetas 0.85,1.5 --data-dir data --results-dir res_rand
./proposed_rand --datasets kroA200       --runs 3 --gens 2000 --thetas 0.85,1.5 --data-dir data --results-dir res_rand3
# Table 6/8 numbers are the "theta=... best= mean= stdev= elimRate[...]" lines in the logs and *_summary.csv files.
