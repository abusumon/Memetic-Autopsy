#!/bin/bash
# Build:  g++ -std=c++17 -O2 -fopenmp -Iinclude -o driver driver.cpp
# Condition names (driver -> paper): None = plain GA | EM_only | LS_only | EM_LS |
#   EM_SE_LS = EM+SE+LS (SE pass 1 only) | EM_LS_SE_both = EM+LS+SE (SE before AND after LS)
# Seeds: run r uses seed r+1, shared across conditions. Defaults: pop 300, mutation 0.05, crossover 0.8, essential 0.1,
# theta 0.5, random-shuffle initialisation, random-immigrant SE replacement. The main ablation used --gens 1000.
cd "$(dirname "$0")"; mkdir -p out; ln -sfn ../data data
# --- A. Verification against results/ablation_master_table.csv (must match exactly) ---
./driver --datasets eil51   --conds EM_only,EM_SE_LS,EM_LS_SE_both --runs 20 --gens 1000 --out out/p1_eil51.csv
./driver --datasets kroA100 --conds EM_only                         --runs 20 --gens 1000 --out out/p1_kroA100.csv
#   expected (mean / std / mean convergence gen): eil51 EM_only 452.00/7.26/292.35 | EM_SE_LS 426.00/0.00/67.60
#   | EM_LS_SE 426.05/0.22/81.75 | kroA100 EM_only 33195.90/1055.26/987.80
# --- B. Plain-GA control for Table 3 ---
./driver --datasets eil51,berlin52,st70,kroA100 --conds None    --runs 20 --gens 1000 --out out/p2_none.csv
./driver --datasets berlin52,st70               --conds EM_only --runs 20 --gens 1000 --out out/p2_emonly.csv
# --- C. Full 17-instance ablation (slow: hours to days on one core; parallelise per instance) ---
# for inst in eil51 berlin52 st70 kroA100 bier127 ch150 pr152 rat195 kroA200 pr226 a280 pr299 lin318 pr439 rat575 u724 rat783; do
#   ./driver --datasets $inst --conds EM_only,LS_only,EM_LS,EM_SE_LS,EM_LS_SE_both --runs 20 --gens 1000 --out out/ablation_$inst.csv
# done
# --- D. NOT RUN (future work in the paper): SE theta / replacement sweep in this framework ---
# ./driver --datasets ch150 --conds EM_LS_SE_both --runs 20 --gens 1000 --theta 0.85 --repl db --tag _t085_db --out out/sweep_ch150.csv
#   (--repl ri = random immigrant (original); --repl db = double-bridge of the offspring)
# qrun.py reads a queue file (lines: out|instance|cond|runs|gens|theta|repl|tag) and resumes interrupted jobs: python3 qrun.py 270
