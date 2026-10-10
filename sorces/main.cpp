// Essential Mutation (EM) framework for GA-TSP
// Reproduces the methodology of:
//   Abu Doush, I. et al. "Essential mutation framework for solving
//   travelling salesman problem." Cluster Computing (2026) 29:49.
//
// Usage:
//   tsp [--datasets a280,berlin52,...] [--runs 30] [--gens 2000]
//       [--pop 300] [--elite 60] [--mrate 0.05] [--crate 0.8] [--erate 0.1]
//       [--data-dir ../data] [--results-dir ../results]
//
// Defaults reproduce the paper's Section 2.1 setup. For a quick sanity
// check use e.g.: tsp --datasets berlin52,eil51,st70 --runs 5 --gens 300

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <filesystem>

#include "TSPInstance.h"
#include "GeneticAlgorithm.h"
#include "Mutations.h"

namespace fs = std::filesystem;

struct Stats {
    double best = 0, mean = 0, stdev = 0, avgTimeMs = 0, avgBestGeneration = 0;
};

static Stats summarize(const std::vector<double>& bests, const std::vector<double>& times,
                        const std::vector<int>& bestGens) {
    Stats s;
    s.best = *std::min_element(bests.begin(), bests.end());
    s.mean = std::accumulate(bests.begin(), bests.end(), 0.0) / bests.size();
    double var = 0.0;
    for (double b : bests) var += (b - s.mean) * (b - s.mean);
    var /= (bests.size() > 1 ? (bests.size() - 1) : 1);
    s.stdev = std::sqrt(var);
    s.avgTimeMs = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    s.avgBestGeneration = std::accumulate(bestGens.begin(), bestGens.end(), 0.0) / bestGens.size();
    return s;
}

static std::vector<std::string> splitCsv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(item);
    return out;
}

int main(int argc, char** argv) {
    // ---- defaults (match the paper's Section 2.1 experimental setup) ----
    std::vector<std::string> datasets = {"berlin52", "eil51", "st70"};
    int runs = 30;
    int generations = 2000;
    int popSize = 300;
    int eliteSize = 60;
    double mutationRate = 0.05;
    double crossoverRate = 0.8;
    double essentialRate = 0.1;
    std::string dataDir = "../data";
    std::string resultsDir = "../results";
    std::vector<MutationType> operators = {MutationType::EM, MutationType::SM,
                                            MutationType::RM, MutationType::CIM};

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (arg == "--datasets") datasets = splitCsv(next());
        else if (arg == "--runs") runs = std::stoi(next());
        else if (arg == "--gens") generations = std::stoi(next());
        else if (arg == "--pop") popSize = std::stoi(next());
        else if (arg == "--elite") eliteSize = std::stoi(next());
        else if (arg == "--mrate") mutationRate = std::stod(next());
        else if (arg == "--crate") crossoverRate = std::stod(next());
        else if (arg == "--erate") essentialRate = std::stod(next());
        else if (arg == "--data-dir") dataDir = next();
        else if (arg == "--results-dir") resultsDir = next();
        else if (arg == "--only-em") operators = {MutationType::EM};
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    fs::create_directories(resultsDir);

    std::cout << "Config: runs=" << runs << " generations=" << generations
              << " pop=" << popSize << " elite=" << eliteSize
              << " mutationRate=" << mutationRate << " crossoverRate=" << crossoverRate
              << " essentialRate=" << essentialRate << "\n\n";

    // Summary CSV across all datasets/operators (mirrors paper's Table 3 + 4)
    std::ofstream summary(resultsDir + "/summary.csv");
    summary << "dataset,operator,best,mean,stdev,avg_time_ms,avg_best_generation\n";

    for (const auto& dsName : datasets) {
        std::string path = dataDir + "/" + dsName + ".tsp";
        if (!fs::exists(path)) {
            std::cerr << "[skip] " << path << " not found. Place the TSPLIB file in " << dataDir << "\n";
            continue;
        }

        TSPInstance inst;
        try {
            inst.load(path);
        } catch (const std::exception& e) {
            std::cerr << "[error] failed to load " << path << ": " << e.what() << "\n";
            continue;
        }

        std::cout << "=== " << dsName << " (n=" << inst.dimension << ") ===\n";

        // Per-dataset detailed run log
        std::ofstream detail(resultsDir + "/" + dsName + "_runs.csv");
        detail << "operator,run,best,elapsed_ms,best_generation\n";

        // Per-dataset convergence trace (best-so-far per generation) for the
        // first run of each operator, matching Fig. 7 of the paper.
        std::ofstream conv(resultsDir + "/" + dsName + "_convergence.csv");
        conv << "generation";
        for (auto op : operators) conv << "," << mutationName(op);
        conv << "\n";
        std::map<MutationType, std::vector<double>> convTraces;

        for (auto op : operators) {
            std::vector<double> bests, times;
            std::vector<int> bestGens;
            for (int r = 0; r < runs; ++r) {
                GAConfig cfg;
                cfg.populationSize = popSize;
                cfg.eliteSize = eliteSize;
                cfg.mutationRate = mutationRate;
                cfg.crossoverRate = crossoverRate;
                cfg.essentialRate = essentialRate;
                cfg.generations = generations;
                cfg.mutation = op;
                cfg.seed = static_cast<unsigned int>(r + 1);

                GeneticAlgorithm ga(inst, cfg);
                GAResult res = ga.run();

                bests.push_back(res.best);
                times.push_back(res.elapsedMs);
                bestGens.push_back(res.bestGeneration);
                detail << mutationName(op) << "," << r << "," << res.best << ","
                       << res.elapsedMs << "," << res.bestGeneration << "\n";

                if (r == 0) convTraces[op] = res.convergence;
            }

            Stats s = summarize(bests, times, bestGens);
            std::cout << "  " << mutationName(op)
                      << " best=" << s.best << " mean=" << s.mean
                      << " stdev=" << s.stdev << " avgTime(ms)=" << s.avgTimeMs
                      << " avgBestGen=" << s.avgBestGeneration << "/" << generations << "\n";

            summary << dsName << "," << mutationName(op) << "," << s.best << ","
                    << s.mean << "," << s.stdev << "," << s.avgTimeMs << ","
                    << s.avgBestGeneration << "\n";
        }

        // Write convergence CSV (align by shortest trace length, in case gens differ)
        size_t maxLen = 0;
        for (auto& kv : convTraces) maxLen = std::max(maxLen, kv.second.size());
        for (size_t g = 0; g < maxLen; ++g) {
            conv << g;
            for (auto op : operators) {
                conv << ",";
                auto it = convTraces.find(op);
                if (it != convTraces.end() && g < it->second.size()) conv << it->second[g];
            }
            conv << "\n";
        }

        std::cout << "\n";
    }

    std::cout << "Results written to " << resultsDir << "/summary.csv "
              << "(plus per-dataset *_runs.csv and *_convergence.csv)\n";
    return 0;
}
