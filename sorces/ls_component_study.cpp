// Phase 5: which local-search operator(s) actually matter?
//
// Your LS pass combines four operators (2-opt, inversion, swap, insertion).
// This driver isolates each one individually, then compares against the
// full combination - all crossed with EM on/off, SE always OFF (this study
// is specifically about EM x LS-operator interaction, matching Phase 5's
// scope; SE is Phase 6's question).
//
// Reuses AblationGA.h completely unmodified: LocalSearchConfig already has
// per-operator toggles (useTwoOpt/useInversion/useSwap/useInsertion), so
// this file is just a driver, not new algorithm code.
//
// Variants (x2 for EM off/on = 10 conditions total):
//   twoOpt_only, inversion_only, swap_only, insertion_only, all_four
//
// Usage:
//   ls_component_study [--datasets a280,berlin52,...] [--runs 20] [--gens 2000]
//            [--pop 300] [--elite 60] [--mrate 0.05] [--crate 0.8]
//            [--erate 0.1] [--ls-window 20] [--ls-passes 2]
//            [--data-dir ../data] [--results-dir ../results]
//
// After this: look at ls_component_summary.csv, find the strongest
// individual operator(s), then (per the roadmap's suggestion) only bother
// testing combinations of the best 2-3 rather than every subset.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <filesystem>

#include "TSPInstance.h"
#include "AblationGA.h"

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

struct LSVariant {
    std::string name;
    bool twoOpt, inversion, swap, insertion;
};

int main(int argc, char** argv) {
    std::vector<std::string> datasets = {"berlin52", "eil51", "st70"};
    int runs = 20;
    int generations = 2000;
    int popSize = 300;
    int eliteSize = 60;
    double mutationRate = 0.05;
    double crossoverRate = 0.8;
    double essentialRate = 0.1;
    std::string dataDir = "../data";
    std::string resultsDir = "../results";
    LocalSearchConfig baseLsCfg; // window/passes shared across variants

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
        else if (arg == "--ls-window") baseLsCfg.neighborhoodWindow = std::stoi(next());
        else if (arg == "--ls-passes") baseLsCfg.maxPasses = std::stoi(next());
        else if (arg == "--data-dir") dataDir = next();
        else if (arg == "--results-dir") resultsDir = next();
        else { std::cerr << "Unknown argument: " << arg << "\n"; return 1; }
    }

    fs::create_directories(resultsDir);

    std::vector<LSVariant> lsVariants = {
        //           name              2opt   inv    swap   ins
        {"twoOpt_only",   true,  false, false, false},
        {"inversion_only",false, true,  false, false},
        {"swap_only",     false, false, true,  false},
        {"insertion_only",false, false, false, true },
        {"all_four",      true,  true,  true,  true },
    };

    std::cout << "LS component study: runs=" << runs << " generations=" << generations
              << " pop=" << popSize << " elite=" << eliteSize
              << " lsWindow=" << baseLsCfg.neighborhoodWindow
              << " lsPasses=" << baseLsCfg.maxPasses << "\n"
              << "Variants: twoOpt_only, inversion_only, swap_only, insertion_only, all_four"
              << " (each x EM off/on, SE always off)\n\n";

    std::ofstream summary(resultsDir + "/ls_component_summary.csv");
    summary << "dataset,condition,ls_variant,em,best,mean,stdev,avg_time_ms,avg_best_generation\n";

    for (const auto& dsName : datasets) {
        std::string path = dataDir + "/" + dsName + ".tsp";
        if (!fs::exists(path)) {
            std::cerr << "[skip] " << path << " not found. Place the TSPLIB file in " << dataDir << "\n";
            continue;
        }
        TSPInstance inst;
        try { inst.load(path); }
        catch (const std::exception& e) {
            std::cerr << "[error] failed to load " << path << ": " << e.what() << "\n";
            continue;
        }

        std::cout << "=== " << dsName << " (n=" << inst.dimension << ") ===\n";

        std::ofstream detail(resultsDir + "/" + dsName + "_ls_component_runs.csv");
        detail << "condition,run,best,elapsed_ms,best_generation\n";

        for (const auto& variant : lsVariants) {
            for (bool em : {false, true}) {
                std::string condName = variant.name + (em ? "_EM" : "_noEM");
                std::vector<double> bests, times;
                std::vector<int> bestGens;

                for (int r = 0; r < runs; ++r) {
                    AblationGAConfig cfg;
                    cfg.populationSize = popSize;
                    cfg.eliteSize = eliteSize;
                    cfg.mutationRate = mutationRate;
                    cfg.crossoverRate = crossoverRate;
                    cfg.essentialRate = essentialRate;
                    cfg.generations = generations;
                    cfg.lsConfig = baseLsCfg;
                    cfg.lsConfig.useTwoOpt = variant.twoOpt;
                    cfg.lsConfig.useInversion = variant.inversion;
                    cfg.lsConfig.useSwap = variant.swap;
                    cfg.lsConfig.useInsertion = variant.insertion;
                    cfg.seed = static_cast<unsigned int>(r + 1); // same seeding convention as ablation.cpp
                    cfg.useEssentialMutation = em;
                    cfg.useLocalSearch = true; // always on - we're comparing WHICH operators, not LS vs no-LS
                    cfg.useSEbeforeLS = false;
                    cfg.useSEafterLS = false;

                    AblationGA ga(inst, cfg);
                    GAResult res = ga.run();

                    bests.push_back(res.best);
                    times.push_back(res.elapsedMs);
                    bestGens.push_back(res.bestGeneration);
                    detail << condName << "," << r << "," << res.best << ","
                           << res.elapsedMs << "," << res.bestGeneration << "\n";
                }

                Stats s = summarize(bests, times, bestGens);
                std::cout << "  " << condName
                          << " best=" << s.best << " mean=" << s.mean
                          << " stdev=" << s.stdev << " avgTime(ms)=" << s.avgTimeMs << "\n";

                summary << dsName << "," << condName << "," << variant.name << "," << (em ? "1" : "0") << ","
                        << s.best << "," << s.mean << "," << s.stdev << ","
                        << s.avgTimeMs << "," << s.avgBestGeneration << "\n";
            }
        }
        std::cout << "\n";
    }

    std::cout << "Results written to " << resultsDir << "/ls_component_summary.csv "
              << "(plus per-dataset *_ls_component_runs.csv)\n"
              << "Next: identify the strongest individual operator(s), then only test\n"
              << "combinations of the best 2-3 rather than every subset.\n";
    return 0;
}
