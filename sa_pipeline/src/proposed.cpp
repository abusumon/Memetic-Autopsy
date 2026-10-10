// Driver for the thesis's actual proposed algorithm (EM + advisor's
// SA-based local search + SE) - and ONLY that. No ablation toggles (see
// ablation.cpp for that), no reproduction of the Abu Doush et al. EM-only
// baseline experiment (see main.cpp for that, kept unchanged/untouched).
//
// Sweeps multiple Similarity Elimination thresholds (theta) in one run,
// per your request - since a single fixed theta risks landing on a value
// that happens to underperform for a given dataset, running several and
// reporting all of them means you always have a fallback/comparison point
// rather than being stuck with one possibly-unlucky choice. Everything
// else about the algorithm stays fixed across the sweep (EM on, SA-LS on,
// EAX crossover, same GA constants) - theta is the only thing varied here.
//
// Usage:
//   proposed [--datasets berlin52,eil51,st70] [--runs 10] [--gens 2000]
//            [--pop 300] [--elite 60] [--mrate 0.05] [--crate 0.8]
//            [--erate 0.1] [--thetas 0.3,0.5,0.7]
//            [--sa-iters 100] [--sa-t0 0.025] [--sa-gamma 0.99]
//            [--time-budget-min 4] [--data-dir ../data] [--results-dir ../results]
//
// Defaults for generations/population/elite/mutation/crossover/essential-
// rate match the EM paper's setup (same values main.cpp already used).
// `--runs` was lowered from the EM paper's 30 to 10: with the SA local
// search now running per-offspring per-generation (unlike the old cheap
// deterministic LS the EM paper's defaults were tuned around), 30 runs x 3
// thetas would take multiple hours even on small instances.
//
// `--time-budget-min` (default 4) is a HARD per-run wall-clock cap, not a
// hand-tuned generation count - the profiled timing this default was
// chosen from (single core: ~79ms/gen at n=52, ~322ms/gen at n=200) shows
// gens=2000 comfortably finishes well under 4 minutes for small instances
// like the default berlin52/eil51/st70, so the cap won't normally trigger
// there. What it actually guarantees: with the default 3 thetas x 10 runs,
// total time per dataset is bounded at 3 x 10 x 4min = 120 minutes
// REGARDLESS of instance size - so pointing this at a much bigger instance
// later won't silently balloon past that, it'll just stop early (with
// whatever best solution it found so far) instead of running for hours.
// If you point this at instances noticeably bigger than ~200 nodes, you'll
// likely see the cap trigger before all `generations` complete - that's
// working as intended, not a bug; raise --time-budget-min deliberately if
// you want to allow more time for those.

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
#include "ProposedGA.h"

namespace fs = std::filesystem;

struct Stats {
    double best = 0, mean = 0, stdev = 0, avgTimeMs = 0, avgBestGeneration = 0;
    // Elimination-rate instrumentation: % of offspring slots replaced with a
    // fresh random tour per generation, averaged across runs, pass 1
    // (pre-LS) and pass 2 (post-LS) reported separately since a tour that
    // survives pass 1 can still get nuked by pass 2 after SA-LS moves it.
    double avgElimRatePass1Pct = 0, avgElimRatePass2Pct = 0;
    double avgGensCompleted = 0;
    // EAXCrossover.h's subtour-merge fallback rate: how often, per
    // generation on average, the fast candidate-list-based merge search
    // found nothing and had to fall back to an exhaustive nearest-pair
    // search (see EAXCrossover.h). A high rate here is the direct signal
    // that instance size is overwhelming the fixed k=20 neighbor-list
    // candidate search.
    double avgMergeFallbacksPerGen = 0;
    // Average number of times the stagnation-escape mechanism fired per
    // run (see ProposedGAConfig::stagnationLimit) - roughly, how many
    // times a run got stuck long enough to trigger a reheat + partial
    // RNN restart.
    double avgStagnationTriggers = 0;
};

static Stats summarize(const std::vector<double>& bests, const std::vector<double>& times,
                        const std::vector<int>& bestGens,
                        const std::vector<long long>& elim1s, const std::vector<long long>& elim2s,
                        const std::vector<int>& gensCompletedVec, int popSize,
                        const std::vector<long long>& mergeFallbacks,
                        const std::vector<long long>& stagnationTriggersVec) {
    Stats s;
    s.best = *std::min_element(bests.begin(), bests.end());
    s.mean = std::accumulate(bests.begin(), bests.end(), 0.0) / bests.size();
    double var = 0.0;
    for (double b : bests) var += (b - s.mean) * (b - s.mean);
    var /= (bests.size() > 1 ? (bests.size() - 1) : 1);
    s.stdev = std::sqrt(var);
    s.avgTimeMs = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    s.avgBestGeneration = std::accumulate(bestGens.begin(), bestGens.end(), 0.0) / bestGens.size();

    // Per-run elimination rate = eliminated / (generationsCompleted * popSize),
    // i.e. the fraction of offspring slots that got replaced with noise on
    // an average generation of that run - then averaged across runs.
    double rate1Sum = 0.0, rate2Sum = 0.0, fallbackPerGenSum = 0.0;
    for (size_t r = 0; r < elim1s.size(); ++r) {
        double denom = static_cast<double>(gensCompletedVec[r]) * static_cast<double>(popSize);
        rate1Sum += denom > 0 ? (static_cast<double>(elim1s[r]) / denom) * 100.0 : 0.0;
        rate2Sum += denom > 0 ? (static_cast<double>(elim2s[r]) / denom) * 100.0 : 0.0;
        double genDenom = static_cast<double>(gensCompletedVec[r]);
        fallbackPerGenSum += genDenom > 0 ? (static_cast<double>(mergeFallbacks[r]) / genDenom) : 0.0;
    }
    s.avgElimRatePass1Pct = rate1Sum / elim1s.size();
    s.avgElimRatePass2Pct = rate2Sum / elim2s.size();
    s.avgGensCompleted = std::accumulate(gensCompletedVec.begin(), gensCompletedVec.end(), 0.0) / gensCompletedVec.size();
    s.avgMergeFallbacksPerGen = fallbackPerGenSum / mergeFallbacks.size();
    s.avgStagnationTriggers = std::accumulate(stagnationTriggersVec.begin(), stagnationTriggersVec.end(), 0.0)
                               / stagnationTriggersVec.size();
    return s;
}

static std::vector<std::string> splitCsv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(item);
    return out;
}

static std::vector<double> splitCsvDouble(const std::string& s) {
    std::vector<double> out;
    for (auto& tok : splitCsv(s)) out.push_back(std::stod(tok));
    return out;
}

int main(int argc, char** argv) {
    std::vector<std::string> datasets = {"berlin52", "eil51", "st70"};
    int runs = 10;              // lowered from the EM paper's 30 - see the note on
                                 // --time-budget-min below for why
    int generations = 2000;
    int popSize = 300;
    int eliteSize = 60;
    double mutationRate = 0.05;
    double crossoverRate = 0.8;
    double essentialRate = 0.1;
    std::vector<double> thetas = {0.3, 0.5, 0.7};
    std::string dataDir = "data";
    std::string resultsDir = "results";
    AnnealingLSConfig saCfg;
    saCfg.iterations = 100; // see ProposedGA.h's performance note - much lower than the
                            // paper's standalone 1000, deliberately, for GA-embedded use
    double timeBudgetMin = 4.0; // per SINGLE run() call, not the whole sweep - see below
    int stagnationLimit = 150;
    double reheatFraction = 0.5;
    double restartFraction = 0.1;

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
        else if (arg == "--thetas") thetas = splitCsvDouble(next());
        else if (arg == "--sa-iters") saCfg.iterations = std::stoi(next());
        else if (arg == "--sa-t0") saCfg.initialTemperature = std::stod(next());
        else if (arg == "--sa-gamma") saCfg.coolingCoefficient = std::stod(next());
        else if (arg == "--time-budget-min") timeBudgetMin = std::stod(next());
        else if (arg == "--stagnation-limit") stagnationLimit = std::stoi(next());
        else if (arg == "--reheat-frac") reheatFraction = std::stod(next());
        else if (arg == "--restart-frac") restartFraction = std::stod(next());
        else if (arg == "--data-dir") dataDir = next();
        else if (arg == "--results-dir") resultsDir = next();
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    fs::create_directories(resultsDir);

    std::cout << "Proposed algorithm config: runs=" << runs << " generations=" << generations
              << " pop=" << popSize << " elite=" << eliteSize
              << " mutationRate=" << mutationRate << " crossoverRate=" << crossoverRate
              << " essentialRate=" << essentialRate
              << " sa.iterations=" << saCfg.iterations << " sa.T0=" << saCfg.initialTemperature
              << " sa.gamma=" << saCfg.coolingCoefficient
              << " timeBudgetMin=" << timeBudgetMin << " (per single run)"
              << " stagnationLimit=" << stagnationLimit
              << " reheatFraction=" << reheatFraction
              << " restartFraction=" << restartFraction << "\n";
    std::cout << "Thetas swept: ";
    for (double t : thetas) std::cout << t << " ";
    std::cout << "\n\n";

    std::ofstream summary(resultsDir + "/proposed_summary.csv");
    summary << "dataset,theta,best,mean,stdev,avg_time_ms,avg_best_generation,"
               "avg_gens_completed,avg_elim_rate_pass1_pct,avg_elim_rate_pass2_pct\n";

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

        std::ofstream detail(resultsDir + "/" + dsName + "_proposed_runs.csv");
        detail << "theta,run,best,elapsed_ms,best_generation,generations_completed,"
                  "eliminated_pass1,eliminated_pass2,elim_rate_pass1_pct,elim_rate_pass2_pct\n";

        for (double theta : thetas) {
            std::vector<double> bests, times;
            std::vector<int> bestGens, gensCompletedVec;
            std::vector<long long> elim1s, elim2s, mergeFallbacks, stagnationTriggersVec;
            std::vector<int> bestTourOverall;
            double bestOverallLen = 1e300;
            long long invalidTourCount = 0;

            for (int r = 0; r < runs; ++r) {
                ProposedGAConfig cfg;
                cfg.populationSize = popSize;
                cfg.eliteSize = eliteSize;
                cfg.mutationRate = mutationRate;
                cfg.crossoverRate = crossoverRate;
                cfg.essentialRate = essentialRate;
                cfg.similarityThreshold = theta;
                cfg.generations = generations;
                cfg.annealingConfig = saCfg;
                cfg.maxWallTimeMs = timeBudgetMin * 60.0 * 1000.0;
                cfg.stagnationLimit = stagnationLimit;
                cfg.reheatFraction = reheatFraction;
                cfg.restartFraction = restartFraction;
                cfg.seed = static_cast<unsigned int>(r + 1);

                ProposedGA ga(inst, cfg);
                GAResult res = ga.run();

                // Live per-run progress line, flushed immediately - without
                // this, a single dataset+theta combo prints NOTHING until
                // all `runs` complete, which with a high --sa-iters (or a
                // large instance near the wall-clock cap) can be 30-40+
                // minutes of apparent silence that looks identical to a
                // hang. std::endl forces a flush (unlike "\n" alone, which
                // can sit in a buffer when stdout is piped, e.g. through
                // Tee-Object) so this is visible in real time either way.
                std::cout << "  [" << dsName << " theta=" << theta << "] run "
                          << (r + 1) << "/" << runs << ": best=" << res.best
                          << " elapsed=" << (res.elapsedMs / 1000.0) << "s"
                          << " gens=" << res.generationsCompleted << "/" << generations
                          << std::endl;

                bests.push_back(res.best);
                times.push_back(res.elapsedMs);
                bestGens.push_back(res.bestGeneration);
                gensCompletedVec.push_back(res.generationsCompleted);
                elim1s.push_back(res.eliminatedPass1);
                elim2s.push_back(res.eliminatedPass2);
                mergeFallbacks.push_back(res.mergeFallbackCount);
                stagnationTriggersVec.push_back(res.stagnationTriggers);

                if (!isValidTour(res.bestTour, inst.dimension)) {
                    ++invalidTourCount;
                    std::cerr << "  [INVALID TOUR] " << dsName << " theta=" << theta
                              << " run=" << r << " - not a valid permutation!\n";
                } else if (res.best < bestOverallLen) {
                    bestOverallLen = res.best;
                    bestTourOverall = res.bestTour;
                }

                double denom = static_cast<double>(res.generationsCompleted) * static_cast<double>(popSize);
                double rate1 = denom > 0 ? (static_cast<double>(res.eliminatedPass1) / denom) * 100.0 : 0.0;
                double rate2 = denom > 0 ? (static_cast<double>(res.eliminatedPass2) / denom) * 100.0 : 0.0;

                detail << theta << "," << r << "," << res.best << ","
                       << res.elapsedMs << "," << res.bestGeneration << ","
                       << res.generationsCompleted << "," << res.eliminatedPass1 << ","
                       << res.eliminatedPass2 << "," << rate1 << "," << rate2 << "\n";
                detail.flush(); // so the per-dataset CSV is also readable mid-run, not just at the end
            }

            Stats s = summarize(bests, times, bestGens, elim1s, elim2s, gensCompletedVec, popSize, mergeFallbacks, stagnationTriggersVec);
            std::cout << "  theta=" << theta
                      << " best=" << s.best << " mean=" << s.mean
                      << " stdev=" << s.stdev << " avgTime(ms)=" << s.avgTimeMs
                      << " avgBestGen=" << s.avgBestGeneration << "/" << generations
                      << " avgGensCompleted=" << s.avgGensCompleted
                      << " elimRate[pass1/pass2]=" << s.avgElimRatePass1Pct << "%/"
                      << s.avgElimRatePass2Pct << "%"
                      << " avgMergeFallbacks/gen=" << s.avgMergeFallbacksPerGen
                      << " avgStagnationTriggers=" << s.avgStagnationTriggers
                      << " invalidTours=" << invalidTourCount << "/" << runs << "\n";

            summary << dsName << "," << theta << "," << s.best << "," << s.mean << ","
                    << s.stdev << "," << s.avgTimeMs << "," << s.avgBestGeneration << ","
                    << s.avgGensCompleted << "," << s.avgElimRatePass1Pct << ","
                    << s.avgElimRatePass2Pct << "\n";

            // Dump the best valid tour found across this theta's runs, for
            // external inspection / cross-check against your own
            // validate.cpp if you want a second opinion. One city index
            // per line, plus the computed length as a trailing comment.
            if (!bestTourOverall.empty()) {
                std::ofstream tourOut(resultsDir + "/" + dsName + "_theta" + std::to_string(theta) + "_besttour.txt");
                tourOut << "# length=" << bestOverallLen << " n=" << inst.dimension << "\n";
                for (int c : bestTourOverall) tourOut << c << "\n";
            }
        }
        std::cout << "\n";
    }

    std::cout << "Results written to " << resultsDir << "/proposed_summary.csv "
              << "(plus per-dataset *_proposed_runs.csv)\n";
    return 0;
}