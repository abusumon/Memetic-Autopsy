// validate.cpp - tour-validity verifier + standalone EM+SE+LS runner.
//
// WHY THIS EXISTS
//   The ablation shows EM_LS_SE under-performing the simpler LS variants.
//   Before treating that as a genuine algorithmic result, we must rule out
//   the mundane alternative: that the EAX crossover / similarity-elimination
//   / local-search steps are silently producing INVALID tours (a permutation
//   with missing or duplicated cities). An invalid tour still has a
//   computable "length", so a bug like that would masquerade as a bad-but-
//   plausible result. My earlier code read flagged EAX's component-extraction
//   fallbacks ("shouldn't happen") as the prime suspect.
//
// WHAT IT DOES
//   1. Component stress tests - hammers eaxCrossover() and LocalSearch.refine()
//      on random tours and asserts every output is a permutation of 0..n-1.
//   2. Full-pipeline check - runs the EXACT AblationGA EM_LS_SE loop
//      (EAX + Essential Mutation + 2x Similarity Elimination + Local Search)
//      as a standalone, instrumented reimplementation, asserting validity
//      after EVERY transformation, EVERY generation, and validating the final
//      best tour. This is also the "EM+SE+LS separately" runner you asked for:
//      it is independent of EnhancedGA.h / AblationGA.h and reuses only the
//      shared component headers, so it can be read/modified without touching
//      the experiment code.
//
// Usage:
//   validate [--datasets eil51,ch150,kroA200] [--gens 150] [--pop 300]
//            [--eax-trials 20000] [--ls-trials 3000] [--seed 1]
//            [--data-dir ../data]

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <numeric>
#include <algorithm>
#include <filesystem>

#include "TSPInstance.h"
#include "Mutations.h"              // essentialEvaluation, essentialMutation
#include "LocalSearch.h"            // LocalSearch
#include "SimilarityElimination.h" // precomputeEdgeSets, similarityElimination
#include "EAXCrossover.h"          // eaxCrossover

namespace fs = std::filesystem;

// ------------------------------------------------------------------ helpers

// Returns true iff `tour` is a permutation of {0,...,n-1}. On failure `why`
// explains the first problem found.
static bool isValidPermutation(const std::vector<int>& tour, int n, std::string& why) {
    if (static_cast<int>(tour.size()) != n) {
        why = "size=" + std::to_string(tour.size()) + " (expected " + std::to_string(n) + ")";
        return false;
    }
    std::vector<char> seen(n, 0);
    for (int c : tour) {
        if (c < 0 || c >= n) { why = "city " + std::to_string(c) + " out of range"; return false; }
        if (seen[c]) { why = "duplicate city " + std::to_string(c); return false; }
        seen[c] = 1;
    }
    return true; // size==n, in-range, no dupes => every city present exactly once
}

static std::vector<std::string> splitCsv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(item);
    return out;
}

// ------------------------------------------------------------ component tests

// Hammers EAX on random parent pairs. Returns number of invalid children.
static long stressEAX(const TSPInstance& inst, long trials, std::mt19937& rng) {
    int n = inst.dimension;
    std::vector<int> base(n);
    std::iota(base.begin(), base.end(), 0);
    long fails = 0, shown = 0;
    for (long t = 0; t < trials; ++t) {
        std::vector<int> p1 = base, p2 = base;
        std::shuffle(p1.begin(), p1.end(), rng);
        std::shuffle(p2.begin(), p2.end(), rng);
        auto ch = eaxCrossover(p1, p2, inst, rng);
        std::string why;
        for (const auto* c : {&ch.first, &ch.second}) {
            if (!isValidPermutation(*c, n, why)) {
                ++fails;
                if (shown++ < 3)
                    std::cout << "      [EAX INVALID] trial " << t << ": " << why << "\n";
            }
        }
    }
    return fails;
}

// Hammers LocalSearch.refine on random tours. Returns number of invalid results.
static long stressLS(const TSPInstance& inst, long trials, std::mt19937& rng) {
    int n = inst.dimension;
    LocalSearchConfig lsCfg; // defaults match proposed.cpp / ablation.cpp
    LocalSearch ls(inst, lsCfg);
    std::vector<int> base(n);
    std::iota(base.begin(), base.end(), 0);
    long fails = 0, shown = 0;
    for (long t = 0; t < trials; ++t) {
        std::vector<int> tour = base;
        std::shuffle(tour.begin(), tour.end(), rng);
        ls.refine(tour);
        std::string why;
        if (!isValidPermutation(tour, n, why)) {
            ++fails;
            if (shown++ < 3)
                std::cout << "      [LS INVALID] trial " << t << ": " << why << "\n";
        }
    }
    return fails;
}

// ------------------------------------------------------- full EM+SE+LS runner

struct PipelineReport {
    long invalidEvents = 0;   // total invalid tours seen across all checkpoints
    double best = 0.0;        // best tour length found
    bool bestValid = false;   // is the returned best tour a valid permutation?
};

// A standalone, instrumented copy of AblationGA's EM_LS_SE loop (all three
// mechanisms ON, EAX crossover ON - i.e. exactly the condition that produced
// the pasted numbers). Every intermediate tour is validity-checked.
static PipelineReport runCheckedPipeline(const TSPInstance& inst, int popSize,
                                         int generations, unsigned seed) {
    PipelineReport rep;
    std::mt19937 rng(seed);
    int n = inst.dimension;

    const double mutationRate = 0.05, crossoverRate = 0.8, essentialRate = 0.1, theta = 0.5;
    LocalSearchConfig lsCfg;
    LocalSearch ls(inst, lsCfg);

    auto check = [&](const std::vector<int>& tour, int gen, const char* step, int idx) {
        std::string why;
        if (!isValidPermutation(tour, n, why)) {
            if (rep.invalidEvents < 5)
                std::cout << "      [PIPELINE INVALID] gen " << gen << " step '" << step
                          << "' offspring " << idx << ": " << why << "\n";
            ++rep.invalidEvents;
        }
    };

    // --- init population ---
    std::vector<std::vector<int>> population(popSize);
    std::vector<int> base(n);
    std::iota(base.begin(), base.end(), 0);
    for (auto& ind : population) { ind = base; std::shuffle(ind.begin(), ind.end(), rng); }

    std::vector<double> fitness(popSize);
    for (int i = 0; i < popSize; ++i) fitness[i] = inst.tourLength(population[i]);

    auto tournament = [&](void) -> const std::vector<int>& {
        std::uniform_int_distribution<int> uidx(0, popSize - 1);
        int b = uidx(rng);
        for (int k = 1; k < 3; ++k) { int c = uidx(rng); if (fitness[c] < fitness[b]) b = c; }
        return population[b];
    };

    std::vector<int> bestTour = population[0];
    rep.best = fitness[0];

    for (int gen = 0; gen < generations; ++gen) {
        EssentialEdgeSet essentialEdges = essentialEvaluation(population, fitness, essentialRate);

        std::vector<std::vector<int>> parents;
        parents.reserve(popSize);
        for (int i = 0; i < popSize; ++i) parents.push_back(tournament());

        std::uniform_real_distribution<double> u01(0.0, 1.0);
        std::vector<std::vector<int>> offspring;
        offspring.reserve(popSize);
        for (int i = 0; i < popSize; i += 2) {
            const auto& p1 = parents[i];
            const auto& p2 = parents[(i + 1) % popSize];
            std::vector<int> c1, c2;
            if (u01(rng) < crossoverRate) {
                auto ch = eaxCrossover(p1, p2, inst, rng);   // <-- EAX (the suspect)
                c1 = std::move(ch.first); c2 = std::move(ch.second);
            } else { c1 = p1; c2 = p2; }
            essentialMutation(c1, mutationRate, essentialEdges, rng);
            essentialMutation(c2, mutationRate, essentialEdges, rng);
            offspring.push_back(std::move(c1));
            offspring.push_back(std::move(c2));
        }
        offspring.resize(popSize);
        for (int i = 0; i < popSize; ++i) check(offspring[i], gen, "post-crossover+EM", i);

        auto popEdgeSets = precomputeEdgeSets(population);

        similarityElimination(offspring, population, fitness, popEdgeSets, inst, theta, rng);
        for (int i = 0; i < popSize; ++i) check(offspring[i], gen, "post-SE1", i);

        for (int i = 0; i < popSize; ++i) ls.refine(offspring[i]);
        for (int i = 0; i < popSize; ++i) check(offspring[i], gen, "post-LS", i);

        similarityElimination(offspring, population, fitness, popEdgeSets, inst, theta, rng);
        for (int i = 0; i < popSize; ++i) check(offspring[i], gen, "post-SE2", i);

        std::vector<double> offFit(popSize);
        for (int i = 0; i < popSize; ++i) offFit[i] = inst.tourLength(offspring[i]);

        // elitist survivor selection over (population + offspring)
        std::vector<int> order(2 * popSize);
        std::iota(order.begin(), order.end(), 0);
        auto fit = [&](int idx) { return idx < popSize ? fitness[idx] : offFit[idx - popSize]; };
        std::sort(order.begin(), order.end(), [&](int a, int b) { return fit(a) < fit(b); });

        std::vector<std::vector<int>> nextPop; nextPop.reserve(popSize);
        std::vector<double> nextFit; nextFit.reserve(popSize);
        for (int i = 0; i < popSize; ++i) {
            int idx = order[i];
            if (idx < popSize) { nextPop.push_back(population[idx]); nextFit.push_back(fitness[idx]); }
            else { nextPop.push_back(offspring[idx - popSize]); nextFit.push_back(offFit[idx - popSize]); }
        }
        population.swap(nextPop);
        fitness.swap(nextFit);

        int argmin = static_cast<int>(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
        if (fitness[argmin] < rep.best) { rep.best = fitness[argmin]; bestTour = population[argmin]; }
    }

    std::string why;
    rep.bestValid = isValidPermutation(bestTour, n, why);
    if (!rep.bestValid)
        std::cout << "      [BEST TOUR INVALID] " << why << "\n";
    return rep;
}

// --------------------------------------------------------------------- main

int main(int argc, char** argv) {
    std::vector<std::string> datasets = {"eil51", "ch150", "kroA200"};
    int generations = 150;
    int popSize = 300;
    long eaxTrials = 20000;
    long lsTrials = 3000;
    unsigned seed = 1;
    std::string dataDir = "../data";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (arg == "--datasets") datasets = splitCsv(next());
        else if (arg == "--gens") generations = std::stoi(next());
        else if (arg == "--pop") popSize = std::stoi(next());
        else if (arg == "--eax-trials") eaxTrials = std::stol(next());
        else if (arg == "--ls-trials") lsTrials = std::stol(next());
        else if (arg == "--seed") seed = static_cast<unsigned>(std::stoul(next()));
        else if (arg == "--data-dir") dataDir = next();
        else { std::cerr << "Unknown argument: " << arg << "\n"; return 1; }
    }

    std::cout << "Validity check: eaxTrials=" << eaxTrials << " lsTrials=" << lsTrials
              << " pipeline(gens=" << generations << ", pop=" << popSize << ")\n\n";

    long grandTotalInvalid = 0;
    for (const auto& ds : datasets) {
        std::string path = dataDir + "/" + ds + ".tsp";
        if (!fs::exists(path)) { std::cerr << "[skip] " << path << " not found\n"; continue; }
        TSPInstance inst;
        try { inst.load(path); }
        catch (const std::exception& e) { std::cerr << "[error] " << path << ": " << e.what() << "\n"; continue; }

        std::cout << "=== " << ds << " (n=" << inst.dimension << ") ===\n";
        std::mt19937 rng(seed);

        long eaxFails = stressEAX(inst, eaxTrials, rng);
        std::cout << "  EAX crossover:      " << (eaxTrials * 2) << " children, "
                  << eaxFails << " invalid  -> " << (eaxFails ? "FAIL" : "ok") << "\n";

        long lsFails = stressLS(inst, lsTrials, rng);
        std::cout << "  LocalSearch.refine: " << lsTrials << " tours, "
                  << lsFails << " invalid  -> " << (lsFails ? "FAIL" : "ok") << "\n";

        PipelineReport rep = runCheckedPipeline(inst, popSize, generations, seed);
        std::cout << "  EM+SE+LS pipeline:  " << rep.invalidEvents << " invalid intermediate tours across "
                  << generations << " gens -> " << (rep.invalidEvents ? "FAIL" : "ok") << "\n";
        std::cout << "  best tour length=" << rep.best
                  << "  final-best-tour valid=" << (rep.bestValid ? "YES" : "NO") << "\n";

        grandTotalInvalid += eaxFails + lsFails + rep.invalidEvents + (rep.bestValid ? 0 : 1);
        std::cout << "\n";
    }

    std::cout << "==================================================\n";
    if (grandTotalInvalid == 0)
        std::cout << "RESULT: no invalid tours anywhere. EM_LS_SE's under-performance is\n"
                  << "        NOT a validity bug - it is a genuine algorithmic result.\n";
    else
        std::cout << "RESULT: " << grandTotalInvalid << " invalidity events detected - the results are\n"
                  << "        contaminated by a correctness bug; fix before drawing conclusions.\n";
    return grandTotalInvalid == 0 ? 0 : 2;
}
