#pragma once
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include "TSPInstance.h"

// Similarity Scoring & Redundancy Elimination (Steps 04 and 06 of the
// proposal). Sim(S1,S2) = |E(S1) ∩ E(S2)| / n, edges treated as undirected
// since the paper's TSPLIB instances are symmetric.
//
// Perf note: this runs O(populationSize x offspringSize) comparisons, twice
// per generation, so it's the single most expensive step in the algorithm
// at paper-scale settings (pop=300). Edge sets are stored as *sorted*
// vectors of int64 keys and compared with a merge-style two-pointer scan,
// which is noticeably faster than std::unordered_set intersection at this
// size (no per-lookup hashing/probing overhead, sequential memory access).
//
// NOTE (implementation choice, not specified by the proposal): the proposal
// says "Eliminate O" without specifying what fills its slot in the
// population so the offspring count stays at N. We replace an eliminated
// offspring with a fresh random tour ("random immigrant"), a standard way
// to inject diversity in GA redundancy-elimination schemes.

inline long long edgeKey(int a, int b) {
    if (a > b) std::swap(a, b);
    return (static_cast<long long>(a) << 32) | static_cast<unsigned int>(b);
}

// Sorted vector of undirected edge keys for a tour.
inline std::vector<long long> edgeSetSorted(const std::vector<int>& tour) {
    int n = static_cast<int>(tour.size());
    std::vector<long long> keys;
    keys.reserve(n);
    for (int i = 0; i < n; ++i)
        keys.push_back(edgeKey(tour[i], tour[(i + 1) % n]));
    std::sort(keys.begin(), keys.end());
    return keys;
}

// Two-pointer intersection count over sorted vectors.
inline int intersectionCount(const std::vector<long long>& a, const std::vector<long long>& b) {
    size_t i = 0, j = 0;
    int common = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) { ++common; ++i; ++j; }
        else if (a[i] < b[j]) ++i;
        else ++j;
    }
    return common;
}

inline double similaritySorted(const std::vector<long long>& e1, const std::vector<long long>& e2, int n) {
    return static_cast<double>(intersectionCount(e1, e2)) / static_cast<double>(n);
}

// Precomputes sorted edge sets for a population (call once per generation,
// reused across both similarity-elimination passes to avoid recomputation).
inline std::vector<std::vector<long long>> precomputeEdgeSets(const std::vector<std::vector<int>>& pop) {
    std::vector<std::vector<long long>> sets;
    sets.reserve(pop.size());
    for (auto& p : pop) sets.push_back(edgeSetSorted(p));
    return sets;
}

// Applies the elimination rule to `offspring` against reference population
// `pop` (with precomputed fitness + precomputed sorted edge sets),
// replacing eliminated individuals with fresh random tours.
// `eliminatedCount`, if non-null, is INCREMENTED (not overwritten) by the
// number of offspring eliminated in this call - lets a caller accumulate a
// running total across both similarity-elimination passes in a generation,
// and across all generations in a run, with two lines at the call site.
// Optional trailing param, defaults to nullptr, so existing callers that
// don't pass it keep compiling and behaving exactly as before.
inline void similarityElimination(
    std::vector<std::vector<int>>& offspring,
    const std::vector<std::vector<int>>& pop,
    const std::vector<double>& popFitness,
    const std::vector<std::vector<long long>>& popEdgeSets,
    const TSPInstance& inst,
    double theta,
    std::mt19937& rng,
    long long* eliminatedCount = nullptr)
{
    int n = inst.dimension;
    std::vector<int> base(n);
    std::iota(base.begin(), base.end(), 0);

    long long eliminatedHere = 0;

    for (auto& O : offspring) {
        double fO = inst.tourLength(O);
        auto eO = edgeSetSorted(O);

        bool eliminate = false;
        for (size_t j = 0; j < pop.size(); ++j) {
            double sim = similaritySorted(eO, popEdgeSets[j], n);
            if (sim >= theta && fO > popFitness[j]) { eliminate = true; break; }
        }

        if (eliminate) {
            // Replace with a fresh random tour (diversity injection).
            std::vector<int> fresh = base;
            std::shuffle(fresh.begin(), fresh.end(), rng);
            O = std::move(fresh);
            ++eliminatedHere;
        }
    }

    if (eliminatedCount) *eliminatedCount += eliminatedHere;
}

// --- Parents-only variant (added for diagnosis) -------------------------
//
// The population-wide version above checks each offspring against ALL
// `pop.size()` reference tours and eliminates on the FIRST match - meaning
// an offspring only needs to resemble ONE out of (typically) 300 population
// members to be killed. Under elitist survivor selection the population
// converges toward a small set of dominant edge-structures fast, and EAX
// offspring inherit shared parental edges by construction - so once the
// population has converged, nearly every offspring resembles SOME member
// closely enough to trigger elimination, at every theta. Measured on
// bier127/kroA200/pr226: elimRate pass1 was ~71-99% regardless of theta,
// meaning the GA spends most generations SA-refining fresh random noise,
// not real crossover offspring - a very different (and much narrower)
// reading of the proposal's "Eliminate O" than what's implemented above.
//
// This variant checks each offspring ONLY against its own two parents (2
// comparisons instead of ~300), which is a common reading of similarity-
// elimination in the literature: catching crossover degenerating into a
// near-duplicate of its own parents, not sweeping for redundancy against
// the whole population. Kept as a SEPARATE function (not a rewrite of
// similarityElimination above) so any other caller of the population-wide
// version (e.g. the ablation study's EM_LS_SE condition) is completely
// unaffected - same isolation principle already used elsewhere in this
// codebase (EnhancedGA.h vs AblationGA.h, main.cpp vs proposed.cpp).
//
// `parentIdxA`/`parentIdxB` are per-offspring indices into `parents` (both
// sized == offspring.size()) identifying which two parents produced each
// Standard double-bridge (4-opt) perturbation: cuts the tour into 4
// segments A|B|C|D at 3 random points and reassembles as A-C-B-D. Changes
// exactly 4 edges - a well-known way to escape 2-opt/3-opt local optima
// and near-duplicate states WITHOUT destroying the tour's learned
// structure, unlike a full random shuffle (which discards everything).
// Falls back to returning the tour unchanged if it's too short for a
// meaningful 4-way cut.
inline std::vector<int> doubleBridgePerturbation(const std::vector<int>& tour, std::mt19937& rng) {
    int n = static_cast<int>(tour.size());
    if (n < 8) return tour;

    std::uniform_int_distribution<int> uidx(1, n - 1);
    std::vector<int> cuts;
    while (cuts.size() < 3) {
        int c = uidx(rng);
        if (std::find(cuts.begin(), cuts.end(), c) == cuts.end()) cuts.push_back(c);
    }
    std::sort(cuts.begin(), cuts.end());
    int p1 = cuts[0], p2 = cuts[1], p3 = cuts[2];

    std::vector<int> result;
    result.reserve(n);
    result.insert(result.end(), tour.begin(), tour.begin() + p1);       // A
    result.insert(result.end(), tour.begin() + p2, tour.begin() + p3);  // C
    result.insert(result.end(), tour.begin() + p1, tour.begin() + p2);  // B
    result.insert(result.end(), tour.begin() + p3, tour.end());         // D
    return result;
}

// offspring. `parents`/`parentFitness`/`parentEdgeSets` are computed once
// per generation by the caller and reused across both elimination passes,
// exactly like `pop`/`popFitness`/`popEdgeSets` already are above.
inline void similarityEliminationVsParents(
    std::vector<std::vector<int>>& offspring,
    const std::vector<int>& parentIdxA,
    const std::vector<int>& parentIdxB,
    const std::vector<std::vector<int>>& parents,
    const std::vector<double>& parentFitness,
    const std::vector<std::vector<long long>>& parentEdgeSets,
    const TSPInstance& inst,
    double theta,
    std::mt19937& rng,
    long long* eliminatedCount = nullptr)
{
    int n = inst.dimension;
    (void)parents; // only parentFitness/parentEdgeSets are needed; kept as a
                    // parameter for call-site clarity (this is the parents-only variant)

    long long eliminatedHere = 0;

    for (size_t i = 0; i < offspring.size(); ++i) {
        auto& O = offspring[i];
        double fO = inst.tourLength(O);
        auto eO = edgeSetSorted(O);

        int pa = parentIdxA[i];
        int pb = parentIdxB[i];

        bool eliminate = false;
        double simA = similaritySorted(eO, parentEdgeSets[pa], n);
        if (simA >= theta && fO > parentFitness[pa]) {
            eliminate = true;
        } else {
            double simB = similaritySorted(eO, parentEdgeSets[pb], n);
            if (simB >= theta && fO > parentFitness[pb]) eliminate = true;
        }

        if (eliminate) {
            // Was: replace with a FULLY RANDOM shuffled tour ("random
            // immigrant"). Measured effect on bier127/kroA200/pr226: at
            // 48-99% elimination per generation, this meant HALF TO
            // NEARLY ALL of the offspring pool was reset to random-tour
            // quality every single generation. A random 51-city tour is
            // bad but a 100-move SA-LS burst can meaningfully repair it in
            // one generation; a random 226-city tour is catastrophically
            // bad and 100 moves barely dents it - so at scale, the GA
            // spent nearly all its budget re-discovering basic structure
            // it already had, every generation, forever. Results stayed
            // 175-450%+ over optimal on larger instances regardless of
            // every other fix in this file's history (comparison scope,
            // LS structure, EAX merge bug) specifically because none of
            // those touched this mechanism.
            //
            // Now: double-bridge perturb the offspring ITSELF instead of
            // discarding it - changes exactly 4 edges (escapes
            // near-duplicate status, satisfies the goal of similarity
            // elimination) while preserving everything else the offspring
            // had already learned, instead of throwing it all away.
            O = doubleBridgePerturbation(O, rng);
            ++eliminatedHere;
        }
    }

    if (eliminatedCount) *eliminatedCount += eliminatedHere;
}