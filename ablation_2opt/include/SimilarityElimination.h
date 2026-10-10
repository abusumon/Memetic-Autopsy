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

inline int g_seReplacement = 0; // EXPERIMENT HOOK: 0 = random immigrant (original behaviour), 1 = double-bridge of the offspring itself

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
inline void similarityElimination(
    std::vector<std::vector<int>>& offspring,
    const std::vector<std::vector<int>>& pop,
    const std::vector<double>& popFitness,
    const std::vector<std::vector<long long>>& popEdgeSets,
    const TSPInstance& inst,
    double theta,
    std::mt19937& rng)
{
    int n = inst.dimension;
    std::vector<int> base(n);
    std::iota(base.begin(), base.end(), 0);

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
            if (g_seReplacement == 1 && n >= 8) {
                std::uniform_int_distribution<int> dcut(1, n - 1);
                int c[3];
                do { c[0] = dcut(rng); c[1] = dcut(rng); c[2] = dcut(rng); std::sort(c, c + 3); }
                while (c[0] == c[1] || c[1] == c[2]);
                std::vector<int> nt; nt.reserve(n);
                nt.insert(nt.end(), O.begin(), O.begin() + c[0]);
                nt.insert(nt.end(), O.begin() + c[1], O.begin() + c[2]);
                nt.insert(nt.end(), O.begin() + c[0], O.begin() + c[1]);
                nt.insert(nt.end(), O.begin() + c[2], O.end());
                O = std::move(nt);
            } else {
                std::vector<int> fresh = base;
                std::shuffle(fresh.begin(), fresh.end(), rng);
                O = std::move(fresh);
            }
        }
    }
}
