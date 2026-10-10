#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include "TSPInstance.h"

enum class MutationType { EM, SM, RM, CIM };

inline std::string mutationName(MutationType t) {
    switch (t) {
        case MutationType::EM:  return "EM";
        case MutationType::SM:  return "SM";
        case MutationType::RM:  return "RM";
        case MutationType::CIM: return "CIM";
    }
    return "?";
}

// ---------------------------------------------------------------------
// Classical operators (Section 2 of the paper)
// ---------------------------------------------------------------------

// Swap Mutation (SM): swap a random city with another random city.
inline void swapMutation(std::vector<int>& individual, double mutationRate, std::mt19937& rng) {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_int_distribution<int> uidx(0, static_cast<int>(individual.size()) - 1);
    for (size_t city = 0; city < individual.size(); ++city) {
        if (u01(rng) < mutationRate) {
            int swapWith = uidx(rng);
            std::swap(individual[city], individual[swapWith]);
        }
    }
}

// Reverse Mutation (RM): randomly select a sub-sequence within the route and reverse it.
inline void reverseMutation(std::vector<int>& individual, double mutationRate, std::mt19937& rng) {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(rng) >= mutationRate) return; // one mutation event per call, gated by rate
    int n = static_cast<int>(individual.size());
    std::uniform_int_distribution<int> uidx(0, n - 1);
    int a = uidx(rng), b = uidx(rng);
    if (a > b) std::swap(a, b);
    std::reverse(individual.begin() + a, individual.begin() + b + 1);
}

// Center Inverse Mutation (CIM): split the route into two parts, invert
// each part, then place them back in the same position.
inline void centerInverseMutation(std::vector<int>& individual, double mutationRate, std::mt19937& rng) {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(rng) >= mutationRate) return;
    int n = static_cast<int>(individual.size());
    if (n < 2) return;
    std::uniform_int_distribution<int> uidx(1, n - 1);
    int cut = uidx(rng);
    std::reverse(individual.begin(), individual.begin() + cut);
    std::reverse(individual.begin() + cut, individual.end());
}

// ---------------------------------------------------------------------
// Essential Mutation (EM) framework - Algorithm 1 (Essential Evaluation)
// and Algorithm 2 (Essential Mutation) from the paper.
// ---------------------------------------------------------------------

// Undirected-edge lookup table: essential[min(a,b)].count(max(a,b)) > 0
// <=> the edge {a,b} is essential. TSP is symmetric (stated explicitly in
// the proposal), but the tour array traverses each physical edge in
// whichever direction that individual happens to visit it - reversion and
// insertion mutations flip segment orientation constantly, so the SAME
// physical edge shows up as (A,B) in one individual and (B,A) in another.
// A directed key (as this used to be) splits one edge's "votes" across two
// different map entries depending on arbitrary traversal direction, which
// can keep a genuinely strong edge below the essential-ratio threshold for
// reasons that have nothing to do with its actual quality. Canonicalizing
// to (min,max) merges those votes back into a single entry, exactly like
// SimilarityElimination.h::edgeKey already does for the same reason.
using EssentialEdgeSet = std::unordered_map<int, std::unordered_set<int>>;

// Algorithm 1 - Essential Evaluation
// Input: population (individuals + their fitness/tour length), essentialRatio
// Output: set of essential edges (the most consistently strong, i.e. lowest
//         average-length, edges across the population).
inline EssentialEdgeSet essentialEvaluation(
    const std::vector<std::vector<int>>& population,
    const std::vector<double>& fitness,      // tour length per individual (lower = better)
    double essentialRatio)
{
    // pathTotalFitness[edge] and pathCounter[edge], edge canonicalized to
    // (min(a,b), max(a,b)) so both traversal directions of the same
    // physical edge accumulate into the same bucket.
    std::unordered_map<long long, double> pathTotalFitness;
    std::unordered_map<long long, long long> pathCounter;
    auto key = [](int a, int b) -> long long {
        if (a > b) std::swap(a, b);
        return (static_cast<long long>(a) << 32) | static_cast<unsigned int>(b);
    };

    for (size_t idx = 0; idx < population.size(); ++idx) {
        const auto& individual = population[idx];
        double fit = fitness[idx];
        int n = static_cast<int>(individual.size());
        for (int i = 0; i < n; ++i) {
            int fromCity = individual[i];
            int toCity = individual[(i + 1) % n]; // wraps to close the tour (handles the final edge too)
            long long k = key(fromCity, toCity);
            pathTotalFitness[k] += fit;
            pathCounter[k] += 1;
        }
    }

    // pathAvgFitness = pathTotalFitness / pathCounter, then rank all unique
    // edges and keep the top (essentialRatio * uniqueEdgesCount) by quality.
    // "Quality" = lowest average tour length contributed, i.e. the edges
    // that most consistently appear in short (high-fitness) tours.
    // from <= to always, since `key` above canonicalized to (min,max).
    struct EdgeAvg { int from, to; double avg; };
    std::vector<EdgeAvg> avgs;
    avgs.reserve(pathTotalFitness.size());
    for (auto& kv : pathTotalFitness) {
        int from = static_cast<int>(kv.first >> 32);
        int to   = static_cast<int>(kv.first & 0xffffffffLL);
        double avg = kv.second / static_cast<double>(pathCounter[kv.first]);
        avgs.push_back({from, to, avg});
    }

    long long uniqueEdgesCounter = static_cast<long long>(avgs.size());
    long long numberOfEssentialEdges = static_cast<long long>(uniqueEdgesCounter * essentialRatio);

    std::sort(avgs.begin(), avgs.end(), [](const EdgeAvg& a, const EdgeAvg& b) {
        return a.avg < b.avg; // ascending: shortest/most-consistent edges first
    });

    EssentialEdgeSet essential;
    for (long long i = 0; i < numberOfEssentialEdges && i < static_cast<long long>(avgs.size()); ++i) {
        essential[avgs[i].from].insert(avgs[i].to);
    }
    return essential;
}

// Algorithm 2 - Essential Mutation
// For each city, with probability mutationRate, pick a random swap partner.
// The swap only happens if the edge (city, city+1) is NOT in the essential
// list. Lookup is canonicalized (min,max) to match how essentialEvaluation
// now stores edges - looking this up directed against an undirected table
// would silently miss every essential edge that this individual happens to
// traverse in the opposite direction, defeating the fix above.
inline void essentialMutation(std::vector<int>& individual, double mutationRate,
                               const EssentialEdgeSet& essentialEdges, std::mt19937& rng) {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    int n = static_cast<int>(individual.size());
    std::uniform_int_distribution<int> uidx(0, n - 1);

    for (int city = 0; city < n; ++city) {
        if (u01(rng) < mutationRate) {
            int swapWith = uidx(rng);

            int a = individual[city];
            int b = individual[(city + 1) % n];
            int lo = std::min(a, b), hi = std::max(a, b);
            bool isEssential = false;
            auto it = essentialEdges.find(lo);
            if (it != essentialEdges.end() && it->second.count(hi)) isEssential = true;

            if (!isEssential) {
                std::swap(individual[city], individual[swapWith]);
            }
        }
    }
}
