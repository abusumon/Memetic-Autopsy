#pragma once
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>
#include "TSPInstance.h"
#include "Mutations.h"           // essentialEvaluation, essentialMutation (Steps 02-03)
#include "LocalSearch.h"         // Step 05
#include "SimilarityElimination.h" // Steps 04, 06
#include "GeneticAlgorithm.h"    // reuse GAConfig-style fields, GAResult
#include "EAXCrossover.h"        // Edge Assembly Crossover (Step 03 crossover operator)

#ifdef _OPENMP
#include <omp.h>
#endif

struct EnhancedGAConfig {
    int populationSize = 300;
    int eliteSize = 60;
    double mutationRate = 0.05;
    double crossoverRate = 0.8;
    double essentialRate = 0.1;
    double similarityThreshold = 0.5; // theta
    int generations = 2000;
    LocalSearchConfig lsConfig;
    unsigned int seed = 0;
};

// Implements Algorithm: EM-LS-SE (see chat writeup / proposal Steps 01-07).
// This is the algorithm used by proposed.cpp for your actual reported
// results - it always runs Essential Mutation + Local Search + Similarity
// Elimination together, with no toggles. For the ablation study (which
// mechanism individually to enable/disable), see AblationGA.h / ablation.cpp
// - a deliberately separate class so nothing about this file changes when
// the ablation driver is built or run.
//
// #6 parallelization note: LocalSearch keeps mutable per-refine() state
// (pos[]/dontLook[]) inside the object, so a single shared LocalSearch is
// NOT safe to call concurrently from multiple threads. We instead keep one
// LocalSearch instance per OpenMP thread (lsPool) and index into it with
// omp_get_thread_num(); each thread's instance owns its own scratch state,
// so no locking is needed. Falls back to plain serial loops automatically
// when compiled without -fopenmp (guarded by #ifdef _OPENMP).
class EnhancedGA {
public:
    EnhancedGA(const TSPInstance& instance, const EnhancedGAConfig& cfg)
        : inst(instance), cfg(cfg), rng(cfg.seed) {
        int numThreads = 1;
#ifdef _OPENMP
        numThreads = omp_get_max_threads();
#endif
        lsPool.reserve(numThreads);
        for (int t = 0; t < numThreads; ++t) lsPool.emplace_back(inst, cfg.lsConfig);
    }

    GAResult run() {
        auto t0 = std::chrono::high_resolution_clock::now();

        int n = inst.dimension;
        std::vector<std::vector<int>> population(cfg.populationSize);
        std::vector<int> base(n);
        std::iota(base.begin(), base.end(), 0);
        for (auto& ind : population) {
            ind = base;
            std::shuffle(ind.begin(), ind.end(), rng);
        }

        GAResult result;
        result.convergence.reserve(cfg.generations);

        std::vector<double> fitness(cfg.populationSize);
        evaluate(population, fitness);

        for (int gen = 0; gen < cfg.generations; ++gen) {
            // --- Step 02: Essential Edge Identification ---
            EssentialEdgeSet essentialEdges = essentialEvaluation(population, fitness, cfg.essentialRate);

            // --- Step 03: Selection, Crossover, Conditional Mutation ---
            std::vector<std::vector<int>> offspring;
            offspring.reserve(cfg.populationSize);
            std::uniform_real_distribution<double> u01(0.0, 1.0);

            std::vector<std::vector<int>> parents;
            parents.reserve(cfg.populationSize);
            for (int i = 0; i < cfg.populationSize; ++i)
                parents.push_back(tournamentSelect(population, fitness));

            for (int i = 0; i < cfg.populationSize; i += 2) {
                const auto& p1 = parents[i];
                const auto& p2 = parents[(i + 1) % cfg.populationSize];
                std::vector<int> c1, c2;
                if (u01(rng) < cfg.crossoverRate) {
                    auto children = eaxCrossover(p1, p2, inst, rng);
                    c1 = std::move(children.first);
                    c2 = std::move(children.second);
                } else {
                    c1 = p1; c2 = p2;
                }
                essentialMutation(c1, cfg.mutationRate, essentialEdges, rng);
                essentialMutation(c2, cfg.mutationRate, essentialEdges, rng);
                offspring.push_back(std::move(c1));
                offspring.push_back(std::move(c2));
            }
            offspring.resize(cfg.populationSize);

            // Population edge sets are needed by both similarity-elimination
            // passes this generation; compute once and reuse (perf).
            auto popEdgeSets = precomputeEdgeSets(population);

            // --- Step 04: 1st Similarity Scoring & Redundancy Elimination ---
            similarityElimination(offspring, population, fitness, popEdgeSets,
                                   inst, cfg.similarityThreshold, rng);

            // --- Step 05: Local Search Refinement (#6: parallel across offspring) ---
#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
#endif
            for (int i = 0; i < static_cast<int>(offspring.size()); ++i) {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                lsPool[tid].refine(offspring[i]);
            }

            // --- Step 06: 2nd Similarity Scoring & Redundancy Elimination ---
            similarityElimination(offspring, population, fitness, popEdgeSets,
                                   inst, cfg.similarityThreshold, rng);

            std::vector<double> offspringFitness(offspring.size());
            evaluate(offspring, offspringFitness);

            // --- Step 07: Population Update (elitist survivor selection) ---
            std::vector<int> order(population.size() + offspring.size());
            std::iota(order.begin(), order.end(), 0);
            auto combinedFitness = [&](int idx) {
                return idx < static_cast<int>(population.size()) ? fitness[idx]
                                                                   : offspringFitness[idx - population.size()];
            };
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return combinedFitness(a) < combinedFitness(b);
            });

            std::vector<std::vector<int>> nextPop;
            std::vector<double> nextFit;
            nextPop.reserve(cfg.populationSize);
            nextFit.reserve(cfg.populationSize);
            for (int i = 0; i < cfg.populationSize; ++i) {
                int idx = order[i];
                if (idx < static_cast<int>(population.size())) {
                    nextPop.push_back(population[idx]);
                    nextFit.push_back(fitness[idx]);
                } else {
                    nextPop.push_back(offspring[idx - population.size()]);
                    nextFit.push_back(offspringFitness[idx - population.size()]);
                }
            }
            population.swap(nextPop);
            fitness.swap(nextFit);

            double bestGen = *std::min_element(fitness.begin(), fitness.end());
            result.convergence.push_back(bestGen);
        }

        result.best = *std::min_element(fitness.begin(), fitness.end());
        result.finalPopAvg = std::accumulate(fitness.begin(), fitness.end(), 0.0) / fitness.size();

        // First generation at which the final best value was reached.
        for (size_t g = 0; g < result.convergence.size(); ++g) {
            if (result.convergence[g] <= result.best + 1e-9) {
                result.bestGeneration = static_cast<int>(g);
                break;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        result.elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return result;
    }

private:
    const TSPInstance& inst;
    EnhancedGAConfig cfg;
    std::mt19937 rng;
    std::vector<LocalSearch> lsPool; // one per OpenMP thread (#6)

    void evaluate(const std::vector<std::vector<int>>& pop, std::vector<double>& fit) {
        // #6: fitness evaluation is embarrassingly parallel (no shared
        // mutable state), safe to parallelize directly.
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < static_cast<int>(pop.size()); ++i)
            fit[i] = inst.tourLength(pop[i]);
    }

    std::vector<int> tournamentSelect(const std::vector<std::vector<int>>& population,
                                       const std::vector<double>& fitness, int k = 3) {
        std::uniform_int_distribution<int> uidx(0, static_cast<int>(population.size()) - 1);
        int best = uidx(rng);
        for (int i = 1; i < k; ++i) {
            int cand = uidx(rng);
            if (fitness[cand] < fitness[best]) best = cand;
        }
        return population[best];
    }
};
