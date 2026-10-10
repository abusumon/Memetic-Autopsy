// ProposedGA: the actual thesis algorithm, and ONLY the actual thesis
// algorithm - no ablation toggles, no reproduction of the Abu Doush et al.
// EM-baseline experiment (that's what the old main.cpp/GeneticAlgorithm.h
// path does; this is deliberately separate from it, same isolation
// principle as AblationGA.h vs EnhancedGA.h).
//
// Pipeline: Essential Mutation (always on - it's one of the three proposed
// components) + Simulated-Annealing-based Local Search, reproducing
// Rahman & Parvez (2021)'s Algorithm 1 STRUCTURALLY, not just its
// operators (replaces the old deterministic greedy LocalSearch.h entirely
// - see AnnealingLocalSearch.h) + Similarity Elimination (single pass,
// post-LS - see below).
//
// Crossover: EAX (Edge Assembly Crossover) - kept as-is. Of what's
// implemented in this codebase, EAX is the stronger operator for TSP: it
// preserves common parental edges exactly and reconstructs the rest via
// AB-cycle decomposition, which is why it replaced Order Crossover (OX,
// still in GeneticAlgorithm.h) in EnhancedGA.h/AblationGA.h in the first
// place - no reason to reconsider that choice here.
//
// SA STRUCTURE NOTE (read AnnealingLocalSearch.h's header first - this is
// the short version): the paper's Algorithm 1 runs ONE shared temperature
// across the WHOLE population, cooling it once per outer iteration, with
// every route getting exactly one candidate move per iteration. This class
// owns that shared `currentTemperature`, decays it once per GA generation
// (after every offspring's one move for that generation), and never resets
// it - making one GA generation the direct structural analogue of one of
// the paper's outer iterations, rather than the earlier design (since
// replaced) which independently re-ran a fresh ~100-iteration anneal from
// T0 for every single offspring, every generation - a fundamentally
// different, much more aggressive per-individual search than what the
// paper actually specifies.
//
// SE NOTE: similarity elimination now runs ONCE, after LS, comparing each
// offspring only against its own two parents (see
// SimilarityElimination.h::similarityEliminationVsParents). It used to run
// twice (pre- and post-LS, against the whole population) - that combination
// measured ~71-99% elimination per generation on bier127/kroA200/pr226
// regardless of theta, because EAX offspring structurally resemble their
// base parent by construction, and comparing an unrefined offspring's
// fitness against an already-refined parent's fitness unfairly penalized
// it before LS had a chance to help. Whether SE is worth keeping in the
// final pipeline is still an open question - re-run the sweep against this
// corrected architecture before deciding.
#pragma once
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <limits>
#include "TSPInstance.h"
#include "Mutations.h"             // essentialEvaluation, essentialMutation
#include "AnnealingLocalSearch.h"  // advisor's SA-based local search
#include "SimilarityElimination.h"
#include "GeneticAlgorithm.h"      // reuse GAResult
#include "EAXCrossover.h"

#ifdef _OPENMP
#include <omp.h>
#endif

struct ProposedGAConfig {
    int populationSize = 300;
    int eliteSize = 60;
    double mutationRate = 0.05;
    double crossoverRate = 0.8;
    double essentialRate = 0.1;
    double similarityThreshold = 0.5; // theta - the value proposed.cpp sweeps
    int generations = 2000;
    AnnealingLSConfig annealingConfig;
    unsigned int seed = 0;
    // Hard wall-clock cap for a single run() call, checked every generation
    // (cheap - one clock read). 0 = unlimited (run all `generations`
    // regardless of time). Exists so total runtime is bounded by a real
    // guarantee rather than a generation count hand-tuned for one
    // particular instance size - a --gens value picked to fit berlin52 in
    // 1-2 hours would silently blow way past that on a bigger instance;
    // this caps it either way. When triggered, run() stops early and
    // returns whatever result it has so far (this is a genuine early stop,
    // not a crash - result.best/convergence are still valid, just shorter
    // than a full uncapped run would have produced).
    double maxWallTimeMs = 0;

    // Stagnation-escape mechanism (added after diagnosing pr226 finding its
    // best result by generation ~254 out of 2000 and then doing nothing for
    // the remaining 87% of its budget - classic premature convergence,
    // worse at larger n). When the best fitness hasn't improved for
    // `stagnationLimit` consecutive generations: (1) reheat the shared
    // annealing temperature back up to `reheatFraction` of T0, giving SA-LS
    // real exploration capability again instead of the near-zero
    // temperature it's converged to by then; (2) replace the worst
    // `restartFraction` of the population (by fitness - elites are
    // untouched by construction, since only the WORST individuals are
    // replaced) with freshly RNN-constructed routes from new starting
    // cities, injecting genuinely new material rather than just perturbing
    // what's already there.
    int stagnationLimit = 150;
    double reheatFraction = 0.5;
    double restartFraction = 0.1;
};

class ProposedGA {
public:
    ProposedGA(const TSPInstance& instance, const ProposedGAConfig& cfg)
        : inst(instance), cfg(cfg), rng(cfg.seed) {
        int numThreads = 1;
#ifdef _OPENMP
        numThreads = omp_get_max_threads();
#endif
        // One independent RNG per OpenMP thread for the annealing LS step -
        // AnnealingLocalSearch::refine takes an rng BY REFERENCE per call
        // (unlike the old LocalSearch, which had no randomness at all), so
        // sharing a single std::mt19937 across threads would be a data
        // race. Seeded deterministically from the main seed so results
        // stay reproducible for a given --seed / run index.
        threadRngs.reserve(numThreads);
        for (int t = 0; t < numThreads; ++t)
            threadRngs.emplace_back(cfg.seed * 1000003u + static_cast<unsigned int>(t) + 1u);

        // One AnnealingLocalSearch instance per thread, pre-allocated here
        // (not constructed inside the parallel loop) - matches the
        // established lsPool pattern EnhancedGA.h/AblationGA.h already use
        // for the same reason (avoid allocation churn across threads
        // inside the hot loop).
        alsPool.reserve(numThreads);
        for (int t = 0; t < numThreads; ++t)
            alsPool.emplace_back(inst, cfg.annealingConfig);

        // Single shared annealing temperature, matching the paper's
        // Algorithm 1: ONE T across the whole population, cooled once per
        // outer iteration (here: once per GA generation, after every
        // offspring has had its one Metropolis move for that generation) -
        // NOT reset per offspring, unlike the old burst-refine design.
        currentTemperature = cfg.annealingConfig.initialTemperature;
    }

    GAResult run() {
        auto t0 = std::chrono::high_resolution_clock::now();

        int n = inst.dimension;
        std::vector<std::vector<int>> population(cfg.populationSize);
        // Population initialization via Repetitive Nearest Neighbor (RNN),
        // matching the paper's Section 3.1 - NOT a uniform random shuffle.
        // Cycles through starting cities 0..n-1, repeating if
        // populationSize > n. See TSPInstance.h::nearestNeighborRoute for
        // why this matters: a random tour is catastrophically far from
        // optimal at n>=200, and this codebase's SA-LS neighborhood is a
        // polishing operator, not a repair operator - it was never going
        // to close that gap within a realistic generation/time budget,
        // which is the leading suspect for the 175-450%+-over-optimal
        // results measured on kroA200/pr226 even with crossover AND
        // similarity elimination both disabled (see chat history / commit
        // notes around this change for the diagnostic trail).
        for (int idx = 0; idx < cfg.populationSize; ++idx) {
            int startCity = idx % n;
            population[idx] = constructRoute(startCity);
        }

        GAResult result;
        result.convergence.reserve(cfg.generations);

        std::vector<double> fitness(cfg.populationSize);
        evaluate(population, fitness);

        bool hitTimeLimit = false;
        long long totalEliminatedPass1 = 0, totalEliminatedPass2 = 0;
        long long totalMergeFallbacks = 0;
        double bestEverFitness = std::numeric_limits<double>::infinity();
        int noImprovementStreak = 0;
        int rnnRestartCounter = 0;      // rotates starting cities across
                                         // multiple stagnation events, so
                                         // repeated escapes inject varied
                                         // routes rather than the same ones
        long long stagnationTriggers = 0;
        for (int gen = 0; gen < cfg.generations; ++gen) {
            if (cfg.maxWallTimeMs > 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double, std::milli>(now - t0).count();
                if (elapsed > cfg.maxWallTimeMs) {
                    hitTimeLimit = true;
                    break;
                }
            }
            // --- Essential Edge Identification ---
            EssentialEdgeSet essentialEdges = essentialEvaluation(population, fitness, cfg.essentialRate);

            // --- Selection, Crossover (EAX), Essential Mutation ---
            std::vector<std::vector<int>> offspring;
            offspring.reserve(cfg.populationSize);
            std::uniform_real_distribution<double> u01(0.0, 1.0);

            std::vector<std::vector<int>> parents;
            parents.reserve(cfg.populationSize);
            for (int i = 0; i < cfg.populationSize; ++i)
                parents.push_back(tournamentSelect(population, fitness));

            // Parent-level fitness/edge-sets, computed once per generation
            // and reused across both elimination passes - mirrors the
            // pattern popFitness/popEdgeSets used for the population-wide
            // version, just scoped to `parents` instead of `population`.
            std::vector<double> parentFitness(cfg.populationSize);
            evaluate(parents, parentFitness);
            auto parentEdgeSets = precomputeEdgeSets(parents);

            // Per-offspring parent indices (2 per offspring, shared by the
            // c1/c2 pair produced from the same parents[i]/parents[i+1]) -
            // fed to similarityEliminationVsParents below so each offspring
            // is checked only against the two tours that actually produced
            // it, not the whole population.
            std::vector<int> parentIdxA, parentIdxB;
            parentIdxA.reserve(cfg.populationSize);
            parentIdxB.reserve(cfg.populationSize);

            for (int i = 0; i < cfg.populationSize; i += 2) {
                const auto& p1 = parents[i];
                int p2idx = (i + 1) % cfg.populationSize;
                const auto& p2 = parents[p2idx];
                std::vector<int> c1, c2;
                if (u01(rng) < cfg.crossoverRate) {
                    auto children = eaxCrossover(p1, p2, inst, rng, &totalMergeFallbacks);
                    c1 = std::move(children.first);
                    c2 = std::move(children.second);
                } else {
                    c1 = p1; c2 = p2;
                }
                essentialMutation(c1, cfg.mutationRate, essentialEdges, rng);
                essentialMutation(c2, cfg.mutationRate, essentialEdges, rng);
                offspring.push_back(std::move(c1));
                offspring.push_back(std::move(c2));
                parentIdxA.push_back(i); parentIdxB.push_back(p2idx);
                parentIdxA.push_back(i); parentIdxB.push_back(p2idx);
            }
            offspring.resize(cfg.populationSize);
            parentIdxA.resize(cfg.populationSize);
            parentIdxB.resize(cfg.populationSize);

            // --- Simulated-Annealing Local Search (advisor's method) ---
            // ONE Metropolis move per offspring, using the shared
            // currentTemperature for this generation (paper's Algorithm 1:
            // every route in the population gets one move per outer
            // iteration, under the same T) - not a per-offspring burst of
            // 100+ iterations restarting from T0 every time.
#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
#endif
            for (int i = 0; i < static_cast<int>(offspring.size()); ++i) {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                alsPool[tid].refine(offspring[i], currentTemperature, threadRngs[tid]);
            }
            // Cool the shared temperature ONCE per generation, after the
            // whole offspring population has had its move - matches
            // Algorithm 1 line 10 ("Decrease the temperature") happening
            // once per outer iteration, after the inner per-route loop.
            currentTemperature *= cfg.annealingConfig.coolingCoefficient;

            // --- Similarity Scoring & Redundancy Elimination (single pass, POST-LS) ---
            // Only one pass now (was two: pre-LS and post-LS). Comparing
            // pre-LS was comparing an unrefined offspring's fitness against
            // an already-refined parent's fitness - an unfair comparison
            // that, combined with EAX offspring structurally resembling
            // their base parent by construction, was driving the ~71-99%
            // elimination rates measured on bier127/kroA200/pr226
            // regardless of theta or population-vs-parents comparison
            // scope. Checking only once, after LS has had its shot at the
            // offspring, is a fairer like-for-like comparison.
            // eliminatedPass1 is intentionally left unused (stays 0) now
            // that there's only one pass - kept as a GAResult field so the
            // existing CSV schema in proposed.cpp doesn't need to change.
            similarityEliminationVsParents(offspring, parentIdxA, parentIdxB, parents,
                                            parentFitness, parentEdgeSets,
                                            inst, cfg.similarityThreshold, rng, &totalEliminatedPass2);

            std::vector<double> offspringFitness(offspring.size());
            evaluate(offspring, offspringFitness);

            // --- Population Update (elitist survivor selection) ---
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
            std::vector<int> carriedOverIndices; // positions in nextPop sourced from
                                                   // `population` (i.e. NOT refined this
                                                   // generation yet - offspring already were)
            nextPop.reserve(cfg.populationSize);
            nextFit.reserve(cfg.populationSize);
            for (int i = 0; i < cfg.populationSize; ++i) {
                int idx = order[i];
                if (idx < static_cast<int>(population.size())) {
                    nextPop.push_back(population[idx]);
                    nextFit.push_back(fitness[idx]);
                    carriedOverIndices.push_back(static_cast<int>(nextPop.size()) - 1);
                } else {
                    nextPop.push_back(offspring[idx - population.size()]);
                    nextFit.push_back(offspringFitness[idx - population.size()]);
                }
            }

            // Refine carried-over survivors too - closes a real gap versus
            // the paper: our elitist selection can let a strong tour sit
            // unchanged for many generations once it keeps winning,
            // whereas the paper has no elitism concept at all - EVERY
            // route gets refined EVERY iteration, including whatever is
            // currently the best. This is not a new/foreign local-search
            // step - it's applying the SAME SA-LS refine(), at the SAME
            // shared currentTemperature, just to a set of tours that would
            // otherwise silently stop receiving it once elite.
#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
#endif
            for (int k = 0; k < static_cast<int>(carriedOverIndices.size()); ++k) {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                int pos = carriedOverIndices[k];
                alsPool[tid].refine(nextPop[pos], currentTemperature, threadRngs[tid]);
            }
            for (int pos : carriedOverIndices) nextFit[pos] = inst.tourLength(nextPop[pos]);

            population.swap(nextPop);
            fitness.swap(nextFit);

            double bestGen = *std::min_element(fitness.begin(), fitness.end());
            result.convergence.push_back(bestGen);

            // --- Stagnation-escape check ---
            if (bestGen < bestEverFitness - 1e-9) {
                bestEverFitness = bestGen;
                noImprovementStreak = 0;
            } else {
                ++noImprovementStreak;
            }

            if (noImprovementStreak >= cfg.stagnationLimit) {
                ++stagnationTriggers;
                noImprovementStreak = 0; // don't re-trigger every generation once past the limit

                // Reheat: give SA-LS real exploration capability again -
                // by this point in a run, currentTemperature has typically
                // cooled to near-zero (pure greedy), which is exactly the
                // state that produces stagnation in the first place.
                double reheatTo = cfg.annealingConfig.initialTemperature * cfg.reheatFraction;
                if (currentTemperature < reheatTo) currentTemperature = reheatTo;

                // Partial restart: replace the WORST restartFraction of the
                // population with fresh RNN-constructed routes. Sorting by
                // fitness descending and taking the top slice means elites
                // (lowest fitness) are untouched by construction.
                int restartCount = static_cast<int>(cfg.restartFraction * cfg.populationSize);
                if (restartCount > 0) {
                    std::vector<int> order(population.size());
                    std::iota(order.begin(), order.end(), 0);
                    std::sort(order.begin(), order.end(), [&](int a, int b) {
                        return fitness[a] > fitness[b]; // descending: worst first
                    });
                    for (int k = 0; k < restartCount && k < static_cast<int>(order.size()); ++k) {
                        int idx = order[k];
                        int startCity = rnnRestartCounter % n;
                        ++rnnRestartCounter;
                        population[idx] = constructRoute(startCity);
                        fitness[idx] = inst.tourLength(population[idx]);
                    }
                }
            }
        }

        result.best = *std::min_element(fitness.begin(), fitness.end());
        {
            auto bestIt = std::min_element(fitness.begin(), fitness.end());
            int bestIdx = static_cast<int>(bestIt - fitness.begin());
            result.bestTour = population[bestIdx];
        }
        result.finalPopAvg = std::accumulate(fitness.begin(), fitness.end(), 0.0) / fitness.size();
        result.eliminatedPass1 = totalEliminatedPass1;
        result.eliminatedPass2 = totalEliminatedPass2;
        result.generationsCompleted = static_cast<int>(result.convergence.size());
        result.mergeFallbackCount = totalMergeFallbacks;
        result.stagnationTriggers = stagnationTriggers;
        (void)hitTimeLimit; // available for callers who want it via a future GAResult field;
                             // currently only affects control flow (early loop exit) above

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
    ProposedGAConfig cfg;
    std::mt19937 rng;
    std::vector<std::mt19937> threadRngs; // one per OpenMP thread (annealing LS)
    std::vector<AnnealingLocalSearch> alsPool; // one per OpenMP thread, pre-allocated
    double currentTemperature = 0.0; // shared across the whole population, see constructor

    // EXPERIMENT HOOK (added for the init-control study): -DRANDOM_INIT replaces RNN
    // construction by a uniform random shuffle everywhere (initial population AND
    // stagnation restarts). Default build is unchanged.
    std::vector<int> constructRoute(int startCity) {
#ifdef RANDOM_INIT
        (void)startCity;
        std::vector<int> t(inst.dimension);
        std::iota(t.begin(), t.end(), 0);
        std::shuffle(t.begin(), t.end(), rng);
        return t;
#else
        return nearestNeighborRoute(inst, startCity);
#endif
    }

    void evaluate(const std::vector<std::vector<int>>& pop, std::vector<double>& fit) {
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