#pragma once
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>
#include "TSPInstance.h"
#include "Mutations.h"

struct GAConfig {
    int populationSize = 300;
    int eliteSize = 60;
    double mutationRate = 0.05;
    double crossoverRate = 0.8;
    double essentialRate = 0.1;   // only used by EM
    int generations = 2000;
    MutationType mutation = MutationType::EM;
    unsigned int seed = 0;
};

struct GAResult {
    double best = 0.0;
    double finalPopAvg = 0.0;
    double elapsedMs = 0.0;
    int bestGeneration = 0;           // generation index at which `best` was first reached
    std::vector<double> convergence; // best-so-far per generation (for plotting)
};

class GeneticAlgorithm {
public:
    GeneticAlgorithm(const TSPInstance& instance, const GAConfig& cfg)
        : inst(instance), cfg(cfg), rng(cfg.seed) {}

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
            // --- Parent selection (tournament) ---
            std::vector<std::vector<int>> parents;
            parents.reserve(cfg.populationSize);
            for (int i = 0; i < cfg.populationSize; ++i)
                parents.push_back(tournamentSelect(population, fitness));

            // --- Crossover (Order Crossover, OX) ---
            std::vector<std::vector<int>> offspring;
            offspring.reserve(cfg.populationSize);
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            for (int i = 0; i < cfg.populationSize; i += 2) {
                const auto& p1 = parents[i];
                const auto& p2 = parents[(i + 1) % cfg.populationSize];
                if (u01(rng) < cfg.crossoverRate) {
                    auto children = orderCrossover(p1, p2);
                    offspring.push_back(std::move(children.first));
                    offspring.push_back(std::move(children.second));
                } else {
                    offspring.push_back(p1);
                    offspring.push_back(p2);
                }
            }
            offspring.resize(cfg.populationSize);

            // --- Threshold / essential-edge computation (Fig. 2 step) ---
            EssentialEdgeSet essentialEdges;
            if (cfg.mutation == MutationType::EM) {
                essentialEdges = essentialEvaluation(population, fitness, cfg.essentialRate);
            }

            // --- Mutation ---
            for (auto& ind : offspring) {
                switch (cfg.mutation) {
                    case MutationType::EM:  essentialMutation(ind, cfg.mutationRate, essentialEdges, rng); break;
                    case MutationType::SM:  swapMutation(ind, cfg.mutationRate, rng); break;
                    case MutationType::RM:  reverseMutation(ind, cfg.mutationRate, rng); break;
                    case MutationType::CIM: centerInverseMutation(ind, cfg.mutationRate, rng); break;
                }
            }

            std::vector<double> offspringFitness(offspring.size());
            evaluate(offspring, offspringFitness);

            // --- Survivor selection: elitism over (population + offspring) ---
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

        // First generation at which the final best value was reached (i.e.
        // how many generations were actually needed - later generations
        // found nothing better).
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
    GAConfig cfg;
    std::mt19937 rng;

    void evaluate(const std::vector<std::vector<int>>& pop, std::vector<double>& fit) {
        for (size_t i = 0; i < pop.size(); ++i)
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

    // Standard Order Crossover (OX1) - produces two valid permutation children.
    std::pair<std::vector<int>, std::vector<int>> orderCrossover(const std::vector<int>& p1,
                                                                   const std::vector<int>& p2) {
        int n = static_cast<int>(p1.size());
        std::uniform_int_distribution<int> uidx(0, n - 1);
        int a = uidx(rng), b = uidx(rng);
        if (a > b) std::swap(a, b);

        auto makeChild = [&](const std::vector<int>& donor, const std::vector<int>& filler) {
            std::vector<int> child(n, -1);
            std::vector<char> used(n, 0);
            for (int i = a; i <= b; ++i) {
                child[i] = donor[i];
                used[donor[i]] = 1;
            }
            int idx = (b + 1) % n;
            for (int i = 0; i < n; ++i) {
                int cand = filler[(b + 1 + i) % n];
                if (!used[cand]) {
                    child[idx] = cand;
                    used[cand] = 1;
                    idx = (idx + 1) % n;
                }
            }
            return child;
        };

        std::vector<int> c1 = makeChild(p1, p2);
        std::vector<int> c2 = makeChild(p2, p1);
        return {c1, c2};
    }
};
