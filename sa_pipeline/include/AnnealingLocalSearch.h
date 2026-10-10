// Simulated-Annealing-based local search, reproducing the neighborhood
// search procedure of Rahman & Parvez (2021), "Repetitive Nearest Neighbor
// Based Simulated Annealing Search Optimization Algorithm for Traveling
// Salesman Problem" (Open Access Library Journal 8:e7520), Section 3.2 /
// Algorithm 1.
//
// IMPORTANT STRUCTURAL NOTE: Algorithm 1 runs ONE shared temperature T
// across the ENTIRE population of n routes, with each route getting
// exactly ONE candidate move per outer iteration, cooling T once per
// iteration after the whole population has been swept. A literal
// translation of that (one move per offspring per GA generation) was
// tried and measured WORSE than the previous design - because in the
// paper, the same n routes persist UNMODIFIED BY CROSSOVER across all
// 1000 iterations, so one move per iteration compounds into real
// refinement over time; in a GA, offspring are replaced/recombined every
// generation and rarely survive long enough for that to happen.
//
// What's implemented instead: the CALLER (ProposedGA::run()) still owns a
// single shared temperature that genuinely decreases across the whole run
// (cooled once per generation, never reset to T0 mid-run - this part IS
// the real fix over the original design, which restarted at T0=0.025
// every single call regardless of how far the run had progressed). But
// each refine() call performs a BURST of `cfg.iterations` moves against
// that generation's constant temperature, rather than just one - giving
// each offspring a meaningful chance to actually improve before being
// judged, which a GA/memetic setting needs and the paper's flat
// SA-only design didn't have to worry about. See refine()'s comment for
// the measured before/after that motivated this.
//
// Paper's reported settings (Table 1, tuned by trial and error for their
// datasets -- may need retuning for yours):
//   T0 = 0.025, cooling coefficient gamma = 0.99, 1000 outer iterations
//   (paper's own run length - not a constraint on the GA's generation
//   count, see ProposedGA.h for how this plays out over more generations).
//
// Operator probabilities: the paper CONTRADICTS ITSELF here - Section
// 3.2.1's prose says swap=0.5/reversion=0.2/insertion=0.3, but Table 1
// ("optimized parameter values") says swap=0.2/reversion=0.5/insertion=0.3.
// This code follows Table 1 (the itemized parameter table is more likely
// to hold the intended final values than prose that may have transposed
// two numbers) - flagged here in case your advisor confirms otherwise.
// (Also note: NO 2-opt in his neighborhood set -- only these three.)
//
// PERFORMANCE REWRITE (this version): the previous implementation copied
// the ENTIRE tour (O(n)) and recomputed the ENTIRE tour length from
// scratch (O(n)) for EVERY proposed move, accepted or not - even though
// swap/reversal/insertion each change at most 3-4 edges. At n~500,
// pop=300, 1000 SA iterations/offspring, 2000 generations, that's on the
// order of hundreds of billions of wasted O(n) operations across a run,
// and is the leading suspect for why pr439/d493 were hitting the wall-
// clock cap before finishing their generation budget - not that they
// genuinely needed more time, but that each move was ~n times more
// expensive than it needed to be.
//
// This version instead computes the length CHANGE (delta) each move would
// cause directly from the handful of edges it actually touches - O(1) for
// swap and reversal, O(1) to *evaluate* for insertion (the O(n) vector
// shift for insertion is real and unavoidable, but is now only paid when
// a move is ACCEPTED, not for every candidate - a large saving at low
// temperature, where most candidates are rejected). The running tour
// length is maintained incrementally (`currentLen += delta`) instead of
// being recomputed from scratch each iteration.
//
// Behavior is otherwise intentionally IDENTICAL to the previous version:
// same operator selection, same random-number draw order/count per
// iteration (so results are numerically reproducible for a given seed),
// same Metropolis accept rule, and refine() still returns the BEST tour
// seen anywhere during the burst (reverting to the call's starting tour
// if nothing in the burst ever beat it) - not just wherever the walk
// happened to end up. Only the cost of getting there changed.
#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "TSPInstance.h"

struct AnnealingLSConfig {
    double initialTemperature = 0.025;
    double coolingCoefficient = 0.99;   // T_{k+1} = coolingCoefficient * T_k,
                                         // applied ONCE per GA generation by
                                         // the caller - see ProposedGA.h
    double swapProb = 0.2;
    double reversionProb = 0.5;
    double insertionProb = 0.3;         // the three probabilities should sum to 1.0

    // Number of Metropolis moves per refine() call - i.e. per offspring,
    // per GA generation (a "burst" evaluated against a constant T for that
    // generation - see refine()'s comment for why this isn't just 1 move,
    // despite the paper's Algorithm 1 doing exactly 1 move per route per
    // outer iteration). Set via proposed.cpp's --sa-iters, default 100.
    int iterations = 1000;
};

class AnnealingLocalSearch {
public:
    AnnealingLocalSearch(const TSPInstance& instance, const AnnealingLSConfig& cfg)
        : inst(instance), cfg(cfg) {}

    // Performs a burst of `cfg.iterations` Metropolis accept/reject moves
    // on `tour`, in place, all evaluated against the SAME externally-owned
    // `T` for this call (T does NOT decay within the burst - only the
    // CALLER's generation-to-generation temperature changes, see
    // ProposedGA.h). Returns the best-ever-seen tour within this burst
    // (matches the paper's "update best solution ever found" at the
    // per-call level), and returns true if that best beat the tour's
    // starting length.
    //
    // WHY A BURST, NOT ONE MOVE (disclosed deviation from a literal
    // reading of Algorithm 1 - worth a line in your methodology section):
    // a single-move-per-call version was tried first, matching the paper's
    // structure exactly (one move per route per outer iteration, shared T
    // cooled once per generation). Measured result: WORSE than before, and
    // elimRate stayed ~99% - because in the paper, the same n routes
    // persist UNMODIFIED BY CROSSOVER across all 1000 iterations, so one
    // move per iteration compounds into real refinement over time. Here,
    // offspring are replaced/recombined every GA generation, so they
    // rarely survive long enough for one move per generation to compound -
    // the assumption that makes the paper's single-move design work simply
    // doesn't hold in a memetic-GA setting. Restoring a multi-move burst
    // per offspring per generation - while still fixing the actual bug
    // (temperature genuinely decreasing across the run instead of
    // resetting to T0=0.025 every single call) - is a deliberate practical
    // compromise between literal fidelity and what the GA context actually
    // needs, not an unexamined reversion to the old design.
    bool refine(std::vector<int>& tour, double T, std::mt19937& rng) const {
        int n = static_cast<int>(tour.size());
        // Degenerate/too-small instances: the O(1) delta formulas below
        // assume prevI/nextI/j/nextJ are meaningfully distinct positions,
        // which isn't guaranteed for tiny n. Real TSPLIB instances are
        // always far larger than this, so this guard is a no-op in
        // practice - it only exists to fail safe rather than silently
        // compute nonsense on a pathological input.
        if (n < 4) return false;

        double startLen = inst.tourLength(tour);   // O(n), paid ONCE per call
        double currentLen = startLen;
        double bestLen = startLen;
        std::vector<int> best = tour;               // O(n), paid ONCE per call
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        std::uniform_int_distribution<int> uidx(0, n - 1);

        for (int it = 0; it < cfg.iterations; ++it) {
            int i = uidx(rng), j = uidx(rng);
            while (j == i) j = uidx(rng);
            if (i > j) std::swap(i, j);

            double r = u01(rng);
            int opType;      // 0 = swap, 1 = reversal, 2 = insertion
            double delta;
            if (r < cfg.swapProb) {
                opType = 0;
                delta = swapDelta(tour, i, j);
            } else if (r < cfg.swapProb + cfg.reversionProb) {
                opType = 1;
                delta = reversalDelta(tour, i, j);
            } else {
                opType = 2;
                delta = insertionDelta(tour, i, j);
            }

            bool accept = (delta <= 0.0);
            if (!accept) {
                double p = std::exp(-delta / std::max(T, 1e-12));
                accept = (u01(rng) < p);
            }

            if (accept) {
                applyOperator(tour, i, j, opType);
                currentLen += delta;
                if (currentLen < bestLen) {
                    bestLen = currentLen;
                    best = tour;   // O(n), only on improvement - rare
                                    // relative to per-iteration cost, same
                                    // as the previous implementation.
                }
            }
            // T intentionally NOT decayed here - see class comment above.
        }

        tour = best;
        return bestLen < startLen - 1e-9;
    }

private:
    const TSPInstance& inst;
    AnnealingLSConfig cfg;

    double d(int a, int b) const { return inst.roundedDist(a, b); }

    // --- O(1) length-delta evaluation, one per operator -----------------
    // Each of these reads `t` WITHOUT mutating it, and returns the change
    // in tour length that applyOperator(t, i, j, opType) below would
    // cause. Positions are always passed with i < j (guaranteed by the
    // caller). All wraparound cases (i==0, j==n-1, adjacent positions)
    // are handled generically rather than special-cased individually,
    // EXCEPT for insertion's one genuine degeneracy (see insertionDelta).

    // Swap t[i] and t[j]. Affected edges are whichever of
    // {(i-1,i), (i,i+1), (j-1,j), (j,j+1)} are distinct once wraparound
    // is applied (adjacent/wraparound positions can make two of these the
    // same physical edge - e.g. i=0,j=n-1 - so edges are deduplicated by
    // position pair before summing, rather than assumed to always be 4
    // distinct edges).
    double swapDelta(const std::vector<int>& t, int i, int j) const {
        int n = static_cast<int>(t.size());
        int prevI = (i - 1 + n) % n, nextI = (i + 1) % n;
        int prevJ = (j - 1 + n) % n, nextJ = (j + 1) % n;

        int edgePos[4][2] = {{prevI, i}, {i, nextI}, {prevJ, j}, {j, nextJ}};
        long long seenKeys[4];
        int seenCount = 0;
        double before = 0.0, after = 0.0;

        auto cityAfter = [&](int pos) -> int {
            if (pos == i) return t[j];
            if (pos == j) return t[i];
            return t[pos];
        };

        for (auto& e : edgePos) {
            int a = e[0], b = e[1];
            long long key = (a < b) ? (static_cast<long long>(a) << 32) | static_cast<unsigned int>(b)
                                     : (static_cast<long long>(b) << 32) | static_cast<unsigned int>(a);
            bool dup = false;
            for (int k = 0; k < seenCount; ++k) if (seenKeys[k] == key) { dup = true; break; }
            if (dup) continue;
            seenKeys[seenCount++] = key;

            before += d(t[a], t[b]);
            after  += d(cityAfter(a), cityAfter(b));
        }
        return after - before;
    }

    // Reverse t[i..j] in place. Internal edges within the reversed segment
    // are preserved (same physical edges, opposite traversal direction -
    // distance is symmetric, so they contribute zero to the delta). Only
    // the two boundary edges change: (i-1,i) becomes (i-1, old t[j]), and
    // (j,j+1) becomes (old t[i], j+1). If the segment spans the whole
    // array (i==0, j==n-1), these two boundary edges are the same
    // physical (wraparound) edge - reversing the entire tour is a true
    // no-op (same cycle, opposite direction), handled as an explicit
    // early-return below rather than left to double-count that edge.
    double reversalDelta(const std::vector<int>& t, int i, int j) const {
        int n = static_cast<int>(t.size());
        int prevI = (i - 1 + n) % n, nextJ = (j + 1) % n;

        if (prevI == j) {
            // Whole-tour reversal (i==0, j==n-1): same cycle, opposite
            // direction - delta is exactly 0.
            return 0.0;
        }

        auto cityAfter = [&](int pos) -> int {
            if (pos == i) return t[j];
            if (pos == j) return t[i];
            return t[pos];
        };

        double before = d(t[prevI], t[i]) + d(t[j], t[nextJ]);
        double after  = d(cityAfter(prevI), cityAfter(i)) + d(cityAfter(j), cityAfter(nextJ));
        return after - before;
    }

    // Move the city at position i to directly behind the city at position
    // j (same "erase i, insert after j" semantics as before). Standard
    // Or-opt-style O(1) delta: removing city i merges its two neighbors
    // into a bridge edge, and inserting it after city j splits that edge
    // into two. The one genuine degeneracy - when the city at j is
    // ALREADY immediately followed by the city at i (equivalently,
    // (j+1) mod n == i, which given i<j only arises when i==0 and
    // j==n-1) - is a true no-op (nothing to move; it's already there) and
    // is guarded explicitly rather than left to the general formula,
    // which would otherwise double-count that shared edge.
    double insertionDelta(const std::vector<int>& t, int i, int j) const {
        int n = static_cast<int>(t.size());
        int prevI = (i - 1 + n) % n;
        int nextI = (i + 1) % n;
        int nextJ = (j + 1) % n;

        if (nextJ == i) return 0.0; // degenerate no-op, see comment above

        double before = d(t[prevI], t[i]) + d(t[i], t[nextI]) + d(t[j], t[nextJ]);
        double after  = d(t[prevI], t[nextI]) + d(t[j], t[i]) + d(t[i], t[nextJ]);
        return after - before;
    }

    // --- Actual in-place mutation, applied ONLY on accepted moves --------
    void applyOperator(std::vector<int>& t, int i, int j, int opType) const {
        switch (opType) {
            case 0: std::swap(t[i], t[j]); break;
            case 1: std::reverse(t.begin() + i, t.begin() + j + 1); break;
            case 2: {
                int city = t[i];
                t.erase(t.begin() + i);
                int insertPos = (i < j) ? j : j + 1; // i<j always here, kept
                                                       // as-is for clarity/
                                                       // symmetry with the
                                                       // delta formula above
                t.insert(t.begin() + insertPos, city);
                break;
            }
        }
    }
};
