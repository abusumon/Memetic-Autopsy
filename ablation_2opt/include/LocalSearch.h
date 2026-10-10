#pragma once
#include <vector>
#include <random>
#include "TSPInstance.h"

// Local search refinement (Step 05 of the proposal). Implements the four
// operators with their delta formulas (Eq. 10-15) and a first-improvement
// acceptance rule (accept a move only if Delta < 0).
//
// Candidate generation (#1): rather than scanning tour-adjacent *positions*
// (which are not necessarily geographically close), each operator scans a
// city's geometric nearest neighbors (TSPInstance::neighborLists), the
// standard approach in real 2-opt/local-search implementations. This finds
// genuinely useful moves instead of wasting scans on cities that happen to
// be tour-adjacent but far apart in space.
//
// Don't-look bits (#2): once a city yields no improving move, it's skipped
// on subsequent passes until one of its tour-edges actually changes (from
// any operator's move). Standard 2-opt speedup - avoids re-scanning cities
// whose neighborhood hasn't changed.

struct LocalSearchConfig {
    bool useTwoOpt    = true;
    bool useInversion = true;
    bool useSwap      = true;
    bool useInsertion = true;
    int  neighborhoodWindow = 20; // k nearest geometric neighbors considered per city
    int  maxPasses = 2;           // sweeps over the whole tour per operator
    bool useDontLookBits = true;
};

class LocalSearch {
public:
    LocalSearch(const TSPInstance& instance, const LocalSearchConfig& cfg)
        : inst(instance), cfg(cfg) {
        inst.buildNeighborLists(cfg.neighborhoodWindow); // idempotent, cheap if already built
    }

    // Refines `tour` in place. Returns true if any improving move was applied.
    bool refine(std::vector<int>& tour) const {
        int n = static_cast<int>(tour.size());
        buildPos(tour);
        dontLook.assign(n, false); // fresh per offspring

        bool anyImproved = false;
        for (int pass = 0; pass < cfg.maxPasses; ++pass) {
            bool improvedThisPass = false;
            if (cfg.useTwoOpt)    improvedThisPass |= twoOptSweep(tour);
            if (cfg.useInversion) improvedThisPass |= inversionSweep(tour);
            if (cfg.useSwap)      improvedThisPass |= swapSweep(tour);
            if (cfg.useInsertion) improvedThisPass |= insertionSweep(tour);
            anyImproved |= improvedThisPass;
            if (!improvedThisPass) break; // converged, no point sweeping again
        }
        return anyImproved;
    }

private:
    const TSPInstance& inst;
    LocalSearchConfig cfg;
    mutable std::vector<int> pos;          // pos[city] = index in tour
    mutable std::vector<bool> dontLook;    // dontLook[city] = true -> skip scanning from it

    inline double d(int a, int b) const { return inst.roundedDist(a, b); }

    void buildPos(const std::vector<int>& t) const {
        int n = static_cast<int>(t.size());
        pos.assign(n, 0);
        for (int i = 0; i < n; ++i) pos[t[i]] = i;
    }

    inline void wake(int city) const { if (cfg.useDontLookBits) dontLook[city] = false; }

    // --- 2-opt (Eq. 10): edges (ci,ci+1) and (cj,cj+1) -> (ci,cj) and (ci+1,cj+1)
    // Applied by reversing the segment between the two cut points.
    bool twoOptSweep(std::vector<int>& t) const {
        int n = static_cast<int>(t.size());
        bool improved = false;
        for (int i = 0; i < n; ++i) {
            int ci = t[i];
            if (cfg.useDontLookBits && dontLook[ci]) continue;
            bool foundForCi = false;

            for (int cj : inst.neighborLists[ci]) {
                int j = pos[cj];
                int lo = std::min(i, j), hi = std::max(i, j);
                if (hi - lo < 2) continue; // adjacent/degenerate, no valid 2-opt here
                int a = t[lo], a1 = t[lo + 1], b = t[hi], b1 = t[(hi + 1) % n];
                double delta = (d(a, b) + d(a1, b1)) - (d(a, a1) + d(b, b1));
                if (delta < 0.0) {
                    std::reverse(t.begin() + lo + 1, t.begin() + hi + 1);
                    for (int p = lo + 1; p <= hi; ++p) pos[t[p]] = p; // fix positions in reversed range
                    wake(a); wake(a1); wake(b); wake(b1);
                    improved = true;
                    foundForCi = true;
                }
            }
            if (cfg.useDontLookBits && !foundForCi) dontLook[ci] = true;
        }
        return improved;
    }

    // --- Inversion (Eq. 13): edges (ci-1,ci) and (cj,cj+1) -> (ci-1,cj) and (ci,cj+1)
    // Same underlying move as 2-opt (reverse a contiguous segment) but with
    // the index convention from the proposal's formula, kept as a distinct
    // pass since the flowchart lists it separately.
    bool inversionSweep(std::vector<int>& t) const {
        int n = static_cast<int>(t.size());
        bool improved = false;
        for (int i = 0; i < n; ++i) {
            int ci = t[i];
            if (cfg.useDontLookBits && dontLook[ci]) continue;
            bool foundForCi = false;

            for (int cj : inst.neighborLists[ci]) {
                int j = pos[cj];
                int lo = std::min(i, j), hi = std::max(i, j);
                if (hi - lo < 1) continue;
                int pim1 = (lo - 1 + n) % n;
                int a = t[pim1], b = t[lo], c = t[hi], e = t[(hi + 1) % n];
                double delta = (d(a, c) + d(b, e)) - (d(a, b) + d(c, e));
                if (delta < 0.0) {
                    std::reverse(t.begin() + lo, t.begin() + hi + 1);
                    for (int p = lo; p <= hi; ++p) pos[t[p]] = p;
                    wake(a); wake(b); wake(c); wake(e);
                    improved = true;
                    foundForCi = true;
                }
            }
            if (cfg.useDontLookBits && !foundForCi) dontLook[ci] = true;
        }
        return improved;
    }

    // --- Swap (Eq. 11-12): swap two cities; delta computed analytically
    // from the (up to 4) edges touching their positions.
    bool swapSweep(std::vector<int>& t) const {
        int n = static_cast<int>(t.size());
        bool improved = false;
        for (int i = 0; i < n; ++i) {
            int ci = t[i];
            if (cfg.useDontLookBits && dontLook[ci]) continue;
            bool foundForCi = false;

            for (int cj : inst.neighborLists[ci]) {
                int j = pos[cj];
                if (j == i) continue;
                int p = std::min(i, j), q = std::max(i, j);
                double delta = swapDelta(t, p, q);
                if (delta < 0.0) {
                    int a = t[p], b = t[q];
                    std::swap(t[p], t[q]);
                    pos[t[p]] = p; pos[t[q]] = q;
                    wake(a); wake(b);
                    improved = true;
                    foundForCi = true;
                }
            }
            if (cfg.useDontLookBits && !foundForCi) dontLook[ci] = true;
        }
        return improved;
    }

    // Computes the cost delta of swapping positions p and q analytically
    // (no vector copy). p < q is assumed (guaranteed by the caller).
    double swapDelta(const std::vector<int>& t, int p, int q) const {
        int n = static_cast<int>(t.size());
        int pm1 = (p - 1 + n) % n, pp1 = (p + 1) % n;
        int qm1 = (q - 1 + n) % n, qp1 = (q + 1) % n;
        int cp = t[p], cq = t[q];

        if (pp1 == q) {
            // Adjacent case: ... t[pm1], cp, cq, t[qp1] ...
            double before = d(t[pm1], cp) + d(cp, cq) + d(cq, t[qp1]);
            double after  = d(t[pm1], cq) + d(cq, cp) + d(cp, t[qp1]);
            return after - before;
        }
        double before = d(t[pm1], cp) + d(cp, t[pp1]) + d(t[qm1], cq) + d(cq, t[qp1]);
        double after  = d(t[pm1], cq) + d(cq, t[pp1]) + d(t[qm1], cp) + d(cp, t[qp1]);
        return after - before;
    }

    // --- Insertion (Eq. 14): remove city at k, insert between i and i+1.
    bool insertionSweep(std::vector<int>& t) const {
        int n = static_cast<int>(t.size());
        bool improved = false;
        for (int i = 0; i < n; ++i) {
            int ci = t[i];
            if (cfg.useDontLookBits && dontLook[ci]) continue;
            bool foundForCi = false;

            for (int ck : inst.neighborLists[ci]) {
                int k = pos[ck];
                double delta = insertionDelta(t, i, k);
                if (delta < 0.0) {
                    int i1 = t[(i + 1) % n];
                    int km1 = t[(k - 1 + n) % n], kp1 = t[(k + 1) % n];
                    applyInsertion(t, i, k);
                    buildPos(t); // insertion shifts many positions; simplest to rebuild
                    wake(ci); wake(i1); wake(ck); wake(km1); wake(kp1);
                    improved = true;
                    foundForCi = true;
                    i = pos[ci]; // i's own position may have shifted
                }
            }
            if (cfg.useDontLookBits && !foundForCi) dontLook[ci] = true;
        }
        return improved;
    }

    double insertionDelta(const std::vector<int>& t, int i, int k) const {
        int n = static_cast<int>(t.size());
        int ci = t[i], ci1 = t[(i + 1) % n];
        int km1 = (k - 1 + n) % n, kp1 = (k + 1) % n;
        int ck = t[k], ckm1 = t[km1], ckp1 = t[kp1];
        if (k == i || k == (i + 1) % n || km1 == i) return 0.0; // degenerate / adjacent

        double insertCost = (d(ci, ck) + d(ck, ci1)) - d(ci, ci1);
        double removeSaving = (d(ckm1, ck) + d(ck, ckp1)) - d(ckm1, ckp1);
        return insertCost - removeSaving;
    }

    void applyInsertion(std::vector<int>& t, int i, int k) const {
        int city = t[k];
        t.erase(t.begin() + k);
        int insertPos = (k < i) ? i : i + 1; // account for the shift after erase
        t.insert(t.begin() + insertPos, city);
    }
};
