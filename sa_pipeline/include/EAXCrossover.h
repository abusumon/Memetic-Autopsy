#pragma once
#include <vector>
#include <array>
#include <unordered_set>
#include <random>
#include <algorithm>
#include "TSPInstance.h"

// Edge Assembly Crossover (EAX) - Nagata & Kobayashi (1997).
// Replaces Order Crossover (OX1) as EnhancedGA's crossover operator.
//
// How it works:
//  1. Edges common to both parents are preserved unconditionally.
//  2. The DIFFERING edges are decomposed into "AB-cycles": cycles that
//     alternate between an edge unique to parent A and an edge unique to
//     parent B (this is always possible - both parents are Hamiltonian
//     cycles, so at every city the count of remaining A-edges equals the
//     count of remaining B-edges).
//  3. A random subset of AB-cycles (the "E-set") is chosen - each cycle
//     included independently with probability 0.5 (the "EAX-rand" variant,
//     the standard simple version in the literature).
//  4. The E-set is applied to a base parent: its A-edges in the E-set are
//     removed, the corresponding B-edges are added. This preserves
//     degree-2 at every city, but generally yields several disjoint
//     sub-tours rather than one Hamiltonian cycle.
//  5. The sub-tours are greedily reconnected ("merged") into a single
//     valid tour, using geometric nearest-neighbor candidates (reusing
//     TSPInstance::neighborLists) to keep the merge step fast.
//
// Two children are produced per crossover call, using each parent in turn
// as the "base" (matching the pair-returning interface the previous OX1
// crossover provided).

namespace eax_detail {

inline long long edgeKey(int a, int b) {
    if (a > b) std::swap(a, b);
    return (static_cast<long long>(a) << 32) | static_cast<unsigned int>(b);
}

} // namespace eax_detail

// Produces ONE child, structurally based on `base` (inherits base's edges
// except where the selected AB-cycles swap in `other`'s edges).
inline std::vector<int> eaxCrossoverSingle(const std::vector<int>& base,
                                            const std::vector<int>& other,
                                            const TSPInstance& inst,
                                            std::mt19937& rng,
                                            long long* fallbackCount = nullptr) {
    int n = static_cast<int>(base.size());
    if (n < 4) return base; // too small for a meaningful cycle decomposition

    // --- Tour-neighbor adjacency for base (A) and other (B) ---
    std::vector<std::array<int, 2>> adjA(n), adjB(n);
    for (int i = 0; i < n; ++i) {
        int c = base[i], p = base[(i - 1 + n) % n], nx = base[(i + 1) % n];
        adjA[c] = {p, nx};
    }
    for (int i = 0; i < n; ++i) {
        int c = other[i], p = other[(i - 1 + n) % n], nx = other[(i + 1) % n];
        adjB[c] = {p, nx};
    }

    // --- Common edges stay fixed, excluded from AB-cycle decomposition ---
    std::unordered_set<long long> edgesA, edgesB;
    edgesA.reserve(n); edgesB.reserve(n);
    for (int i = 0; i < n; ++i) edgesA.insert(eax_detail::edgeKey(base[i], base[(i + 1) % n]));
    for (int i = 0; i < n; ++i) edgesB.insert(eax_detail::edgeKey(other[i], other[(i + 1) % n]));

    // --- Remaining (differing) adjacency, with a FIXED index pairing:
    // RA[v][i] is "paired" with RB[v][i] for every valid i. This pairing
    // is what makes the walk below deterministic, guaranteed to close into
    // a cycle, and guaranteed to alternate colors - a naive "just grab any
    // available same-color edge" walk can leave other vertices stranded
    // (found via testing: it can call .back() on an empty list). Fixing a
    // transition (pairing) at every vertex up front avoids that entirely:
    // this is the standard technique for decomposing an Eulerian-style
    // graph into edge-disjoint alternating cycles without backtracking.
    std::vector<std::vector<int>> RA(n), RB(n);
    for (int v = 0; v < n; ++v) {
        for (int u : adjA[v]) if (edgesB.find(eax_detail::edgeKey(v, u)) == edgesB.end()) RA[v].push_back(u);
        for (int u : adjB[v]) if (edgesA.find(eax_detail::edgeKey(v, u)) == edgesA.end()) RB[v].push_back(u);
        // RA[v].size() == RB[v].size() always (both equal 2 - commonEdgesAt(v))
    }
    std::vector<std::vector<bool>> RAused(n), RBused(n);
    for (int v = 0; v < n; ++v) {
        RAused[v].assign(RA[v].size(), false);
        RBused[v].assign(RB[v].size(), false);
    }

    // Finds the index j such that RA[v][j]==target (or RB[v][j]==target),
    // among not-yet-used entries. Used to locate "which slot did this edge
    // arrive through" at the vertex we just moved to.
    auto findSlot = [](const std::vector<int>& lst, const std::vector<bool>& used, int target) -> int {
        for (int j = 0; j < static_cast<int>(lst.size()); ++j)
            if (!used[j] && lst[j] == target) return j;
        return -1;
    };

    // --- Decompose differing edges into AB-cycles ---
    struct ABCycle { std::unordered_set<long long> aEdges, bEdges; };
    std::vector<ABCycle> cycles;

    for (int start = 0; start < n; ++start) {
        for (int startSlot = 0; startSlot < static_cast<int>(RA[start].size()); ++startSlot) {
            if (RAused[start][startSlot]) continue;

            ABCycle cyc;
            int current = start;
            int slot = startSlot;
            char nextColor = 'A';
            bool closed = false;

            while (!closed) {
                if (nextColor == 'A') {
                    int neighbor = RA[current][slot];
                    RAused[current][slot] = true;
                    cyc.aEdges.insert(eax_detail::edgeKey(current, neighbor));
                    int mirror = findSlot(RA[neighbor], RAused[neighbor], current);
                    if (mirror < 0) break; // should not happen (see closure-check note below); defensive abort only
                    RAused[neighbor][mirror] = true;
                    current = neighbor;
                    slot = mirror;      // pairing rule: same index in RB[current]
                    nextColor = 'B';
                } else {
                    int neighbor = RB[current][slot];
                    RBused[current][slot] = true;
                    cyc.bEdges.insert(eax_detail::edgeKey(current, neighbor));
                    int mirror = findSlot(RB[neighbor], RBused[neighbor], current);
                    if (mirror < 0) break;
                    RBused[neighbor][mirror] = true;
                    current = neighbor;
                    slot = mirror;      // pairing rule: same index in RA[current]
                    nextColor = 'A';
                }
                // Closure check: the walk is a deterministic bijective state
                // machine on (vertex, slot, pending-color) - it MUST return
                // to its exact starting state eventually. Checking just
                // current==start is NOT sufficient: a degree-4 vertex can
                // be passed through via its OTHER slot before the cycle
                // that actually contains startSlot closes, which would
                // wrongly terminate early and strand the remaining edges
                // (this was the actual bug - verified via testing, it
                // produced "vertex has degree 0" / missing-city results).
                if (current == start && slot == startSlot && nextColor == 'A') closed = true;
            }
            if (closed) cycles.push_back(std::move(cyc));
        }
    }

    if (cycles.empty()) return base; // parents share every edge - nothing to recombine

    // --- E-set selection (EAX-rand: each cycle included w.p. 0.5) ---
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::vector<bool> selected(cycles.size(), false);
    bool anySelected = false;
    for (size_t i = 0; i < cycles.size(); ++i) {
        selected[i] = u01(rng) < 0.5;
        anySelected = anySelected || selected[i];
    }
    if (!anySelected) {
        std::uniform_int_distribution<size_t> uidx(0, cycles.size() - 1);
        selected[uidx(rng)] = true;
    }

    // --- Apply E-set to base: remove its A-edges, add the B-edges ---
    std::unordered_set<long long> childEdges = edgesA;
    for (size_t i = 0; i < cycles.size(); ++i) {
        if (!selected[i]) continue;
        for (auto e : cycles[i].aEdges) childEdges.erase(e);
        for (auto e : cycles[i].bEdges) childEdges.insert(e);
    }

    // --- Extract connected components (sub-tours) ---
    std::vector<std::vector<int>> adjList(n);
    for (auto key : childEdges) {
        int a = static_cast<int>(key >> 32);
        int b = static_cast<int>(key & 0xffffffffLL);
        adjList[a].push_back(b);
        adjList[b].push_back(a);
    }

    std::vector<bool> visited(n, false);
    std::vector<std::vector<int>> subtours;
    for (int v = 0; v < n; ++v) {
        if (visited[v] || adjList[v].size() != 2) continue; // guard: skip malformed (shouldn't happen)
        std::vector<int> path;
        int prev = -1, current = v;
        while (!visited[current]) {
            visited[current] = true;
            path.push_back(current);
            if (adjList[current].size() != 2) break; // safety guard
            int nxt = (adjList[current][0] != prev) ? adjList[current][0] : adjList[current][1];
            prev = current;
            current = nxt;
        }
        subtours.push_back(std::move(path));
    }

    if (subtours.empty()) return base; // safety fallback, should not occur
    if (subtours.size() == 1) return subtours[0]; // already a full Hamiltonian cycle

    // --- Merge sub-tours via greedy nearest-neighbor reconnection ---
    inst.buildNeighborLists(20); // idempotent if already built elsewhere at this k

    while (subtours.size() > 1) {
        auto& X = subtours[0];
        auto& Y = subtours[1];
        int lx = static_cast<int>(X.size()), ly = static_cast<int>(Y.size());

        std::vector<int> posInY(inst.dimension, -1);
        for (int i = 0; i < ly; ++i) posInY[Y[i]] = i;

        double bestDelta = 1e18;
        int bestXi = -1, bestYi = -1; bool bestFlip = false;

        for (int xi = 0; xi < lx; ++xi) {
            int x1 = X[xi], x2 = X[(xi + 1) % lx];
            for (int cand : inst.neighborLists[x1]) {
                int yi = posInY[cand];
                if (yi < 0) continue;
                int y1 = Y[yi], y2 = Y[(yi + 1) % ly];
                double removed = inst.roundedDist(x1, x2) + inst.roundedDist(y1, y2);
                double addStraight = inst.roundedDist(x1, y1) + inst.roundedDist(x2, y2);
                double addCross    = inst.roundedDist(x1, y2) + inst.roundedDist(x2, y1);
                if (addStraight - removed < bestDelta) { bestDelta = addStraight - removed; bestXi = xi; bestYi = yi; bestFlip = false; }
                if (addCross - removed < bestDelta)    { bestDelta = addCross - removed;    bestXi = xi; bestYi = yi; bestFlip = true; }
            }
        }

        if (bestXi < 0) {
            // Candidate-list fast path found NO overlap between X and Y at
            // all (can happen when there are many small subtours - a
            // city's k=20 geometric nearest neighbors may simply not
            // include anyone from this specific subtour, and this gets
            // more likely as subtour count grows with instance size).
            // Previously this fell back to bestXi=0/bestYi=0 - blind
            // concatenation with ZERO regard for distance, capable of
            // injecting one huge, essentially-random-length edge into the
            // tour per occurrence. This is the leading suspect for why
            // larger instances (kroA200/pr226) came back 175%/373% over
            // optimal while bier127 (fewer/larger subtours on average)
            // stayed reasonable - a handful of these per crossover call is
            // enough to blow up total tour length far past anything a
            // merely-weak GA would produce. Fixed: fall back to a genuine
            // exhaustive O(lx*ly) nearest-pair search between X and Y
            // instead of a blind default - still finds an actually-good
            // reconnection point, just without the neighbor-list shortcut.
            // `fallbackCount`, if non-null, is incremented so callers can
            // measure how often this path actually triggers.
            if (fallbackCount) ++(*fallbackCount);
            for (int xi = 0; xi < lx; ++xi) {
                int x1 = X[xi], x2 = X[(xi + 1) % lx];
                for (int yi = 0; yi < ly; ++yi) {
                    int y1 = Y[yi], y2 = Y[(yi + 1) % ly];
                    double removed = inst.roundedDist(x1, x2) + inst.roundedDist(y1, y2);
                    double addStraight = inst.roundedDist(x1, y1) + inst.roundedDist(x2, y2);
                    double addCross    = inst.roundedDist(x1, y2) + inst.roundedDist(x2, y1);
                    if (addStraight - removed < bestDelta) { bestDelta = addStraight - removed; bestXi = xi; bestYi = yi; bestFlip = false; }
                    if (addCross - removed < bestDelta)    { bestDelta = addCross - removed;    bestXi = xi; bestYi = yi; bestFlip = true; }
                }
            }
        }

        std::vector<int> merged;
        merged.reserve(lx + ly);
        for (int i = 0; i <= bestXi; ++i) merged.push_back(X[i]);
        if (!bestFlip) {
            for (int i = 0; i < ly; ++i) merged.push_back(Y[(bestYi + 1 + i) % ly]);
        } else {
            for (int i = 0; i < ly; ++i) merged.push_back(Y[(bestYi - i + ly) % ly]);
        }
        for (int i = bestXi + 1; i < lx; ++i) merged.push_back(X[i]);

        subtours.erase(subtours.begin() + 1);
        subtours[0] = std::move(merged);
    }

    return subtours[0];
}

// Produces a pair of children (base=p1, base=p2), matching the interface
// the previous OX1 orderCrossover() provided.
inline std::pair<std::vector<int>, std::vector<int>> eaxCrossover(
    const std::vector<int>& p1, const std::vector<int>& p2,
    const TSPInstance& inst, std::mt19937& rng, long long* fallbackCount = nullptr) {
    std::vector<int> c1 = eaxCrossoverSingle(p1, p2, inst, rng, fallbackCount);
    std::vector<int> c2 = eaxCrossoverSingle(p2, p1, inst, rng, fallbackCount);
    return {c1, c2};
}