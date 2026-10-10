#include <bits/stdc++.h>
using namespace std;

struct City { double x, y; };
using Tour = vector<int>;

static inline int64_t edgeKey(int a, int b, int n) {
    if (a > b) swap(a, b);
    return (int64_t)a * (int64_t)n + (int64_t)b;
}

static vector<City> readTSP(const string& path) {
    ifstream in(path);
    if (!in) { cerr << "Cannot open " << path << "\n"; exit(1); }
    vector<City> cities;
    string line;
    bool inCoords = false;
    while (getline(in, line)) {
        // trim
        size_t a = line.find_first_not_of(" \t\r\n");
        if (a == string::npos) continue;
        size_t b = line.find_last_not_of(" \t\r\n");
        line = line.substr(a, b - a + 1);
        if (line == "NODE_COORD_SECTION") { inCoords = true; continue; }
        if (line == "EOF" || line.rfind("EOF", 0) == 0) break;
        if (!inCoords) continue;
        if (line == "DISPLAY_DATA_SECTION" ) { break; }
        istringstream iss(line);
        int id; double x, y;
        if (!(iss >> id >> x >> y)) continue;
        cities.push_back({x, y});
    }
    if (cities.empty()) { cerr << "No coordinates parsed from " << path << "\n"; exit(1); }
    return cities;
}

// ============================================================
// Distance matrix
// ============================================================
struct DistMatrix {
    int n;
    vector<double> d; // n*n
    DistMatrix(const vector<City>& c) : n((int)c.size()), d((size_t)n * n) {
        for (int i = 0; i < n; ++i) {
            d[(size_t)i * n + i] = 0.0;
            for (int j = i + 1; j < n; ++j) {
                double dx = c[i].x - c[j].x, dy = c[i].y - c[j].y;
                double v = round(sqrt(dx * dx + dy * dy));
                d[(size_t)i * n + j] = v;
                d[(size_t)j * n + i] = v;
            }
        }
    }
    inline double operator()(int i, int j) const { return d[(size_t)i * n + j]; }
};

static double tourLength(const Tour& t, const DistMatrix& D) {
    double total = 0.0;
    int n = (int)t.size();
    for (int i = 0; i < n; ++i) total += D(t[i], t[(i + 1) % n]);
    return total;
}

// Set of undirected edges (as keys) of a tour.
static unordered_set<int64_t> tourEdgeSet(const Tour& t, int n) {
    unordered_set<int64_t> s;
    s.reserve(t.size() * 2);
    int m = (int)t.size();
    for (int i = 0; i < m; ++i) s.insert(edgeKey(t[i], t[(i + 1) % m], n));
    return s;
}

static bool isValidTour(const Tour& t, int n) {
    if ((int)t.size() != n) return false;
    vector<char> seen(n, 0);
    for (int c : t) {
        if (c < 0 || c >= n || seen[c]) return false;
        seen[c] = 1;
    }
    return true;
}

// ============================================================
// Random helpers
// ============================================================
static Tour randomTour(int n, mt19937& rng) {
    Tour t(n);
    iota(t.begin(), t.end(), 0);
    shuffle(t.begin(), t.end(), rng);
    return t;
}

static int tournamentSelect(const vector<double>& lengths, int k, mt19937& rng) {
    uniform_int_distribution<int> pick(0, (int)lengths.size() - 1);
    int best = pick(rng);
    for (int i = 1; i < k; ++i) {
        int cand = pick(rng);
        if (lengths[cand] < lengths[best]) best = cand;
    }
    return best;
}

static Tour doubleBridge(const Tour& t, mt19937& rng) {
    int n = (int)t.size();
    if (n < 8) return t;
    // pick 3 distinct cut points 1 <= p1 < p2 < p3 < n
    vector<int> cuts;
    uniform_int_distribution<int> pick(1, n - 1);
    while ((int)cuts.size() < 3) {
        int c = pick(rng);
        if (find(cuts.begin(), cuts.end(), c) == cuts.end()) cuts.push_back(c);
    }
    sort(cuts.begin(), cuts.end());
    int p1 = cuts[0], p2 = cuts[1], p3 = cuts[2];
    Tour res;
    res.reserve(n);
    res.insert(res.end(), t.begin(), t.begin() + p1);
    res.insert(res.end(), t.begin() + p3, t.end());
    res.insert(res.end(), t.begin() + p2, t.begin() + p3);
    res.insert(res.end(), t.begin() + p1, t.begin() + p2);
    return res;
}

static bool twoOptPass(Tour& t, const DistMatrix& D) {
    int n = (int)t.size();
    bool improved = false;
    for (int i = 0; i < n - 1; ++i) {
        int a = t[i], b = t[i + 1];
        double dab = D(a, b);
        for (int j = i + 2; j < n; ++j) {
            if (i == 0 && j == n - 1) continue; // would "reverse" the whole tour, no-op
            int c = t[j], e = t[(j + 1) % n];
            double delta = D(a, c) + D(b, e) - dab - D(c, e);
            if (delta < -1e-9) {
                reverse(t.begin() + i + 1, t.begin() + j + 1);
                improved = true;
                b = t[i + 1];
                dab = D(a, b);
            }
        }
    }
    return improved;
}

static void twoOptLocalSearch(Tour& t, const DistMatrix& D, int maxPasses = 40) {
    while (maxPasses-- > 0 && twoOptPass(t, D)) {}
}

static Tour perturbedRestart(const Tour& base, mt19937& rng, int kicks, const DistMatrix& D) {
    Tour t = base;
    for (int k = 0; k < kicks; ++k) {
        t = doubleBridge(t, rng);
        twoOptLocalSearch(t, D);
    }
    return t;
}

// ============================================================
// EAX crossover
// ============================================================
struct AdjPair { int a = -1, b = -1; int cnt = 0; };

static void addAdj(vector<AdjPair>& adj, int u, int v) {
    if (adj[u].cnt == 0) { adj[u].a = v; adj[u].cnt = 1; }
    else { adj[u].b = v; adj[u].cnt = 2; }
}

// Build adjacency (2 neighbours per city) from a tour.
static vector<AdjPair> buildAdj(const Tour& t) {
    int n = (int)t.size();
    vector<AdjPair> adj(n);
    for (int i = 0; i < n; ++i) {
        int u = t[i], v = t[(i + 1) % n];
        addAdj(adj, u, v);
        addAdj(adj, v, u);
    }
    return adj;
}

struct CycleEdge { int u, v; char color; }; // color 'A' or 'B'

static vector<vector<CycleEdge>> findABCycles(const vector<AdjPair>& adjA,
                                               const vector<AdjPair>& adjB,
                                               int n) {
    // remaining (non-common) neighbour lists
    vector<vector<int>> remA(n), remB(n);
    auto isNeighbor = [](const AdjPair& p, int v) { return p.a == v || p.b == v; };
    for (int u = 0; u < n; ++u) {
        int an[2] = {adjA[u].a, adjA[u].b};
        for (int i = 0; i < adjA[u].cnt; ++i) {
            int v = an[i];
            if (!isNeighbor(adjB[u], v)) remA[u].push_back(v);
        }
        int bn[2] = {adjB[u].a, adjB[u].b};
        for (int i = 0; i < adjB[u].cnt; ++i) {
            int v = bn[i];
            if (!isNeighbor(adjA[u], v)) remB[u].push_back(v);
        }
    }
    unordered_set<int64_t> usedA, usedB;
    vector<vector<CycleEdge>> cycles;

    auto takeUnused = [&](vector<int>& lst, unordered_set<int64_t>& used, int from, int n_) -> int {
        for (size_t idx = 0; idx < lst.size(); ++idx) {
            int v = lst[idx];
            int64_t k = edgeKey(from, v, n_);
            if (!used.count(k)) return (int)idx;
        }
        return -1;
    };

    for (int start = 0; start < n; ++start) {
        while (true) {
            int idx = takeUnused(remA[start], usedA, start, n);
            if (idx < 0) break;
            int v0 = remA[start][idx];
            usedA.insert(edgeKey(start, v0, n));
            vector<CycleEdge> cyc;
            cyc.push_back({start, v0, 'A'});
            int current = v0;
            char nextColor = 'B';
            int guard = 0, maxGuard = 4 * n + 10;
            bool closed = false;
            while (guard++ < maxGuard) {
                if (current == start) { closed = true; break; }
                if (nextColor == 'B') {
                    int j = takeUnused(remB[current], usedB, current, n);
                    if (j < 0) break; // shouldn't happen given degree parity; bail defensively
                    int nv = remB[current][j];
                    usedB.insert(edgeKey(current, nv, n));
                    cyc.push_back({current, nv, 'B'});
                    current = nv;
                    nextColor = 'A';
                } else {
                    int j = takeUnused(remA[current], usedA, current, n);
                    if (j < 0) break;
                    int nv = remA[current][j];
                    usedA.insert(edgeKey(current, nv, n));
                    cyc.push_back({current, nv, 'A'});
                    current = nv;
                    nextColor = 'B';
                }
            }
            if (closed && cyc.size() >= 2) cycles.push_back(cyc);
        }
    }
    return cycles;
}

static Tour orderCrossover(const Tour& p1, const Tour& p2, mt19937& rng) {
    int n = (int)p1.size();
    uniform_int_distribution<int> pick(0, n - 1);
    int i = pick(rng), j = pick(rng);
    if (i > j) swap(i, j);
    vector<int> child(n, -1);
    vector<char> used(n, 0);
    for (int k = i; k <= j; ++k) { child[k] = p1[k]; used[p1[k]] = 1; }
    int pos = (j + 1) % n;
    for (int k = 0; k < n; ++k) {
        int c = p2[(j + 1 + k) % n];
        if (!used[c]) { child[pos] = c; used[c] = 1; pos = (pos + 1) % n; }
    }
    return child;
}

// Extract disjoint cycles from a 2-regular adjacency structure.
static vector<vector<int>> extractCycles(vector<AdjPair>& adj, int n) {
    vector<char> visited(n, 0);
    vector<vector<int>> cycles;
    for (int s = 0; s < n; ++s) {
        if (visited[s] || adj[s].cnt == 0) continue;
        vector<int> cyc;
        int prev = -1, cur = s;
        do {
            visited[cur] = 1;
            cyc.push_back(cur);
            int a = adj[cur].a, b = adj[cur].b;
            int nxt = (a != prev) ? a : b;
            prev = cur;
            cur = nxt;
        } while (cur != s && cur != -1 && (int)cyc.size() <= n);
        cycles.push_back(cyc);
    }
    return cycles;
}


static vector<int> mergeTwoCycles(const vector<int>& main_, const vector<int>& other, const DistMatrix& D) {
    int lm = (int)main_.size(), lo = (int)other.size();
    double bestDelta = numeric_limits<double>::infinity();
    int bestI = 0, bestJ = 0; bool bestReversed = false;
    for (int i = 0; i < lm; ++i) {
        int a = main_[i], b = main_[(i + 1) % lm];
        double dab = D(a, b);
        for (int j = 0; j < lo; ++j) {
            int c = other[j], dd = other[(j + 1) % lo];
            double dcd = D(c, dd);
            double costSame = D(a, dd) + D(b, c) - dab - dcd;
            double costRev  = D(a, c)  + D(b, dd) - dab - dcd;
            if (costSame < bestDelta) { bestDelta = costSame; bestI = i; bestJ = j; bestReversed = false; }
            if (costRev  < bestDelta) { bestDelta = costRev;  bestI = i; bestJ = j; bestReversed = true; }
        }
    }
    // rotate 'other' so it starts right after the cut edge (index bestJ+1)
    vector<int> rotated;
    rotated.reserve(lo);
    for (int k = 0; k < lo; ++k) rotated.push_back(other[(bestJ + 1 + k) % lo]);
    if (bestReversed) reverse(rotated.begin(), rotated.end());
    vector<int> merged;
    merged.reserve(lm + lo);
    for (int k = 0; k <= bestI; ++k) merged.push_back(main_[k]);
    merged.insert(merged.end(), rotated.begin(), rotated.end());
    for (int k = bestI + 1; k < lm; ++k) merged.push_back(main_[k]);
    return merged;
}

static Tour eaxCrossover(const Tour& parentA, const Tour& parentB, const DistMatrix& D, mt19937& rng) {
    int n = D.n;
    vector<AdjPair> adjA = buildAdj(parentA);
    vector<AdjPair> adjB = buildAdj(parentB);
    auto cycles = findABCycles(adjA, adjB, n);

    if (cycles.empty()) {
        // parents identical (or degenerate) w.r.t. edges: nothing to recombine
        return parentA;
    }

    // choose a random non-empty subset of AB-cycles to apply
    vector<int> chosen;
    bernoulli_distribution coin(0.5);
    for (size_t c = 0; c < cycles.size(); ++c) if (coin(rng)) chosen.push_back((int)c);
    if (chosen.empty()) {
        uniform_int_distribution<int> pick(0, (int)cycles.size() - 1);
        chosen.push_back(pick(rng));
    }

    // start from parent A's adjacency, apply E-set (remove A-edges, add B-edges)
    vector<AdjPair> childAdj = adjA;
    auto removeEdge = [&](int u, int v) {
        if (childAdj[u].a == v) { childAdj[u].a = childAdj[u].b; childAdj[u].b = -1; childAdj[u].cnt--; }
        else if (childAdj[u].b == v) { childAdj[u].b = -1; childAdj[u].cnt--; }
    };
    auto addEdgeChild = [&](int u, int v) { addAdj(childAdj, u, v); };

    for (int ci : chosen) {
        for (const auto& e : cycles[ci]) {
            if (e.color == 'A') removeEdge(e.u, e.v), removeEdge(e.v, e.u);
        }
    }
    for (int ci : chosen) {
        for (const auto& e : cycles[ci]) {
            if (e.color == 'B') addEdgeChild(e.u, e.v), addEdgeChild(e.v, e.u);
        }
    }

    // sanity check: every node should now have degree exactly 2
    bool broken = false;
    for (int u = 0; u < n; ++u) if (childAdj[u].cnt != 2) { broken = true; break; }
    if (broken) return orderCrossover(parentA, parentB, rng); // defensive fallback

    vector<vector<int>> fragments = extractCycles(childAdj, n);
    if (fragments.empty()) return orderCrossover(parentA, parentB, rng);

    vector<int> merged = fragments[0];
    for (size_t f = 1; f < fragments.size(); ++f) merged = mergeTwoCycles(merged, fragments[f], D);

    if (!isValidTour(merged, n)) return orderCrossover(parentA, parentB, rng); // defensive fallback
    return merged;
}

// ============================================================
// Adaptive Essential Mutation (Abu Doush et al., made adaptive)
// ============================================================
struct EssentialInfo {
    unordered_set<int64_t> essentialEdges;
    double diversity;       // fraction of unique edges across the population
    double essentialRatio;  // adaptive
    double mutationRate;    // adaptive
};

static const double ESSENTIAL_MIN = 0.05;
static const double ESSENTIAL_MAX = 0.30;
static const double MUTRATE_MIN   = 0.01;
static const double MUTRATE_MAX   = 0.10;

static EssentialInfo computeEssentialInfo(const vector<Tour>& population,
                                           const vector<double>& lengths,
                                           int n) {
    unordered_map<int64_t, pair<double,int>> stats; // edge -> (fitnessSum, freq)
    for (size_t p = 0; p < population.size(); ++p) {
        double fit = 1.0 / max(lengths[p], 1e-9);
        const Tour& t = population[p];
        int m = (int)t.size();
        for (int i = 0; i < m; ++i) {
            int64_t k = edgeKey(t[i], t[(i + 1) % m], n);
            auto& e = stats[k];
            e.first += fit;
            e.second += 1;
        }
    }
    size_t totalEdgeInstances = population.size() * (size_t)n;
    double diversity = totalEdgeInstances > 0
        ? min(1.0, (double)stats.size() / (double)totalEdgeInstances)
        : 1.0;

    EssentialInfo info;
    info.diversity = diversity;
    info.essentialRatio = ESSENTIAL_MIN + (1.0 - diversity) * (ESSENTIAL_MAX - ESSENTIAL_MIN);
    info.mutationRate   = MUTRATE_MAX   - diversity         * (MUTRATE_MAX   - MUTRATE_MIN);

    // rank edges by average fitness, keep top essentialRatio fraction
    vector<pair<double,int64_t>> ranked;
    ranked.reserve(stats.size());
    for (auto& kv : stats) ranked.push_back({kv.second.first / kv.second.second, kv.first});
    sort(ranked.begin(), ranked.end(), greater<>());
    size_t numEssential = (size_t)llround(info.essentialRatio * ranked.size());
    numEssential = min(numEssential, ranked.size());
    for (size_t i = 0; i < numEssential; ++i) info.essentialEdges.insert(ranked[i].second);
    return info;
}

// Algorithm 2 of Abu Doush et al., with mutationRate/essentialEdges supplied
// adaptively each generation instead of fixed.
static void essentialMutate(Tour& t, const unordered_set<int64_t>& essentialEdges,
                             double mutationRate, int n, mt19937& rng) {
    int m = (int)t.size();
    uniform_real_distribution<double> uni(0.0, 1.0);
    uniform_int_distribution<int> pickPos(0, m - 1);
    for (int city = 0; city < m; ++city) {
        if (uni(rng) < mutationRate) {
            int swapWith = pickPos(rng);
            int64_t k = edgeKey(t[city], t[(city + 1) % m], n);
            if (!essentialEdges.count(k)) swap(t[city], t[swapWith]);
        }
    }
}

// ============================================================
// Adaptive Similarity Elimination
// ============================================================
static const double THETA_MAX = 0.90; // tolerant when improving
static const double THETA_MIN = 0.50; // aggressive when stagnating
static const int STAGNATION_CAP = 30; // generations

static double adaptiveTheta(int stagnation) {
    double frac = min(1.0, stagnation / (double)STAGNATION_CAP);
    return THETA_MAX - frac * (THETA_MAX - THETA_MIN);
}

static double edgeSimilarity(const Tour& t, const unordered_set<int64_t>& refEdges, int n) {
    int m = (int)t.size();
    int shared = 0;
    for (int i = 0; i < m; ++i) if (refEdges.count(edgeKey(t[i], t[(i + 1) % m], n))) shared++;
    return (double)shared / m;
}

// ============================================================
// Self-Escape local search (Rahman & Ma, Section 6.2)
// ============================================================
static int sharedEdgeCount(const Tour& t, const unordered_set<int64_t>& refEdges, int n) {
    int m = (int)t.size();
    int shared = 0;
    for (int i = 0; i < m; ++i) if (refEdges.count(edgeKey(t[i], t[(i + 1) % m], n))) shared++;
    return shared;
}

// Try the 3-edge-removal + reconnection escape move. Returns true if the
// tour was improved in place.
static bool selfEscapeAttempt(Tour& t, const DistMatrix& D, mt19937& rng) {
    int n = (int)t.size();
    if (n < 8) return false;
    uniform_int_distribution<int> pick(0, n - 1);
    int i = pick(rng), j = pick(rng), k = pick(rng);
    // need 3 distinct sorted cut points
    int idx[3] = {i, j, k};
    sort(idx, idx + 3);
    if (idx[0] == idx[1] || idx[1] == idx[2]) return false;
    i = idx[0]; j = idx[1]; k = idx[2];

    vector<int> p1(t.begin() + i + 1, t.begin() + j + 1);
    vector<int> p2(t.begin() + j + 1, t.begin() + k + 1);
    vector<int> p3;
    p3.insert(p3.end(), t.begin() + k + 1, t.end());
    p3.insert(p3.end(), t.begin(), t.begin() + i + 1);

    auto buildTour = [&](const vector<int>& a, const vector<int>& b, const vector<int>& c) {
        vector<int> res;
        res.reserve(n);
        res.insert(res.end(), a.begin(), a.end());
        res.insert(res.end(), b.begin(), b.end());
        res.insert(res.end(), c.begin(), c.end());
        return res;
    };

    vector<int> p2r(p2.rbegin(), p2.rend());
    vector<int> p1r(p1.rbegin(), p1.rend());
    vector<int> p3r(p3.rbegin(), p3.rend());


    vector<Tour> candidates = {
        buildTour(p1, p2r, p3),
        buildTour(p1r, p2, p3r),
        buildTour(p1r, p2r, p3r)
    };

    double bestLen = tourLength(t, D);
    int bestIdx = -1;
    for (size_t c = 0; c < candidates.size(); ++c) {
        double len = tourLength(candidates[c], D);
        if (len < bestLen - 1e-9) { bestLen = len; bestIdx = (int)c; }
    }
    if (bestIdx >= 0) { t = candidates[bestIdx]; return true; }
    return false;
}

static const int    RESTART_START           = 2 * STAGNATION_CAP; // 60
static const int    RESTART_SPAN            = 200;
static const double RESTART_FRACTION_MAX    = 0.5;
static const int    RESTART_KICKS_MIN       = 2;
static const int    RESTART_KICKS_MAX       = 6;

struct RestartInfo {
    double fraction = 0.0;
    int kicks = RESTART_KICKS_MIN;
};

static RestartInfo computeRestartInfo(int stagnation) {
    RestartInfo info;
    if (stagnation <= RESTART_START) return info; // fraction stays 0
    double frac = min(1.0, (double)(stagnation - RESTART_START) / (double)RESTART_SPAN);
    info.fraction = frac * RESTART_FRACTION_MAX;
    info.kicks = RESTART_KICKS_MIN + (int)llround(frac * (RESTART_KICKS_MAX - RESTART_KICKS_MIN));
    return info;
}

static void applyPerturbedRestart(vector<Tour>& offspring, const vector<Tour>& elitePool,
                                   const DistMatrix& D, const RestartInfo& rinfo, mt19937& rng) {
    if (rinfo.fraction <= 0.0 || offspring.empty() || elitePool.empty()) return;
    int numRestart = (int)llround(rinfo.fraction * offspring.size());
    if (numRestart <= 0) return;

    vector<double> lens(offspring.size());
    for (size_t i = 0; i < offspring.size(); ++i) lens[i] = tourLength(offspring[i], D);
    vector<int> order(offspring.size());
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) { return lens[a] > lens[b]; }); // worst first

    uniform_int_distribution<int> pickElite(0, (int)elitePool.size() - 1);
    for (int r = 0; r < numRestart && r < (int)order.size(); ++r) {
        const Tour& base = elitePool[pickElite(rng)];
        offspring[order[r]] = perturbedRestart(base, rng, rinfo.kicks, D);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0]
             << " instance.tsp [generations=1000] [popSize=200] [eliteFrac=0.1] [crossoverRate=0.8] [seed=random]\n";
        return 1;
    }
    string path = argv[1];
    int generations   = argc > 2 ? atoi(argv[2]) : 1000;
    int popSize       = argc > 3 ? atoi(argv[3]) : 200;
    double eliteFrac  = argc > 4 ? atof(argv[4]) : 0.10;
    double crossRate  = argc > 5 ? atof(argv[5]) : 0.80;
    unsigned seed     = argc > 6 ? (unsigned)atol(argv[6]) : (unsigned)chrono::high_resolution_clock::now().time_since_epoch().count();

    vector<City> cities = readTSP(path);
    int n = (int)cities.size();
    DistMatrix D(cities);
    mt19937 rng(seed);

    int eliteSize = max(1, (int)llround(popSize * eliteFrac));
    if (eliteSize >= popSize) eliteSize = popSize - 1;

    cout << "Loaded " << n << " cities from " << path << "\n";
    cout << "popSize=" << popSize << " generations=" << generations
         << " eliteSize=" << eliteSize << " crossoverRate=" << crossRate
         << " seed=" << seed << "\n";

    vector<Tour> population(popSize);
    vector<double> lengths(popSize);
    for (int i = 0; i < popSize; ++i) {
        population[i] = randomTour(n, rng);
        lengths[i] = tourLength(population[i], D);
    }

    auto bestIt = min_element(lengths.begin(), lengths.end());
    Tour bestTour = population[bestIt - lengths.begin()];
    double bestLen = *bestIt;
    int stagnation = 0;

    const int PRINT_INTERVAL = 50;
    const int EARLY_STOP_UNCHANGED_PRINTS = 5;
    double lastPrintedBest = numeric_limits<double>::infinity();
    int unchangedPrintStreak = 0;
    bool stoppedEarly = false;

    auto t0 = chrono::high_resolution_clock::now();

    for (int gen = 1; gen <= generations; ++gen) {
        // ---- adaptive Essential Mutation setup (population-diversity driven) ----
        EssentialInfo einfo = computeEssentialInfo(population, lengths, n);

        // ---- elite selection ----
        vector<int> order(popSize);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) { return lengths[a] < lengths[b]; });
        vector<Tour> nextPop;
        nextPop.reserve(popSize);
        for (int e = 0; e < eliteSize; ++e) nextPop.push_back(population[order[e]]);

        // ---- parent selection + crossover + adaptive essential mutation ----
        int numOffspring = popSize - eliteSize;
        vector<Tour> offspring;
        offspring.reserve(numOffspring);
        uniform_real_distribution<double> uni(0.0, 1.0);
        for (int i = 0; i < numOffspring; ++i) {
            int pa = tournamentSelect(lengths, 3, rng);
            int pb = tournamentSelect(lengths, 3, rng);
            Tour child;
            if (uni(rng) < crossRate) child = eaxCrossover(population[pa], population[pb], D, rng);
            else child = population[pa];
            essentialMutate(child, einfo.essentialEdges, einfo.mutationRate, n, rng);
            offspring.push_back(move(child));
        }

        // ---- adaptive similarity elimination ----
        unordered_set<int64_t> bestEdges = tourEdgeSet(bestTour, n);
        double theta = adaptiveTheta(stagnation);
        for (auto& child : offspring) {
            if (edgeSimilarity(child, bestEdges, n) > theta) child = doubleBridge(child, rng);
        }

        // ---- self-escape local search (trapped-tour detection + escape) ----
        vector<int> shared(offspring.size());
        long long sharedSum = 0;
        for (size_t i = 0; i < offspring.size(); ++i) {
            shared[i] = sharedEdgeCount(offspring[i], bestEdges, n);
            sharedSum += shared[i];
        }
        double avgShared = offspring.empty() ? 0.0 : (double)sharedSum / offspring.size();
        for (size_t i = 0; i < offspring.size(); ++i) {
            if (shared[i] < avgShared) selfEscapeAttempt(offspring[i], D, rng);
        }

        RestartInfo rinfo = computeRestartInfo(stagnation);
        applyPerturbedRestart(offspring, nextPop, D, rinfo, rng);

        // ---- evaluate offspring, survivor replacement ----
        for (auto& child : offspring) nextPop.push_back(move(child));
        population = move(nextPop);
        lengths.resize(popSize);
        for (int i = 0; i < popSize; ++i) lengths[i] = tourLength(population[i], D);

        auto it = min_element(lengths.begin(), lengths.end());
        double genBest = *it;
        if (genBest < bestLen - 1e-9) {
            bestLen = genBest;
            bestTour = population[it - lengths.begin()];
            stagnation = 0;
        } else {
            stagnation++;
        }

        if (gen % PRINT_INTERVAL == 0 || gen == generations) {
            cout << "gen " << gen << "  best=" << fixed << setprecision(3) << bestLen
                 << "  diversity=" << setprecision(3) << einfo.diversity
                 << "  essentialRatio=" << setprecision(3) << einfo.essentialRatio
                 << "  mutationRate=" << setprecision(3) << einfo.mutationRate
                 << "  theta=" << setprecision(3) << theta
                 << "  restartFrac=" << setprecision(3) << rinfo.fraction
                 << "  restartKicks=" << rinfo.kicks
                 << "  stagnation=" << stagnation << "\n";
        }

        if (gen % PRINT_INTERVAL == 0) {
            if (fabs(bestLen - lastPrintedBest) < 1e-9) {
                unchangedPrintStreak++;
            } else {
                lastPrintedBest = bestLen;
                unchangedPrintStreak = 1;
            }
            if (unchangedPrintStreak >= EARLY_STOP_UNCHANGED_PRINTS) {
                cout << "\nEarly stop: best=" << fixed << setprecision(3) << bestLen
                     << " unchanged for " << EARLY_STOP_UNCHANGED_PRINTS
                     << " consecutive checks (" << EARLY_STOP_UNCHANGED_PRINTS * PRINT_INTERVAL
                     << " generations) -- stopping at gen " << gen
                     << " instead of running to " << generations << ".\n";
                stoppedEarly = true;
                break;
            }
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();

    cout << "\n=== Result ===\n";
    cout << "Best tour length: " << fixed << setprecision(4) << bestLen << "\n";
    cout << "Valid tour: " << (isValidTour(bestTour, n) ? "yes" : "NO -- BUG") << "\n";
    cout << "Stopped: " << (stoppedEarly ? "early (unchanged-best threshold)" : "reached generation limit") << "\n";
    cout << "Time: " << secs << " s\n";
    cout << "Tour: ";
    for (int c : bestTour) cout << c << " ";
    cout << "\n";
    return 0;
}