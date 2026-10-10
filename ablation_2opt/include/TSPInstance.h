#pragma once
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

// Loads a symmetric TSPLIB (.tsp) instance in EUC_2D format and provides
// distance / tour-length helpers used throughout the GA.
struct TSPInstance {
    std::string name;
    int dimension = 0;
    std::vector<double> x, y;

    // #4: cached n x n distance matrix, built once in load(). Avoids
    // recomputing sqrt()/round() millions of times during local search.
    std::vector<std::vector<double>> distMatrix;

    // #1: geometric nearest-neighbor candidate lists, built lazily on first
    // use by LocalSearch. neighborLists[c] = the kNeighborsBuilt closest
    // other cities to c, sorted nearest-first. Using *geometric* neighbors
    // (rather than tour-adjacent positions) lets local search actually find
    // the moves that matter instead of scanning arbitrary tour-order windows.
    mutable std::vector<std::vector<int>> neighborLists;
    mutable int neighborListsK = 0; // 0 = not built yet

    static std::string trim(const std::string& s) {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return "";
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    }

    // Loads a .tsp file (NODE_COORD_SECTION / EUC_2D). Ignores comments etc.
    void load(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open())
            throw std::runtime_error("Could not open TSP file: " + path);

        std::string line;
        bool inCoords = false;
        while (std::getline(in, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;

            if (trimmed.rfind("NAME", 0) == 0) {
                size_t colon = trimmed.find(':');
                if (colon != std::string::npos) name = trim(trimmed.substr(colon + 1));
                continue;
            }
            if (trimmed.rfind("DIMENSION", 0) == 0) {
                size_t colon = trimmed.find(':');
                if (colon != std::string::npos) dimension = std::stoi(trim(trimmed.substr(colon + 1)));
                continue;
            }
            if (trimmed.rfind("NODE_COORD_SECTION", 0) == 0) { inCoords = true; continue; }
            if (trimmed.rfind("EOF", 0) == 0) break;
            if (trimmed.rfind("EDGE_WEIGHT_TYPE", 0) != std::string::npos && trimmed.rfind("EDGE_WEIGHT_TYPE", 0) == 0) {
                if (trimmed.find("EUC_2D") == std::string::npos) {
                    // Only EUC_2D is supported by this loader.
                }
                continue;
            }

            if (inCoords) {
                std::istringstream iss(trimmed);
                int id; double cx, cy;
                if (iss >> id >> cx >> cy) {
                    x.push_back(cx);
                    y.push_back(cy);
                }
            }
        }
        if (dimension == 0) dimension = static_cast<int>(x.size());
        if (static_cast<int>(x.size()) != dimension)
            throw std::runtime_error("Parsed coordinate count does not match DIMENSION in " + path);

        buildDistanceMatrix();
    }

    inline double dist(int a, int b) const {
        double dx = x[a] - x[b];
        double dy = y[a] - y[b];
        return std::sqrt(dx * dx + dy * dy);
    }

    void buildDistanceMatrix() {
        distMatrix.assign(dimension, std::vector<double>(dimension, 0.0));
        for (int i = 0; i < dimension; ++i)
            for (int j = i + 1; j < dimension; ++j) {
                double d = std::round(dist(i, j));
                distMatrix[i][j] = d;
                distMatrix[j][i] = d;
            }
    }

    // #1: builds/rebuilds geometric nearest-neighbor lists (k per city).
    // Idempotent - only rebuilds if k changes, so callers can call this
    // freely every time without worrying about redundant O(n^2 log n) work.
    void buildNeighborLists(int k) const {
        if (neighborListsK == k && !neighborLists.empty()) return; // already built
        k = std::min(k, dimension - 1);
        neighborLists.assign(dimension, {});
        for (int c = 0; c < dimension; ++c) {
            std::vector<int> others;
            others.reserve(dimension - 1);
            for (int o = 0; o < dimension; ++o) if (o != c) others.push_back(o);
            std::partial_sort(others.begin(), others.begin() + k, others.end(),
                               [&](int a, int b) { return distMatrix[c][a] < distMatrix[c][b]; });
            neighborLists[c].assign(others.begin(), others.begin() + k);
        }
        neighborListsK = k;
    }

    // TSPLIB rounds EUC_2D distances to nearest integer (nint) for the
    // "official" tour length; the paper's cost values (Table 3) match this
    // convention. Now backed by the cached matrix (#4) instead of
    // recomputing sqrt()/round() on every call.
    inline double roundedDist(int a, int b) const {
        return distMatrix[a][b];
    }

    double tourLength(const std::vector<int>& tour) const {
        double total = 0.0;
        for (size_t i = 0; i + 1 < tour.size(); ++i)
            total += roundedDist(tour[i], tour[i + 1]);
        total += roundedDist(tour.back(), tour.front());
        return total;
    }
};
