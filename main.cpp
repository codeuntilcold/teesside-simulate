///bin/sh -c 'g++ -std=c++17 -O3 -o "${0%.cpp}" "$0" && exec "${0%.cpp}" "$@"' "$0" "$@"; exit $?

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

constexpr char STRATEGY_C = 'C';
constexpr char STRATEGY_D = 'D';

using Population = std::vector<std::vector<char>>;
using Fitnesses = std::vector<std::vector<double>>;

struct Coord {
    int x, y;
};

std::string formatFloat(double f) {
    std::ostringstream oss;
    oss << std::setprecision(15) << f;
    std::string s = oss.str();
    if (s.find('.') == std::string::npos) {
        s += ".0";
    }
    return s;
}

Population newPopulation(int sizeX, int sizeY, double initialCooperatorRatio, std::mt19937& rng) {
    Population pop(sizeX, std::vector<char>(sizeY));
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < sizeX; i++) {
        for (int j = 0; j < sizeY; j++) {
            pop[i][j] = (dist(rng) < initialCooperatorRatio) ? STRATEGY_C : STRATEGY_D;
        }
    }
    return pop;
}

Population copyPopulation(const Population& p) {
    return p;
}

int countCooperators(const Population& p) {
    int count = 0;
    for (const auto& row : p) {
        for (char c : row) {
            if (c == STRATEGY_C) count++;
        }
    }
    return count;
}

Fitnesses newFitnesses(int sizeX, int sizeY) {
    return Fitnesses(sizeX, std::vector<double>(sizeY, 0.0));
}

double sumFitnesses(const Fitnesses& f) {
    double total = 0.0;
    for (const auto& row : f) {
        for (double v : row) {
            total += v;
        }
    }
    return total;
}

std::vector<Coord> getNeighbors(int i, int j, int sizeX, int sizeY) {
    std::vector<Coord> neighbors;
    neighbors.reserve(4);
    if (i > 0) neighbors.push_back({i - 1, j});
    if (i < sizeX - 1) neighbors.push_back({i + 1, j});
    if (j > 0) neighbors.push_back({i, j - 1});
    if (j < sizeY - 1) neighbors.push_back({i, j + 1});
    return neighbors;
}

double fermi(double focal, double neighbor, double K = 0.3) {
    return 1.0 / (1.0 + std::exp((focal - neighbor) / K));
}

Fitnesses calculateFitness(const Population& population, double beta) {
    int sizeX = population.size();
    int sizeY = population[0].size();
    Fitnesses fitnesses = newFitnesses(sizeX, sizeY);

    for (int i = 0; i < sizeX; i++) {
        for (int j = 0; j < sizeY; j++) {
            char strategy = population[i][j];
            double totalPayoff = 0.0;
            auto neighbors = getNeighbors(i, j, sizeX, sizeY);

            for (const auto& n : neighbors) {
                char neighborStrategy = population[n.x][n.y];
                if (strategy == STRATEGY_C) {
                    totalPayoff += (neighborStrategy == STRATEGY_C) ? 1.0 : 0.0;
                } else {
                    totalPayoff += (neighborStrategy == STRATEGY_C) ? beta : 0.0;
                }
            }
            fitnesses[i][j] = totalPayoff;
        }
    }
    return fitnesses;
}

Fitnesses calculateFitnessPGG(const Population& population, double r) {
    int sizeX = population.size();
    int sizeY = population[0].size();
    Fitnesses fitnesses = newFitnesses(sizeX, sizeY);

    for (int i = 0; i < sizeX; i++) {
        for (int j = 0; j < sizeY; j++) {
            auto neighbors = getNeighbors(i, j, sizeX, sizeY);

            int nCoop = 0;
            for (const auto& n : neighbors) {
                if (population[n.x][n.y] == STRATEGY_C) nCoop++;
            }

            double contributionCost = 0.0;
            if (population[i][j] == STRATEGY_C) {
                nCoop++;
                contributionCost = 1.0;
            }

            double groupSize = neighbors.size() + 1;
            fitnesses[i][j] = (nCoop * r / groupSize) - contributionCost;
        }
    }
    return fitnesses;
}

std::pair<Fitnesses, double> pop(const Population& population, Fitnesses fitnesses, int pc, double theta, double a) {
    double cost = 0.0;
    int xc = countCooperators(population);
    int sizeX = population.size();
    int sizeY = population[0].size();

    if (xc < pc) {
        for (int i = 0; i < sizeX; i++) {
            for (int j = 0; j < sizeY; j++) {
                if (population[i][j] == STRATEGY_C) {
                    fitnesses[i][j] += theta;
                }
            }
        }
        cost = (theta / a) * xc;
    }
    return {fitnesses, cost};
}

std::pair<Fitnesses, double> neb(const std::string& mode, const Population& population, Fitnesses fitnesses,
                                  int nc, double epsilon, double theta, double a) {
    double cost = 0.0;
    int sizeX = population.size();
    int sizeY = population[0].size();

    if (mode == "i" || mode == "ii") {
        for (int i = 0; i < sizeX; i++) {
            for (int j = 0; j < sizeY; j++) {
                if (population[i][j] == STRATEGY_C) {
                    Coord* highestNeighborD = nullptr;
                    Coord* highestNeighborC = nullptr;
                    std::vector<Coord> neighborsD;
                    Coord hndStorage, hncStorage;

                    std::vector<Coord> dirs = {{i-1, j}, {i+1, j}, {i, j-1}, {i, j+1}};
                    for (const auto& n : dirs) {
                        int x = n.x, y = n.y;
                        if (x >= 0 && x < sizeX && y >= 0 && y < sizeY) {
                            if (population[x][y] == STRATEGY_D) {
                                neighborsD.push_back({x, y});
                                if (!highestNeighborD || fitnesses[x][y] > fitnesses[highestNeighborD->x][highestNeighborD->y]) {
                                    hndStorage = {x, y};
                                    highestNeighborD = &hndStorage;
                                }
                            } else if (population[x][y] == STRATEGY_C) {
                                if (!highestNeighborC || fitnesses[x][y] > fitnesses[highestNeighborC->x][highestNeighborC->y]) {
                                    hncStorage = {x, y};
                                    highestNeighborC = &hncStorage;
                                }
                            }
                        }
                    }

                    if (highestNeighborD && highestNeighborC &&
                        fitnesses[highestNeighborD->x][highestNeighborD->y] >= fitnesses[highestNeighborC->x][highestNeighborC->y]) {

                        if (mode == "i") {
                            cost += (fitnesses[highestNeighborD->x][highestNeighborD->y] - fitnesses[i][j] + epsilon) / a;
                            fitnesses[i][j] = fitnesses[highestNeighborD->x][highestNeighborD->y] + epsilon;
                        } else if (mode == "ii") {
                            for (const auto& neighbor : neighborsD) {
                                Coord* highestNeighborNeighborD = nullptr;
                                Coord* highestNeighborNeighborC = nullptr;
                                Coord hnndStorage, hnncStorage;

                                std::vector<Coord> nnDirs = {
                                    {neighbor.x-1, neighbor.y}, {neighbor.x+1, neighbor.y},
                                    {neighbor.x, neighbor.y-1}, {neighbor.x, neighbor.y+1}
                                };
                                for (const auto& nn : nnDirs) {
                                    int nx = nn.x, ny = nn.y;
                                    if (nx >= 0 && nx < sizeX && ny >= 0 && ny < sizeY) {
                                        if (population[nx][ny] == STRATEGY_D) {
                                            if (!highestNeighborNeighborD || fitnesses[nx][ny] > fitnesses[highestNeighborNeighborD->x][highestNeighborNeighborD->y]) {
                                                hnndStorage = {nx, ny};
                                                highestNeighborNeighborD = &hnndStorage;
                                            }
                                        } else if (population[nx][ny] == STRATEGY_C) {
                                            if (!highestNeighborNeighborC || fitnesses[nx][ny] > fitnesses[highestNeighborNeighborC->x][highestNeighborNeighborC->y]) {
                                                hnncStorage = {nx, ny};
                                                highestNeighborNeighborC = &hnncStorage;
                                            }
                                        }
                                    }
                                }

                                if (highestNeighborNeighborD && highestNeighborNeighborC &&
                                    fitnesses[highestNeighborNeighborD->x][highestNeighborNeighborD->y] >= fitnesses[highestNeighborNeighborC->x][highestNeighborNeighborC->y]) {
                                    cost += (fitnesses[highestNeighborNeighborD->x][highestNeighborNeighborD->y] - fitnesses[i][j] + epsilon) / a;
                                    fitnesses[i][j] = fitnesses[highestNeighborNeighborD->x][highestNeighborNeighborD->y] + epsilon;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < sizeX; i++) {
            for (int j = 0; j < sizeY; j++) {
                if (population[i][j] == STRATEGY_C) {
                    auto neighbors = getNeighbors(i, j, sizeX, sizeY);
                    int neighborsCount = neighbors.size();

                    int countC = 0;
                    for (const auto& n : neighbors) {
                        if (population[n.x][n.y] == STRATEGY_C) countC++;
                    }

                    if (countC <= nc * (neighborsCount / 4.0)) {
                        cost += theta / a;
                        fitnesses[i][j] += theta;
                    }
                }
            }
        }
    }

    return {fitnesses, cost};
}

Population updatePopulation(const Population& population, const Fitnesses& fitnesses, double K, std::mt19937& rng) {
    Population newPop = copyPopulation(population);
    int sizeX = population.size();
    int sizeY = population[0].size();
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < sizeX; i++) {
        for (int j = 0; j < sizeY; j++) {
            double focalFitness = fitnesses[i][j];
            auto neighbors = getNeighbors(i, j, sizeX, sizeY);

            if (neighbors.empty()) continue;

            std::uniform_int_distribution<int> idxDist(0, neighbors.size() - 1);
            Coord neighborPos = neighbors[idxDist(rng)];
            double neighborFitness = fitnesses[neighborPos.x][neighborPos.y];

            double prob = fermi(focalFitness, neighborFitness, K);

            if (dist(rng) < prob) {
                newPop[i][j] = population[neighborPos.x][neighborPos.y];
            }
        }
    }

    return newPop;
}

struct SimulationResult {
    Population finalPopulation;
    std::vector<int> historyFrequency;
    std::vector<double> historyFitness;
    std::vector<double> historyCost;
    std::vector<double> historySocialWelfare;
};

SimulationResult simulatePopulation(
    int sizeX, int sizeY, int generations,
    double a,
    double initialCooperatorRatio, double beta,
    const std::string& strategy, double pc, int nc, double theta, double epsilon,
    const std::string& gameType, double r,
    std::mt19937& rng
) {
    Population population = newPopulation(sizeX, sizeY, initialCooperatorRatio, rng);

    std::vector<int> historyFrequency;
    std::vector<double> historyCost;
    std::vector<double> historyFitness;
    std::vector<double> historySocialWelfare;

    historyFrequency.reserve(generations);
    historyCost.reserve(generations);
    historyFitness.reserve(generations);
    historySocialWelfare.reserve(generations);

    for (int gen = 0; gen < generations; gen++) {
        int cooperatorCount = countCooperators(population);
        if (cooperatorCount == 0 || cooperatorCount == sizeX * sizeY) {
            break;
        }

        historyFrequency.push_back(cooperatorCount);

        Fitnesses fitnesses;
        if (gameType == "pgg") {
            fitnesses = calculateFitnessPGG(population, r);
        } else {
            fitnesses = calculateFitness(population, beta);
        }

        double cost = 0.0;
        if (strategy == "pop") {
            auto result = pop(population, fitnesses, static_cast<int>(pc * sizeX * sizeY), theta * a, a);
            fitnesses = result.first;
            cost = result.second;
        } else if (strategy == "neb-i") {
            auto result = neb("i", population, fitnesses, 0, epsilon * a, 0, a);
            fitnesses = result.first;
            cost = result.second;
        } else if (strategy == "neb-ii") {
            auto result = neb("ii", population, fitnesses, 0, epsilon * a, 0, a);
            fitnesses = result.first;
            cost = result.second;
        } else if (strategy == "neb") {
            auto result = neb("", population, fitnesses, nc, 0, theta * a, a);
            fitnesses = result.first;
            cost = result.second;
        }

        historyCost.push_back(cost);
        double totalFitness = sumFitnesses(fitnesses);
        historyFitness.push_back(totalFitness);
        historySocialWelfare.push_back(totalFitness - cost);

        population = updatePopulation(population, fitnesses, 0.3, rng);
    }

    return {population, historyFrequency, historyFitness, historyCost, historySocialWelfare};
}

std::string intVecToString(const std::vector<int>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) oss << ", ";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

std::string floatVecToString(const std::vector<double>& v) {
    std::ostringstream oss;
    oss << std::setprecision(15) << "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) oss << ", ";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

void writeCSV(const std::string& filename, const std::vector<std::string>& headers,
              const std::vector<std::vector<std::string>>& rows) {
    fs::path filePath(filename);
    fs::create_directories(filePath.parent_path());

    std::ofstream f(filename);
    for (size_t i = 0; i < headers.size(); i++) {
        if (i > 0) f << ",";
        f << headers[i];
    }
    f << "\n";

    for (const auto& row : rows) {
        for (size_t i = 0; i < row.size(); i++) {
            if (i > 0) f << ",";
            // Quote fields containing commas or brackets
            if (row[i].find(',') != std::string::npos || row[i].find('[') != std::string::npos) {
                f << "\"" << row[i] << "\"";
            } else {
                f << row[i];
            }
        }
        f << "\n";
    }
}

std::vector<double> arange(double start, double end, double step) {
    std::vector<double> result;
    for (double v = start; v <= end + step / 2; v += step) {
        result.push_back(v);
    }
    return result;
}

std::string getArg(int argc, char* argv[], const std::string& name, const std::string& defaultVal) {
    std::string prefix = "--" + name + "=";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.rfind(prefix, 0) == 0) {
            return arg.substr(prefix.size());
        }
    }
    return defaultVal;
}

int main(int argc, char* argv[]) {
    std::string strategy = getArg(argc, argv, "strategy", "pop");
    int nc = std::stoi(getArg(argc, argv, "nc", "4"));
    double pc = std::stod(getArg(argc, argv, "pc", "0.75"));
    std::string gameType = getArg(argc, argv, "game-type", "pd");
    double r = std::stod(getArg(argc, argv, "r", "3.0"));
    int seedStart = std::stoi(getArg(argc, argv, "seed-start", "0"));
    int seedEnd = std::stoi(getArg(argc, argv, "seed-end", "10"));
    double thetaStart = std::stod(getArg(argc, argv, "theta-start", "0.0"));
    double thetaEnd = std::stod(getArg(argc, argv, "theta-end", "10.0"));
    double thetaStep = std::stod(getArg(argc, argv, "theta-step", "0.1"));
    std::string outputDir = getArg(argc, argv, "output-dir", "data_cpp");
    double beta = std::stod(getArg(argc, argv, "beta", "1.8"));

    auto thetaList = arange(thetaStart, thetaEnd, thetaStep);

    for (int seed = seedStart; seed < seedEnd; seed++) {
        std::mt19937 rng(seed);

        std::vector<std::vector<std::string>> resFreq;
        std::vector<std::vector<std::string>> resSocialWelfare;
        std::vector<std::vector<std::string>> resCost;
        std::vector<std::vector<std::string>> resFitness;

        double a = 1.0;
        for (double theta : thetaList) {
            auto result = simulatePopulation(
                100, 100, 250,
                a,
                0.5, beta,
                strategy, pc, nc, theta, 4.5,
                gameType, r,
                rng
            );

            std::string seedStr = std::to_string(seed);
            std::string thetaStr = formatFloat(theta);
            std::string aStr = formatFloat(a);

            resFreq.push_back({seedStr, thetaStr, aStr, intVecToString(result.historyFrequency)});
            resSocialWelfare.push_back({seedStr, thetaStr, aStr, floatVecToString(result.historySocialWelfare)});
            resCost.push_back({seedStr, thetaStr, aStr, floatVecToString(result.historyCost)});
            resFitness.push_back({seedStr, thetaStr, aStr, floatVecToString(result.historyFitness)});
        }

        std::string dataDir;
        std::string paramStr;

        if (strategy == "pop") {
            if (gameType == "pgg") {
                dataDir = outputDir + "/" + strategy + "_theta_pgg_r=" + formatFloat(r);
            } else {
                dataDir = outputDir + "/" + strategy + "_theta_pd_b=" + formatFloat(beta);
            }
            paramStr = "pc=" + formatFloat(pc);
        } else if (strategy == "neb") {
            if (gameType == "pgg") {
                dataDir = outputDir + "/" + strategy + "_theta_pgg_r=" + formatFloat(r);
            } else {
                dataDir = outputDir + "/" + strategy + "_theta_pd_b=" + formatFloat(beta);
            }
            paramStr = "nc=" + std::to_string(nc);
        } else {
            if (gameType == "pgg") {
                dataDir = outputDir + "/" + strategy + "_theta_pgg_r=" + formatFloat(r);
            } else {
                dataDir = outputDir + "/" + strategy + "_theta_pd_b=" + formatFloat(beta);
            }
            paramStr = "pc=" + formatFloat(pc);
        }

        std::ostringstream thetaRangeOss;
        thetaRangeOss << std::fixed << std::setprecision(1) << "theta_" << thetaStart << "-" << thetaEnd;
        std::string thetaRangeStr = thetaRangeOss.str();

        std::string filePrefix = dataDir + "/seed_" + std::to_string(seed) + "_" + thetaRangeStr + "_" + paramStr;

        writeCSV(filePrefix + "_cooperator_frequency.csv", {"Seed", "Theta", "A", "Cooperator_Frequency"}, resFreq);
        writeCSV(filePrefix + "_social_welfare.csv", {"Seed", "Theta", "A", "Social_Welfare"}, resSocialWelfare);
        writeCSV(filePrefix + "_cost.csv", {"Seed", "Theta", "A", "Cost"}, resCost);
        writeCSV(filePrefix + "_population_payoff.csv", {"Seed", "Theta", "A", "Payoff"}, resFitness);
    }

    return 0;
}
