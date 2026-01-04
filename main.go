package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	StrategyC = 'C'
	StrategyD = 'D'
)

type Population [][]byte
type Fitnesses [][]float64

func newPopulation(sizeX, sizeY int, initialCooperatorRatio float64, rng *rand.Rand) Population {
	pop := make(Population, sizeX)
	for i := range pop {
		pop[i] = make([]byte, sizeY)
		for j := range pop[i] {
			if rng.Float64() < initialCooperatorRatio {
				pop[i][j] = StrategyC
			} else {
				pop[i][j] = StrategyD
			}
		}
	}
	return pop
}

func (p Population) copy() Population {
	cp := make(Population, len(p))
	for i := range p {
		cp[i] = make([]byte, len(p[i]))
		copy(cp[i], p[i])
	}
	return cp
}

func (p Population) countCooperators() int {
	count := 0
	for i := range p {
		for j := range p[i] {
			if p[i][j] == StrategyC {
				count++
			}
		}
	}
	return count
}

func newFitnesses(sizeX, sizeY int) Fitnesses {
	f := make(Fitnesses, sizeX)
	for i := range f {
		f[i] = make([]float64, sizeY)
	}
	return f
}

func (f Fitnesses) sum() float64 {
	total := 0.0
	for i := range f {
		for j := range f[i] {
			total += f[i][j]
		}
	}
	return total
}

func (f Fitnesses) copy() Fitnesses {
	cp := make(Fitnesses, len(f))
	for i := range f {
		cp[i] = make([]float64, len(f[i]))
		copy(cp[i], f[i])
	}
	return cp
}

func getNeighbors(i, j, sizeX, sizeY int) [][2]int {
	neighbors := make([][2]int, 0, 4)
	if i > 0 {
		neighbors = append(neighbors, [2]int{i - 1, j})
	}
	if i < sizeX-1 {
		neighbors = append(neighbors, [2]int{i + 1, j})
	}
	if j > 0 {
		neighbors = append(neighbors, [2]int{i, j - 1})
	}
	if j < sizeY-1 {
		neighbors = append(neighbors, [2]int{i, j + 1})
	}
	return neighbors
}

func fermi(focal, neighbor, K float64) float64 {
	return 1.0 / (1.0 + math.Exp((focal-neighbor)/K))
}

func calculateFitness(population Population, beta float64) Fitnesses {
	sizeX := len(population)
	sizeY := len(population[0])
	fitnesses := newFitnesses(sizeX, sizeY)

	payoffMatrix := map[byte]map[byte]float64{
		StrategyC: {StrategyC: 1.0, StrategyD: 0.0},
		StrategyD: {StrategyC: beta, StrategyD: 0.0},
	}

	for i := 0; i < sizeX; i++ {
		for j := 0; j < sizeY; j++ {
			strategy := population[i][j]
			totalPayoff := 0.0
			neighbors := getNeighbors(i, j, sizeX, sizeY)

			for _, n := range neighbors {
				neighborStrategy := population[n[0]][n[1]]
				totalPayoff += payoffMatrix[strategy][neighborStrategy]
			}
			fitnesses[i][j] = totalPayoff
		}
	}
	return fitnesses
}

func calculateFitnessPGG(population Population, r float64) Fitnesses {
	sizeX := len(population)
	sizeY := len(population[0])
	fitnesses := newFitnesses(sizeX, sizeY)

	for i := 0; i < sizeX; i++ {
		for j := 0; j < sizeY; j++ {
			neighbors := getNeighbors(i, j, sizeX, sizeY)

			nCoop := 0
			for _, n := range neighbors {
				if population[n[0]][n[1]] == StrategyC {
					nCoop++
				}
			}

			contributionCost := 0.0
			if population[i][j] == StrategyC {
				nCoop++
				contributionCost = 1.0
			}

			groupSize := float64(len(neighbors) + 1)
			fitnesses[i][j] = (float64(nCoop) * r / groupSize) - contributionCost
		}
	}
	return fitnesses
}

func pop(population Population, fitnesses Fitnesses, pc int, theta, a float64) (Fitnesses, float64) {
	cost := 0.0
	xc := population.countCooperators()

	if xc < pc {
		sizeX := len(population)
		sizeY := len(population[0])
		for i := 0; i < sizeX; i++ {
			for j := 0; j < sizeY; j++ {
				if population[i][j] == StrategyC {
					fitnesses[i][j] += theta
				}
			}
		}
		cost = (theta / a) * float64(xc)
	}
	return fitnesses, cost
}

func neb(mode string, population Population, fitnesses Fitnesses, nc int, epsilon, theta, a float64) (Fitnesses, float64) {
	cost := 0.0
	sizeX := len(population)
	sizeY := len(population[0])

	if mode == "i" || mode == "ii" {
		for i := 0; i < sizeX; i++ {
			for j := 0; j < sizeY; j++ {
				if population[i][j] == StrategyC {
					var highestNeighborD *[2]int
					var highestNeighborC *[2]int
					var neighborsD [][2]int

					for _, n := range [][2]int{{i - 1, j}, {i + 1, j}, {i, j - 1}, {i, j + 1}} {
						x, y := n[0], n[1]
						if x >= 0 && x < sizeX && y >= 0 && y < sizeY {
							if population[x][y] == StrategyD {
								neighborsD = append(neighborsD, n)
								if highestNeighborD == nil || fitnesses[x][y] > fitnesses[highestNeighborD[0]][highestNeighborD[1]] {
									highestNeighborD = &[2]int{x, y}
								}
							} else if population[x][y] == StrategyC {
								if highestNeighborC == nil || fitnesses[x][y] > fitnesses[highestNeighborC[0]][highestNeighborC[1]] {
									highestNeighborC = &[2]int{x, y}
								}
							}
						}
					}

					if highestNeighborD != nil && highestNeighborC != nil &&
						fitnesses[highestNeighborD[0]][highestNeighborD[1]] >= fitnesses[highestNeighborC[0]][highestNeighborC[1]] {

						if mode == "i" {
							cost += (fitnesses[highestNeighborD[0]][highestNeighborD[1]] - fitnesses[i][j] + epsilon) / a
							fitnesses[i][j] = fitnesses[highestNeighborD[0]][highestNeighborD[1]] + epsilon
						} else if mode == "ii" {
							for _, neighbor := range neighborsD {
								var highestNeighborNeighborD *[2]int
								var highestNeighborNeighborC *[2]int

								for _, nn := range [][2]int{
									{neighbor[0] - 1, neighbor[1]},
									{neighbor[0] + 1, neighbor[1]},
									{neighbor[0], neighbor[1] - 1},
									{neighbor[0], neighbor[1] + 1},
								} {
									nx, ny := nn[0], nn[1]
									if nx >= 0 && nx < sizeX && ny >= 0 && ny < sizeY {
										if population[nx][ny] == StrategyD {
											if highestNeighborNeighborD == nil || fitnesses[nx][ny] > fitnesses[highestNeighborNeighborD[0]][highestNeighborNeighborD[1]] {
												highestNeighborNeighborD = &[2]int{nx, ny}
											}
										} else if population[nx][ny] == StrategyC {
											if highestNeighborNeighborC == nil || fitnesses[nx][ny] > fitnesses[highestNeighborNeighborC[0]][highestNeighborNeighborC[1]] {
												highestNeighborNeighborC = &[2]int{nx, ny}
											}
										}
									}
								}

								if highestNeighborNeighborD != nil && highestNeighborNeighborC != nil &&
									fitnesses[highestNeighborNeighborD[0]][highestNeighborNeighborD[1]] >= fitnesses[highestNeighborNeighborC[0]][highestNeighborNeighborC[1]] {
									cost += (fitnesses[highestNeighborNeighborD[0]][highestNeighborNeighborD[1]] - fitnesses[i][j] + epsilon) / a
									fitnesses[i][j] = fitnesses[highestNeighborNeighborD[0]][highestNeighborNeighborD[1]] + epsilon
								}
							}
						}
					}
				}
			}
		}
	} else {
		for i := 0; i < sizeX; i++ {
			for j := 0; j < sizeY; j++ {
				if population[i][j] == StrategyC {
					neighbors := getNeighbors(i, j, sizeX, sizeY)
					neighborsCount := len(neighbors)

					countC := 0
					for _, n := range neighbors {
						if population[n[0]][n[1]] == StrategyC {
							countC++
						}
					}

					if float64(countC) <= float64(nc)*(float64(neighborsCount)/4.0) {
						cost += theta / a
						fitnesses[i][j] += theta
					}
				}
			}
		}
	}

	return fitnesses, cost
}

func updatePopulation(population Population, fitnesses Fitnesses, K float64, rng *rand.Rand) Population {
	newPop := population.copy()
	sizeX := len(population)
	sizeY := len(population[0])

	for i := 0; i < sizeX; i++ {
		for j := 0; j < sizeY; j++ {
			focalFitness := fitnesses[i][j]
			neighbors := getNeighbors(i, j, sizeX, sizeY)

			if len(neighbors) == 0 {
				continue
			}

			neighborPos := neighbors[rng.Intn(len(neighbors))]
			neighborFitness := fitnesses[neighborPos[0]][neighborPos[1]]

			prob := fermi(focalFitness, neighborFitness, K)

			if rng.Float64() < prob {
				newPop[i][j] = population[neighborPos[0]][neighborPos[1]]
			}
		}
	}

	return newPop
}

type SimulationResult struct {
	FinalPopulation      Population
	HistoryFrequency     []int
	HistoryFitness       []float64
	HistoryCost          []float64
	HistorySocialWelfare []float64
}

func simulatePopulation(
	sizeX, sizeY, generations int,
	a float64,
	initialCooperatorRatio, beta float64,
	strategy string, pc float64, nc int, theta, epsilon float64,
	gameType string, r float64,
	rng *rand.Rand,
) SimulationResult {
	population := newPopulation(sizeX, sizeY, initialCooperatorRatio, rng)

	historyFrequency := make([]int, 0, generations)
	historyCost := make([]float64, 0, generations)
	historyFitness := make([]float64, 0, generations)
	historySocialWelfare := make([]float64, 0, generations)

	for gen := 0; gen < generations; gen++ {
		cooperatorCount := population.countCooperators()
		if cooperatorCount == 0 || cooperatorCount == sizeX*sizeY {
			break
		}

		historyFrequency = append(historyFrequency, cooperatorCount)

		var fitnesses Fitnesses
		if gameType == "pgg" {
			fitnesses = calculateFitnessPGG(population, r)
		} else {
			fitnesses = calculateFitness(population, beta)
		}

		var cost float64
		switch strategy {
		case "pop":
			fitnesses, cost = pop(population, fitnesses, int(pc*float64(sizeX*sizeY)), theta*a, a)
		case "neb-i":
			fitnesses, cost = neb("i", population, fitnesses, 0, epsilon*a, 0, a)
		case "neb-ii":
			fitnesses, cost = neb("ii", population, fitnesses, 0, epsilon*a, 0, a)
		case "neb":
			fitnesses, cost = neb("", population, fitnesses, nc, 0, theta*a, a)
		}

		historyCost = append(historyCost, cost)
		totalFitness := fitnesses.sum()
		historyFitness = append(historyFitness, totalFitness)
		historySocialWelfare = append(historySocialWelfare, totalFitness-cost)

		population = updatePopulation(population, fitnesses, 0.3, rng)
	}

	return SimulationResult{
		FinalPopulation:      population,
		HistoryFrequency:     historyFrequency,
		HistoryFitness:       historyFitness,
		HistoryCost:          historyCost,
		HistorySocialWelfare: historySocialWelfare,
	}
}

func intSliceToStrings(s []int) []string {
	result := make([]string, len(s))
	for i, v := range s {
		result[i] = strconv.Itoa(v)
	}
	return result
}

func floatSliceToStrings(s []float64) []string {
	result := make([]string, len(s))
	for i, v := range s {
		result[i] = strconv.FormatFloat(v, 'f', -1, 64)
	}
	return result
}

func writeCSV(filename string, headers []string, rows [][]string) error {
	dir := filepath.Dir(filename)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}

	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	if err := w.Write(headers); err != nil {
		return err
	}
	for _, row := range rows {
		if err := w.Write(row); err != nil {
			return err
		}
	}
	return nil
}

func arange(start, end, step float64) []float64 {
	var result []float64
	for v := start; v <= end+step/2; v += step {
		result = append(result, v)
	}
	return result
}

func main() {
	strategy := flag.String("strategy", "pop", "Interference strategy to use (pop, neb, neb-i, neb-ii)")
	nc := flag.Int("nc", 4, "Number of cooperators for NEB strategy")
	pc := flag.Float64("pc", 0.75, "Cooperation threshold for POP strategy")
	gameType := flag.String("game-type", "pd", "Game type: pd (Prisoner's Dilemma) or pgg (Public Goods Game)")
	r := flag.Float64("r", 3.0, "Multiplication factor for Public Goods Game")
	seedStart := flag.Int("seed-start", 0, "Starting seed for simulations")
	seedEnd := flag.Int("seed-end", 10, "Ending seed for simulations")
	thetaStart := flag.Float64("theta-start", 0.0, "Starting theta for range")
	thetaEnd := flag.Float64("theta-end", 10.0, "Ending theta for range")
	thetaStep := flag.Float64("theta-step", 0.1, "Theta step size")
	outputDir := flag.String("output-dir", "data_go", "Output directory for CSV files")
	beta := flag.Float64("beta", 1.8, "Beta value for Prisoner's Dilemma payoff matrix")

	flag.Parse()

	thetaList := arange(*thetaStart, *thetaEnd, *thetaStep)

	for seed := *seedStart; seed < *seedEnd; seed++ {
		rng := rand.New(rand.NewSource(int64(seed)))

		var resFreq [][]string
		var resSocialWelfare [][]string
		var resCost [][]string
		var resFitness [][]string

		a := 1.0
		for _, theta := range thetaList {
			result := simulatePopulation(
				100, 100, 250,
				a,
				0.5, *beta,
				*strategy, *pc, *nc, theta, 4.5,
				*gameType, *r,
				rng,
			)

			freqStr := "[" + strings.Join(intSliceToStrings(result.HistoryFrequency), ", ") + "]"
			swStr := "[" + strings.Join(floatSliceToStrings(result.HistorySocialWelfare), ", ") + "]"
			costStr := "[" + strings.Join(floatSliceToStrings(result.HistoryCost), ", ") + "]"
			fitStr := "[" + strings.Join(floatSliceToStrings(result.HistoryFitness), ", ") + "]"

			seedStr := strconv.Itoa(seed)
			thetaStr := strconv.FormatFloat(theta, 'f', -1, 64)
			aStr := strconv.FormatFloat(a, 'f', -1, 64)

			resFreq = append(resFreq, []string{seedStr, thetaStr, aStr, freqStr})
			resSocialWelfare = append(resSocialWelfare, []string{seedStr, thetaStr, aStr, swStr})
			resCost = append(resCost, []string{seedStr, thetaStr, aStr, costStr})
			resFitness = append(resFitness, []string{seedStr, thetaStr, aStr, fitStr})
		}

		var dataDir string
		var paramStr string

		if *strategy == "pop" {
			if *gameType == "pgg" {
				dataDir = fmt.Sprintf("%s/%s_theta_pgg_r=%g", *outputDir, *strategy, *r)
			} else {
				dataDir = fmt.Sprintf("%s/%s_theta_pd_b=%g", *outputDir, *strategy, *beta)
			}
			paramStr = fmt.Sprintf("pc=%g", *pc)
		} else if *strategy == "neb" {
			if *gameType == "pgg" {
				dataDir = fmt.Sprintf("%s/%s_theta_pgg_r=%g", *outputDir, *strategy, *r)
			} else {
				dataDir = fmt.Sprintf("%s/%s_theta_pd_b=%g", *outputDir, *strategy, *beta)
			}
			paramStr = fmt.Sprintf("nc=%d", *nc)
		} else {
			if *gameType == "pgg" {
				dataDir = fmt.Sprintf("%s/%s_theta_pgg_r=%g", *outputDir, *strategy, *r)
			} else {
				dataDir = fmt.Sprintf("%s/%s_theta_pd_b=%g", *outputDir, *strategy, *beta)
			}
			paramStr = fmt.Sprintf("pc=%g", *pc)
		}

		thetaRangeStr := fmt.Sprintf("theta_%.1f-%.1f", *thetaStart, *thetaEnd)
		filePrefix := fmt.Sprintf("%s/seed_%d_%s_%s", dataDir, seed, thetaRangeStr, paramStr)

		headers := []string{"Seed", "Theta", "A", "Cooperator_Frequency"}
		if err := writeCSV(filePrefix+"_cooperator_frequency.csv", headers, resFreq); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing frequency CSV: %v\n", err)
		}

		headers = []string{"Seed", "Theta", "A", "Social_Welfare"}
		if err := writeCSV(filePrefix+"_social_welfare.csv", headers, resSocialWelfare); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing social welfare CSV: %v\n", err)
		}

		headers = []string{"Seed", "Theta", "A", "Cost"}
		if err := writeCSV(filePrefix+"_cost.csv", headers, resCost); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing cost CSV: %v\n", err)
		}

		headers = []string{"Seed", "Theta", "A", "Payoff"}
		if err := writeCSV(filePrefix+"_population_payoff.csv", headers, resFitness); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing payoff CSV: %v\n", err)
		}
	}
}
