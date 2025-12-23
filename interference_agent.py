def pop(population, fitnesses, pc, theta=4.5, a=1):
    cost = 0
    xc = sum((sum(row == 'C') for row in population))
    if xc < pc:
        fitnesses[population == 'C'] += theta
        cost += (theta / a) * xc
    return fitnesses, cost

def neb(mode, population, fitnesses, nc, epsilon=4.5, theta=4.5, a=1):
    cost = 0
    if mode == 'i' or mode == 'ii':
        for i in range(population.shape[0]):
            for j in range(population.shape[1]):
                if population[i, j] == 'C':
                    neighbors_d = []
                    highest_neighbor_d = None
                    highest_neighbor_c = None
                    for x, y in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                        if 0 <= x < population.shape[0] and 0 <= y < population.shape[1]:
                            if population[x, y] == 'D':
                                neighbors_d.append((x, y))
                                if highest_neighbor_d is None or fitnesses[x, y] > fitnesses[highest_neighbor_d]:
                                    highest_neighbor_d = (x, y)
                            elif population[x, y] == 'C':
                                if highest_neighbor_c is None or fitnesses[x, y] > fitnesses[highest_neighbor_c]:
                                    highest_neighbor_c = (x, y)

                    if highest_neighbor_d is not None and highest_neighbor_c is not None and fitnesses[highest_neighbor_d] >= fitnesses[highest_neighbor_c]:
                        if mode == 'i':
                            cost += (fitnesses[highest_neighbor_d] - fitnesses[i, j] + epsilon) / a
                            fitnesses[i, j] = fitnesses[highest_neighbor_d] + epsilon
                        elif mode == 'ii':
                            for neighbor in neighbors_d:
                                highest_neighbor_neighbor_d = None
                                highest_neighbor_neighbor_c = None
                                for nx, ny in [(neighbor[0]-1, neighbor[1]), (neighbor[0]+1, neighbor[1]), (neighbor[0], neighbor[1]-1), (neighbor[0], neighbor[1]+1)]:
                                    if 0 <= nx < population.shape[0] and 0 <= ny < population.shape[1]:
                                        if population[nx, ny] == 'D':
                                            if highest_neighbor_neighbor_d is None or fitnesses[nx, ny] > fitnesses[highest_neighbor_neighbor_d]:
                                                highest_neighbor_neighbor_d = (nx, ny)
                                        elif population[nx, ny] == 'C':
                                            if highest_neighbor_neighbor_c is None or fitnesses[nx, ny] > fitnesses[highest_neighbor_neighbor_c]:
                                                highest_neighbor_neighbor_c = (nx, ny)
                                if highest_neighbor_neighbor_d is not None and highest_neighbor_neighbor_c is not None and fitnesses[highest_neighbor_neighbor_d] >= fitnesses[highest_neighbor_neighbor_c]:
                                    cost += (fitnesses[highest_neighbor_neighbor_d] - fitnesses[i, j] + epsilon) / a
                                    fitnesses[i, j] = fitnesses[highest_neighbor_neighbor_d] + epsilon

    else:
        for i in range(population.shape[0]):
            for j in range(population.shape[1]):
                if population[i, j] == 'C':
                    neighbors = []
                    neighbors_count = 0
                    if i > 0:
                        neighbors.append((i - 1, j))
                        neighbors_count += 1
                    if i < population.shape[0] - 1:
                        neighbors.append((i + 1, j))
                        neighbors_count += 1
                    if j > 0:
                        neighbors.append((i, j - 1))
                        neighbors_count += 1    
                    if j < population.shape[1] - 1:
                        neighbors.append((i, j + 1))
                        neighbors_count += 1
                    
                    count_C = sum(1 for x, y in neighbors if population[x, y] == 'C')
                    
                    if count_C <= nc * (neighbors_count / 4):
                        cost += theta / a
                        fitnesses[i, j] += theta
                    
    return fitnesses, cost 