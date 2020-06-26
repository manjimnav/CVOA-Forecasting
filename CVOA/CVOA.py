from CVOA.Individual import Individual
from copy import deepcopy
import numpy as np
import sys as sys
import random as random
from multiprocessing import Pool
import math


class CVOA:
    bestSolution = None
    bestModel = None
    MIN_SPREAD = 0
    MAX_SPREAD = 5
    MIN_SUPERSPREAD = 6
    MAX_SUPERSPREAD = 25
    P_TRAVEL = 0.1
    P_REINFECTION = 0.01
    SUPERSPREADER_PERC = 0.04
    DEATH_PERC = 0.5

    def __init__(self, size_fixed_part, fixed_part_max_values,fixed_part_min_values,
                 max_time, processes=2, use_var_part=True,discrete=True, fitness=None,var_part_max_value=None,
                 min_size_var_part=None, max_size_var_part=None, initial_fixed_part=None, initial_var_part=None):
        self.infected = []
        self.recovered = []
        self.deaths = []
        self.min_size_var_part = min_size_var_part
        self.max_size_var_part = max_size_var_part
        self.size_fixed_part = size_fixed_part
        self.fixed_part_max_values = fixed_part_max_values
        self.fixed_part_min_values = fixed_part_min_values
        self.var_part_max_value = var_part_max_value
        self.max_time = max_time

        self.processes = processes
        self.use_var_part = use_var_part
        self.discrete = discrete
        self.fitness = fitness
        self.initial_fixed_part = initial_fixed_part
        self.initial_var_part = initial_var_part

    def calcSearchSpaceSize(self):
        """
        :return: Total search space
        """
        t = 1
        res = 0
        # Part 1. Fixed part possible combinations.
        for i in range(len(self.fixed_part_max_values)):
            t *= (self.fixed_part_max_values[i] - self.fixed_part_min_values[i]) + 1
        res += t
        if self.use_var_part:
            res *= self.max_size_var_part
            # Part 2. Var part possible combinations.
            res *= pow(self.var_part_max_value, self.max_size_var_part)
        return res

    def run_fitness(self, pop):
        return [self.fitness(x) for x in pop]

    def propagateDisease(self):
        new_infected_list = []
        # Step 1. Assess fitness for each individual.

        pool = Pool(processes=self.processes)
        items_per_group = math.ceil(len(self.infected) / self.processes)
        #results = pool.map(self.run_fitness, [self.infected[i:i + items_per_group] for i in
                                              #range(0, len(self.infected), items_per_group)])
        #for i, ind in enumerate((item for sublist in results for item in sublist)):

           # self.infected[i] = ind
        for ind in self.infected:
            ind = self.fitness(ind)
            # If x.fitness is NaN, move from infected list to deaths lists
            if np.isnan(ind.fitness):
                self.deaths.append(ind)
                self.infected.remove(ind)


        # Step 2. Sort the infected list by fitness (ascendent).
        self.infected = sorted(self.infected, key=lambda i: i.fitness)
        # Step 3. Update best global solution, if proceed.
        if self.bestSolution.fitness is None or self.infected[0].fitness < self.bestSolution.fitness:
            # self.bestModel = model
            # model.save("bestModel.h5")
            self.bestSolution = deepcopy(self.infected[0])

        #resetTF()  # Release GPU memory
        # Step 4. Assess indexes to point super-spreaders and deaths parts of the infected list.
        if len(self.infected) == 1:
            idx_super_spreader = 1
        else:
            idx_super_spreader = self.SUPERSPREADER_PERC * len(self.infected)
        if len(self.infected) == 1:
            idx_deaths = sys.maxsize
        else:
            idx_deaths = len(self.infected) - (self.DEATH_PERC * len(self.infected))

        # Step 5. Disease propagation.
        i = 0
        for x in self.infected:
            # Step 5.1 If the individual belongs to the death part, then die!
            if i >= idx_deaths:
                self.deaths.append(x)
            else:
                # Step 5.2 Determine the number of new infected individuals.
                if i < idx_super_spreader:  # This is the super-spreader!
                    ninfected = self.MIN_SUPERSPREAD + random.randint(0, self.MAX_SUPERSPREAD - self.MIN_SUPERSPREAD)
                else:
                    ninfected = random.randint(0, self.MAX_SPREAD)
                # Step 5.3 Determine whether the individual has traveled
                if random.random() < self.P_TRAVEL:
                    traveler = True
                else:
                    traveler = False
                # Step 5.4 Determine the travel distance, which is how far is the new infected individual.
                if traveler:
                    travel_distance = -1  # The value -1 is to indicate that all the individual elements can be affected.
                else:
                    travel_distance = 1
                # Step 5.5 Infect!!
                for j in range(ninfected):
                    new_infected = x.infect(
                        travel_distance=travel_distance)  # new_infected = infect(x, travel_distance)
                    if new_infected not in self.deaths and new_infected not in self.infected and new_infected not in new_infected_list and new_infected not in self.recovered:
                        new_infected_list.append(new_infected)
                    elif new_infected in self.recovered and new_infected not in new_infected_list:
                        if random.random() < self.P_REINFECTION:
                            new_infected_list.append(new_infected)
                            self.recovered.remove(new_infected)
            i += 1
        # Step 6. Add the current infected individuals to the recovered list.
        self.recovered.extend(self.infected)
        # Step 7. Update the infected list with the new infected individuals.
        self.infected = new_infected_list

    def run(self):
        epidemic = True
        time = 0
        # Step 1. Infect to Patient Zero
        """pz = Individual.random(size_fixed_part=self.size_fixed_part, min_size_var_part=self.min_size_var_part,
                               max_size_var_part=self.max_size_var_part,
                               fixed_part_max_values=self.fixed_part_max_values,
                               var_part_max_value=self.var_part_max_value, use_var_part=self.use_var_part,
                               discrete=self.discrete)
        """
        # custom pz
        pz = Individual.random(size_fixed_part=self.size_fixed_part, min_size_var_part=self.min_size_var_part,
                                       max_size_var_part=self.max_size_var_part,
                                       fixed_part_max_values=self.fixed_part_max_values,
                                    fixed_part_min_values=self.fixed_part_min_values,
                                       var_part_max_value=self.var_part_max_value, use_var_part=self.use_var_part,
                                       discrete=self.discrete)

        if self.initial_fixed_part is not None:
            pz.fixed_part = self.initial_fixed_part

        if self.initial_var_part is not None:
            #pz.fixed_part = [100.24, 20.3]
            pz.var_part = self.initial_var_part
        self.infected.append(pz)
        print("Patient Zero: " + str(pz) + "\n")
        self.bestSolution = deepcopy(pz)
        # Step 2. The main loop for the disease propagation
        total_ss = self.calcSearchSpaceSize()
        while epidemic and time < self.max_time:
            self.propagateDisease()

            """
            if self.use_generator:
                mse, mape = getMetrics_denormalized(self.bestModel, self.batch, self.scaler,
                                                    val_gen=self.valid_generator)
                mse = mse[0]
            else:
                mse, dmape = getMetrics_denormalized(model=self.bestModel, xval=self.xval, yval=self.yval,
                                                     batch=self.batch, scaler=self.scaler)
            """

            print("Iteration ", (time + 1))
            # print("Best fitness so far: ", "{:.4f}".format(self.bestSolution.fitness))
            print("Best fitness so far: ", "{:.4f}".format(self.bestSolution.fitness))
            print("Best individual: ", self.bestSolution)
            print("Infected: ", str(len(self.infected)), "; Recovered: ", str(len(self.recovered)), "; Deaths: ",
                  str(len(self.deaths)))
            print(
                "Recovered/Infected: " + str("{:.4f}".format(100 * ((len(self.recovered)) / (len(self.infected)+1)))) + "%")
            current_ss = len(self.infected) + len(self.recovered) + len(self.deaths)
            print("Search space covered so far = " + str(current_ss) + " / " + str(total_ss) + " = " +
                  str("{:.4f}".format(100 * (current_ss) / total_ss)) + "%\n")
            if not self.infected:
                epidemic = False
            time += 1

        return self.bestSolution
