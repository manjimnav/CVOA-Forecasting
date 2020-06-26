from CVOA.CVOA import CVOA
import argparse
import time
import math
from numpy import prod, sum


def fitness_base(individual):
    individual.fitness = individual.fixed_part[0] ** 2 + individual.fixed_part[1] ** 2
    return individual


def sphere(individual):

    individual.fitness = sum([el**2 for el in individual.fixed_part])

    return individual


def genrosenbrock(individual):
    individual.fitness = sum([abs(el) for i, el in enumerate(individual.fixed_part[:-1])]) + prod([abs(el) for i, el in enumerate(individual.fixed_part[:-1])])

    return individual


def schwefel(individual):
    alpha = 418.982887
    fitness = 0
    for i in range(len(individual.fixed_part)):
        fitness -= individual.fixed_part[i] * math.sin(math.sqrt(math.fabs(individual.fixed_part[i])))
    individual.fitness = float(fitness) + alpha * len(individual.fixed_part)
    return individual


def genschwefel(individual):
    alpha = 418.982887
    fitness = 0
    for i in range(len(individual.fixed_part)):
        fitness -= individual.fixed_part[i] * math.sin(math.sqrt(math.fabs(individual.fixed_part[i])))
    individual.fitness = float(fitness) + alpha * len(individual.fixed_part)
    return individual


def ackley(individual):
    firstSum = 0.0
    secondSum = 0.0
    for c in individual.fixed_part:
        firstSum += c ** 2.0
        secondSum += math.cos(2.0 * math.pi * c)
    n = float(len(individual.fixed_part))
    individual.fitness = -20.0 * math.exp(-0.2 * math.sqrt(firstSum / n)) - math.exp(secondSum / n) + 20 + math.e

    return individual


def get_fitness(function):
    fitness = None
    if function == 'sphere':
        fitness = sphere
    elif function == 'genrosenbrock':
        fitness = genrosenbrock
    elif function == 'schwefel':
        fitness = schwefel
    elif function == 'genschwefel':
        fitness = genschwefel
    elif function == 'ackley':
        fitness = ackley

    return fitness


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--processes", help="Number of processes (Default 2)",
                        type=int, default=2)
    parser.add_argument("-uv", "--use_variable", help="Use variable size in optimization",
                        action="store_true", default=False)
    parser.add_argument("-di", "--discrete", help="Discrete optimization problem",
                        action="store_true", default=False)
    parser.add_argument("-f", "--function", help="Type of function to optimize",
                        choices=['sphere', 'rosenbrock', 'ackley', 'schwefel', 'genschwefel',
                                 'genrastrigin', 'gengriewank', 'weierstrass'], default='sphere')
    parser.add_argument("-d", "--dimmension", help="Dimmension of the functions",
                        type=int, default=10)
    parser.add_argument("-mx", "--max", help="Max for all dimmensions",
                        type=int, default=100)
    parser.add_argument("-mn", "--min", help="Min for all dimmensions",
                        type=int, default=-100)
    parser.add_argument("-t", "--time", help="Max time",
                        type=int, default=100)

    args = parser.parse_args()

    fitness = get_fitness(args.function)

    print("initializing...")
    max_values = [args.max for _ in range(args.dimmension)]
    min_values = [args.min for _ in range(args.dimmension)]

    cvoa = CVOA(size_fixed_part=args.dimmension, fixed_part_max_values=max_values, fixed_part_min_values=min_values, max_time=args.time,
                use_var_part=args.use_variable, discrete=args.discrete, fitness=fitness, processes=args.processes)

    start_time = int(round(time.time() * 1000))
    solution = cvoa.run()
    delta_time = int(round(time.time() * 1000)) - start_time

    print("Best solution: " + str(solution))
    print("Best fitness: " + str(fitness(solution)))
    print("Execution time: " + str(round(delta_time / 60000, 2)) + " mins")


if __name__ == '__main__':
    main()
