from CVOA.CVOA import CVOA
import argparse
import time


def fitness(individual):
    individual.fitness = individual.fixed_part[0] ** 2 + individual.fixed_part[1] ** 2
    return individual


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--processes", help="Number of processes (Default 2)",
                        type=int, default=2)
    parser.add_argument("-uv", "--use_variable", help="Use variable size in optimization",
                        action="store_true", default=False)
    parser.add_argument("-di", "--discrete", help="Discrete optimization problem",
                        action="store_true", default=False)

    args = parser.parse_args()
    print("initializing...")
    cvoa = CVOA(size_fixed_part=2, fixed_part_max_values=[100, 200], max_time=20,
                use_var_part=args.use_variable, discrete=args.discrete, fitness=fitness)

    start_time = int(round(time.time() * 1000))
    solution = cvoa.run()
    delta_time = int(round(time.time() * 1000)) - start_time

    print("Best solution: " + str(solution))
    print("Best fitness: " + str(fitness(solution)))
    print("Execution time: " + str(round(delta_time / 60000, 2)) + " mins")

if __name__ == '__main__':
    main()