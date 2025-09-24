# General framework for genetic algorithm:
#
# Generate initial population
# For each generation:
#   Evaluate each individual
#   Save the elite
#   Genetic operators:
#       Crossover
#       Mutation
#       Selection
#   Generate new population

# Generate initial population


from enum import Enum
from random import random, sample, shuffle
from typing import Callable
from solver import Solver
from variable import Variable
from restriction import Restriction
from copy import deepcopy
from itertools import accumulate
from number import Number


# Chromosome is a list of assigned values for the variables
type Chromosome = list[Number]


# A member of the population has two properties:
# The chromosome itself and its fitness
class PopulationMember:
    chromosome: Chromosome
    fitness: float

    def __init__(self, chromosome: Chromosome, fitness: float) -> None:
        self.chromosome = chromosome
        self.fitness = fitness


# The population in merely a list of members
type Population = list[PopulationMember]


# The method by which individuals are selected for the next generation
class SelectionMethods(Enum):
    TOURNAMENT = 0
    RAFFLE = 1


class GeneticAlgorithm(Solver):
    # Genetic operator for mutations.
    # Returns a procedural new chromosome
    # based on the input one.
    mutation_op: Callable[[Chromosome], Chromosome] | None

    # Probability of mutating a member of the augmented population
    mutation_prob: float

    # Genetic operator for crossover.
    # Gets two chromosomes as input and generates
    # two child chromosomes as a result
    crossover_op: (
        Callable[[Chromosome, Chromosome],
                 tuple[Chromosome, Chromosome]] | None
    )

    # Criteria for stopping the process.
    # Parameters are the number of generations and
    # the current population at the time of evaluation
    stop_criteria: Callable[[int, Population], bool]

    # The number of individuals in a population
    population_size: int

    # The number of individuals to be preserved unchanged
    # for the next generation
    elite_size: int

    # Function to determine likeness of an individual
    # to be selected for the next generation
    fitness: Callable[[Chromosome], float]

    # Function to generate chromosomes for the initial population
    initial_chromosome_func: Callable[[list[Variable]], Chromosome]

    # Method for selection at the end of the algorithm
    selection_method: SelectionMethods

    test: list[Number] = [1, 2, 3, 4]

    def __init__(
        self,
        variables: list[Variable],
        restrictions: list[Restriction],
        objective: Callable[..., float],
        *,
        maximize: bool = False,
        population_size: int = 10,
        mutation_prob=0.05,
        elite_size: int = 0,
        selection_method: SelectionMethods = SelectionMethods.TOURNAMENT,
        stop_criteria: Callable[[int, Population], bool] = lambda g, _: g > 5,
        mutation_op: Callable[[Chromosome], Chromosome] | None = None,
        crossover_op: Callable[[Chromosome, Chromosome],
                               tuple[Chromosome, Chromosome]]
        | None = None,
        fitness_func: Callable[
            [Chromosome, Callable[..., float], Callable[..., bool]], float
        ]
        | None = None,
        initial_chromosome_func: Callable[[list[Variable]], Chromosome] = lambda x: [
            0 for _ in x
        ],
    ):
        """
        Creates a new instance of a genetic algorithm-based solver.

        Takes as arguments the regular parameters (variables, restrictions,
        objective function) and a few more:

        population_size: Number of individuals in the population
        mutation_prob: Probability of creating a mutated individual
        elite_size: Number of individuals preserved unchanged
        selection_method: How to choose individuals for following populations
        stop_criteria(generations, population) -> bool: When to stop the search
        mutation_op(ind) -> ind: How to mutate a single individual
        crossover_op(mom, dad) -> (ind, ind): How to combine two individuals
        fitness_func(ind, o, v) -> float: How to calculate the fitness. o and v
            are placeholders the objective function and the viability function
        initial_chromosome_func(variables) -> ind: How to create the individuals
            of the initial population
        """

        super().__init__(variables, restrictions, objective, maximize=maximize)

        self.mutation_op = mutation_op
        self.mutation_prob = mutation_prob
        self.crossover_op = crossover_op
        self.stop_criteria = stop_criteria
        self.population_size = population_size
        self.elite_size = elite_size
        self.initial_chromosome_func = initial_chromosome_func
        self.selection_method = selection_method

        # The fitness_func argument gets as parameters the chromosome as well as two
        # functions. These are placeholders for self.objective and self.viable_chromosome
        if fitness_func is None:
            if self.maximize:
                self.fitness = lambda x: objective(x)
            else:
                self.fitness = lambda x: -objective(x)
        else:
            self.fitness = lambda x: fitness_func(
                x, self.objective, self.viable_chromosome
            )

    def solve(self) -> None:
        """
        Optimize the variable values for the objective function given the constraints
        """
        generations: int = 0
        population: Population = self.set_initial_population()
        self.bests = [population[0].fitness]

        # Run generations until the stopping criteria is met:
        while not self.stop_criteria(generations, population):
            population = self.generation(population)
            self.bests.append(population[0].fitness)
            generations += 1

        for variable, value in zip(self.variables, population[0].chromosome):
            variable.set_value(value)

    def set_initial_population(self) -> Population:
        """
        Create an initial population from the initial_chromosome_func
        """
        chromosomes: list[Chromosome] = []
        chromosomes: list[Chromosome] = []

        for _ in range(self.population_size):
            new_chromosome = self.initial_chromosome_func(self.variables)
            chromosomes.append(new_chromosome)

        return [PopulationMember(c, self.fitness(c)) for c in chromosomes]

    def generation(self, population: Population) -> Population:
        """
        Run the steps in each generation and output a new evolved population
        """
        population.sort(reverse=True, key=lambda c: c.fitness)
        augmented_population: Population = deepcopy(population)

        # Save the elite
        new_population = deepcopy(augmented_population[: self.elite_size])

        # Genetic operators
        self.crossover(augmented_population)
        self.mutation(augmented_population)
        self.selection(augmented_population, new_population)

        return new_population

    def crossover(self, augmented_population):
        """
        Perform the crossover operation (if defined) in the augmented population
        and generate new individuals for it
        """
        # If no operator is defined, skip this step
        if self.crossover_op is None:
            return

        # Retrieve two lists of indexes in the augmented population
        shuffle1 = list(range(len(augmented_population)))
        shuffle2 = list(range(len(augmented_population)))

        # Shuffle the lists of indexes
        shuffle(shuffle1)
        shuffle(shuffle2)

        # Iterate through the pairs of parents, crossover them and add the output
        # to the augmented population
        for i1, i2 in zip(shuffle1, shuffle2):
            s1 = augmented_population[i1]
            s2 = augmented_population[i2]
            ns1, ns2 = self.crossover_op(s1.chromosome, s2.chromosome)
            augmented_population.append(
                PopulationMember(ns1, self.fitness(ns1)))
            augmented_population.append(
                PopulationMember(ns2, self.fitness(ns2)))

    def mutation(self, augmented_population):
        """
        Perform the mutation operation (if defined) in the augmented population
        and generate new individuals for it
        """
        # If no operator is defined, skip this step
        if self.mutation_op is None:
            return

        # Create a list for the mutated individuals
        mutated_individuals: list[PopulationMember] = []

        # For each individual in the augmented population, try to perform the
        # mutation operator in them with mutation_prob chance
        for individual in augmented_population:
            if random() < self.mutation_prob:
                mutated: Chromosome = self.mutation_op(individual.chromosome)
                mutated_individuals.append(
                    PopulationMember(mutated, self.fitness(mutated))
                )

        # Add the mutated individuals to the augmented population
        if mutated_individuals:
            augmented_population += mutated_individuals

    def selection(self, augmented_population, new_population):
        """
        Run the correct method to select individuals from the augmented population
        """
        match self.selection_method:
            case SelectionMethods.TOURNAMENT:
                self.selection_by_tournament(
                    augmented_population, new_population)
            case SelectionMethods.RAFFLE:
                self.selection_by_raffle(augmented_population, new_population)
            case _:
                raise Exception("Selection method passed does not exist")

    def selection_by_tournament(self, augmented_population, new_population):
        """
        Select individuals from the augmented_population by tournament
        """
        for _ in range(self.elite_size, self.population_size):
            i1, i2 = sample(augmented_population, k=2)
            if i1.fitness > i2.fitness:
                new_population.append(i1)
            else:
                new_population.append(i2)

    def selection_by_raffle(self, augmented_population, new_population):
        """
        Select individuals from the augmented_population by raffle
        """
        ...
        raffle_wheel: list[float] = list(
            accumulate([p.fitness for p in augmented_population])
        )
        limit: float = raffle_wheel[-1]

        for _ in range(self.population_size):
            selected: float = random() * limit
            selected_index: int = 0
            for index, prob in enumerate(raffle_wheel):
                if prob >= selected:
                    selected_index = index
                    break

            new_population.append(augmented_population[selected_index])

    def viable_chromosome(self, chromosome: Chromosome) -> bool:
        """
        Check if chromosome satisfies all restrictions
        """
        for variable, value in zip(self.variables, chromosome):
            variable.set_value(value)
        return self.satisfied()
