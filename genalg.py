from geneal.genetic_algorithms import BinaryGenAlgSolver
from geneal.applications.fitness_functions.binary import fitness_functions_binary
from neuralNet import evaluate_model, Model
from random import random
from math import floor

class GenAlgSolver(ContinuousGenAlgSolver, BinaryGenAlgSolver):
    def __init__(self, *args, **kwargs):
        BinaryGenAlgSolver.__init__(self, *args, **kwargs)
        ContinuousGenAlgSolver.__init__(self, *args, **kwargs)

    def fitness_function(self, chromosome):
        """
        Implements the logic that calculates the fitness
        measure of an individual.

        :param chromosome: chromosome of genes representing an individual
        :return: the fitness of the individual
        """
        return evaluate_model(chromosome)

    def initialize_population(self): ## TODO ##
        """
        Initializes the population of the problem

        :param pop_size: number of individuals in the population
        :param n_genes: number of genes representing the problem. In case of the binary
        solver, it represents the number of genes times the number of bits per gene
        :return: a numpy array with a randomized initialized population
        """
        pass

    def create_offspring(self, first_parent, sec_parent, crossover_pt, offspring_number): ## TODO ##
        """
        Creates an offspring from 2 parents. It uses the crossover point(s)
        to determine how to perform the crossover

        :param first_parent: first parent's chromosome
        :param sec_parent: second parent's chromosome
        :param crossover_pt: point(s) at which to perform the crossover
        :param offspring_number: whether it's the first or second offspring from a pair of parents.
        Important if there's different logic to be applied to each case.
        :return: the resulting offspring.
        """
        first_kernels = first_parent.conv1.parameters()
        second_kernels = sec_parent.conv1.parameters()
        return Model(first_kernels[:crossover_pt], second_kernels[crossover_pt:])

    def mutate_population(self, population, n_mutations): ## TODO ##
        """
        Mutates the population according to a given user defined rule.

        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed. This number is 
        calculated according to mutation_rate, but can be adjusted as needed inside this function
        :return: the mutated population
        """
        for i in range(n_mutations):
            rand = random()
            popind = floor(rand*len(population))
            #layerind = floor(rand*3) # RGB in = 3 channels
            #RGBind = 
            #l1ind = 
            #l2ind = 
            #population[popind].conv1.parameters()[RGBind][l1ind][l2ind] += 

solver = BinaryGenAlgSolver(
    n_genes=64, # number of kernels in first layer
    #fitness_function=fitness_functions_binary(1), 
    n_bits=75, # number of bits describing each gene (variable) [number of weights in first layers kernels]
    pop_size=10, # population size (number of individuals)
    max_gen=500, # maximum number of generations
    mutation_rate=0.05, # mutation rate to apply to the population
    selection_rate=0.5, # percentage of the population to select for mating
    selection_strategy="roulette_wheel", # strategy to use for selection. see below for more details
)