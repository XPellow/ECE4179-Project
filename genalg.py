from geneal.genetic_algorithms import BinaryGenAlgSolver
from geneal.applications.fitness_functions.binary import fitness_functions_binary
from neuralNet import evaluate_model, Model
from random import random
from math import floor
import numpy as np

class CNNGenAlgSolver(ContinuousGenAlgSolver, BinaryGenAlgSolver):
    def __init__(self, *args, **kwargs):
        # super funcs
        BinaryGenAlgSolver.__init__(self, *args, **kwargs)
        ContinuousGenAlgSolver.__init__(self, *args, **kwargs)

        # get attributes
        self.Model = kwargs["model"]
        self.num_channels = kwargs["num_channels"]
        self.train_set = kwargs["train_set"]
        self.test_set = kwargs["test_set"]
        self.device = kwargs["device"]
        self.loss_function = kwargs["loss_function"]
        self.optimizerFunc = kwargs["optimizer"]
        self.lr = kwargs["learning_rate"]

        # get extrapolated attributes
        example_model = self.Model()
        self.kernel_width = example_model.kernel_size
        self.kernel_height = example_model.kernel_size
        self.n_genes = example_model.nkernels # num of weights of first layer

    def fitness_function(self, chromosome): ## TODO
        """
        Implements the logic that calculates the fitness
        measure of an individual.
        :param chromosome: chromosome of genes representing an individual
        :return: the fitness of the individual
        """
        full_train(chromosome, train_loader, test_loader, loss_func, optimizer, device)
        return evaluate_model(chromosome, device, test_set)

    def initialize_population(self):
        """
        Initializes the population of the problem
        :param pop_size: number of individuals in the population
        :param n_genes: number of genes representing the problem. In case of the binary
        solver, it represents the number of genes times the number of bits per gene
        :return: a numpy array with a randomized initialized population
        """

        population = []
        self.optimizers = []
        for i in range(pop_size):
            new_model = self.Model()
            population.append(new_model)
            self.optimizers.append(self.optimizer(new_model.parameters(), self.lr))

        return np.array(population)


    def create_offspring(self, first_parent, sec_parent, crossover_pt, offspring_number):
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
        first_genome = first_parent.get_genome()
        second_genome = sec_parent.get_genome()
        model = Model(first_parent.nkernels, first_parent.nclasses)
        return model.init_genome(first_genome[:crossover_pt], second_genome[crossover_pt:])

    def mutate_population(self, population, n_mutations): ## This is retarded lmao ##
        """
        Mutates the population according to a given user defined rule.
        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed. This number is 
        calculated according to mutation_rate, but can be adjusted as needed inside this function
        :return: the mutated population
        """
        for i in range(n_mutations): # Get a random weight in the kernel and increase it
            popind = floor(random()*len(population))
            channelind = floor(random()*self.num_channels)
            k_widthind = floor(random()*self.kernel_width)
            k_heightind = floor(random()*self.kernel_height)
            mutated = population[popind]
            mutated_genome = mutated.get_genome()
            mutated_genome[channelind][k_widthind][k_heightind] += random()-0.5
            mutated.init_genome(mutated_genome) # double check ordering of width & height & that this works
        return population
