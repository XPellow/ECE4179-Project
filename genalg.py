from geneal.genetic_algorithms import ContinuousGenAlgSolver
from geneal.applications.fitness_functions.binary import fitness_functions_binary
from neuralNet import full_train
from random import random
from math import floor
import numpy as np

class CNNGenAlgSolver:
    def __init__(self, **kwargs):
        # Super funcs
        '''ContinuousGenAlgSolver.__init__(self,
            pop_size=kwargs["pop_size"],
            max_gen=kwargs["max_gen"],
            mutation_rate=kwargs["mutation_rate"],
            selection_rate=kwargs["selection_rate"],
            selection_strategy=kwargs["selection_strategy"],
            n_genes=1 # FIX 
        )'''

        # Get attributes
        self.Model = kwargs["model"]
        self.pop_size = kwargs["pop_size"]
        self.pool_size = kwargs["pool_size"]
        self.num_channels = kwargs["num_channels"]
        self.train_loaders = kwargs["train_loaders"]
        self.test_loaders = kwargs["test_loaders"]
        self.device = kwargs["device"]
        self.loss_func = kwargs["loss_function"]
        self.optimizerFunc = kwargs["optimizer"]
        self.lr = kwargs["learning_rate"]
        self.max_gen = kwargs["max_gen"]
        self.n_epochs = kwargs["n_epochs"]
        self.nkernels = kwargs["nkernels"]
        self.nclasses = kwargs["nclasses"]

        # Get extrapolated attributes
        example_model = self.Model(self.nkernels, self.nclasses)
        self.kernel_width = example_model.kernel_size
        self.kernel_height = example_model.kernel_size
        self.n_genes = example_model.nkernels # num of weights of first layer
        self.nloaders = len(self.train_loaders)

        # Setup loggers
        #   Each logger is an array where each element is a generations loss/acc logs
        #   e.g train_losses = 
        #   [[gen1-model1.train_losses, ... , gen1-modeln], ..., 
        #    [genm-model1, ..., genm-modeln]]
        self.train_losses = [[] for i in range(self.max_gen+1)]
        self.test_losses = [[] for i in range(self.max_gen+1)]
        self.train_accs = [[] for i in range(self.max_gen+1)]
        self.test_accs = [[] for i in range(self.max_gen+1)]


    def fitness_function(self, chromosome):
        """
        Implements the logic that calculates the fitness
        measure of an individual.
        :param chromosome: chromosome of genes representing an individual
        :return: the fitness of the individual
        """
        # Pick a random loader
        idx = np.random.randint(self.nloaders)
        test_loader = self.test_loaders[idx]
        train_loader = self.train_loaders[idx]

        optimizer = self.optimizerFunc(chromosome.parameters(), self.lr)
        train_loss, test_loss, train_acc, test_acc = full_train(
            model=chromosome, 
            n_epochs=self.n_epochs, 
            train_loader=train_loader, 
            test_loader=test_loader, 
            loss_func=self.loss_func, 
            optimizer=optimizer, 
            device=self.device, 
            freeze=True
        )

        # Logging data
        self.train_losses[self.gen].append(train_loss)
        self.test_losses[self.gen].append(test_loss)
        self.train_accs[self.gen].append(train_acc)
        self.test_accs[self.gen].append(test_acc)

        return test_acc[-1] # Returns the latest accuracy of the current model


    def calculate_fitness(self, population):
        """
        Calculates the fitness of the population
        :param population: population state at a given iteration
        :return: the fitness of the current population
        """
        return np.array(list(map(self.fitness_function, population)))


    def initialize_population(self):
        """
        Initializes the population of the problem
        :param pop_size: number of individuals in the population
        :param n_genes: number of genes representing the problem. In case of the binary
        solver, it represents the number of genes times the number of bits per gene
        :return: a numpy array with a randomized initialized population
        """

        population = []
        for i in range(self.pop_size):
            new_model = self.Model(self.nkernels, self.nclasses).to(self.device)
            population.append(new_model)

        return np.array(population)


    def create_offspring(self, genes):
        """
        Creates a new population given a set of genes
        """

        population = []
        for i in range(self.pop_size):
            new_model = self.Model(self.nkernels, self.nclasses).to(self.device)
            new_genome = np.random.choice(genes, size=self.nkernels, replace=False)
            new_model.init_genome(new_genome)
            population.append(new_model)
        
        return population


    def mutate_population(self, population):
        """
        just in case we decide to make mutations
        """
        return population


    def liquidate(self, population):
        '''
        Given some population (usually fitness tested), returns the set of all of its 
        genes.
        '''
        genes = []
        for i in population:
            for j in i.conv1.weight:
                genes.append(j)

        return genes

    def solve(self):
        population = self.initialize_population()

        self.gen = 0
        while True:
            # Keep track of current generation
            print("Generation: {}/{}".format(self.gen, self.max_gen))

            # Find the normalzied fitness of each model
            fitness = self.calculate_fitness(population)
            fitness /= sum(fitness) # Normalizes array

            if self.gen >= self.max_gen: break # Just get fitness of models & break at end

            # Get the genes of the best models & build a new population
            fittest_models = np.random.choice(population, size=self.pool_size, replace=False, p=fitness)
            genes = self.liquidate(fittest_models)
            population = self.create_offspring(genes)
            population = self.mutate_population(population)

            self.gen += 1