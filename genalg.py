from geneal.genetic_algorithms import ContinuousGenAlgSolver
from geneal.applications.fitness_functions.binary import fitness_functions_binary
from neuralNet import full_train
from random import random
from math import floor
import numpy as np

class CNNGenAlgSolver(ContinuousGenAlgSolver):
    def __init__(self, *args, **kwargs):
        # Super funcs
        ContinuousGenAlgSolver.__init__(self,
            pop_size=kwargs["pop_size"],
            max_gen=kwargs["max_gen"],
            mutation_rate=kwargs["mutation_rate"],
            selection_rate=kwargs["selection_rate"],
            selection_strategy=kwargs["selection_strategy"],
            n_genes=1 # FIX 
        )

        # Get attributes
        #self.pop_size = kwargs["pop_size"]
        self.Model = kwargs["model"]
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
        self.train_losses = [[] for i in range(self.max_gen)]
        self.test_losses = [[] for i in range(self.max_gen)]
        self.train_accs = [[] for i in range(self.max_gen)]
        self.test_accs = [[] for i in range(self.max_gen)]


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
        gen = chromosome.generation
        self.train_losses[gen].append(train_loss)
        self.test_losses[gen].append(test_loss)
        self.train_accs[gen].append(train_acc)
        self.test_accs[gen].append(test_acc)

        return test_acc[-1] # Returns the latest accuracy of the current model


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
            new_model.generation = 0
            population.append(new_model)

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

        model = Model(self.nkernels, self.nclasses).to(self.device)
        model.generation = max(first_parent.generation, sec_parent.generation) + 1
        model.init_genome(first_genome[:crossover_pt], second_genome[crossover_pt:])
        
        return model


    def mutate_population(self, population, n_mutations): ## This is retarded lmao ##
        """
        Mutates the population according to a given user defined rule.
        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed. This number is 
        calculated according to mutation_rate, but can be adjusted as needed inside this function
        :return: the mutated population
        """
        '''
        for i in range(n_mutations): # Get a random weight in the kernel and increase it
                                    popind = floor(random()*len(population))
                                    channelind = floor(random()*self.num_channels)
                                    k_widthind = floor(random()*self.kernel_width)
                                    k_heightind = floor(random()*self.kernel_height)
                                    mutated = population[popind]
                                    mutated_genome = mutated.get_genome()
                                    mutated_genome[channelind][k_widthind][k_heightind] += random()-0.5
                                    mutated.init_genome(mutated_genome) # double check ordering of width & height & that this works
        '''
        return population
