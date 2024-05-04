'''
TODO
- Save the solution and fitness data in a file 
    - Could store in global variable inside the onGeneration() function 

Optimize 
- Store start state with its majority value so that we don't calculate it each simulation
- Finding the next lattice 
    - We read the next cell and double cell value multiple times 
    - Making it return early if nextState = currState (should be able to do this with no increased runtime)
- Set the parallel processing to be some value (use processes over threads)

'''
import pygad
import numpy as np 
import random as rand 
import matplotlib.pyplot as plt

start_states = []  # [Majority Cell, Lattice]
ca_steps = 100   #number of time steps we will do 
ca_radius = 3

''' Generating the initial states '''
# Generates a normal curve of values between 0 and 1 
def generate_normal_1_percents(n):
    # Generate normal curve
    rand_values = np.random.normal(size=n)

    # Normalize the values to between [0,1]
    min = np.min(rand_values)
    max = np.max(rand_values)
    normalized_values = (rand_values - min) / (max - min)
    return normalized_values

# Creates an initial state using the percent value (from 0 to 1) as the chance to choose 1 
def random_lattice(length, percent):
    lattice = []
    states = [0, 1]

    count1 = round(length * percent)
    count0 = length - count1

    for i in range(length):
        weights = [count0, count1]
        newCell = rand.choices(states, weights=weights, k=1)[0]
        lattice.append(newCell)
        if (newCell == 1):
            count1 -= 1
        else:
            count0 -= 1
    return lattice 

# Creates a random rule 
def random_rule(ca_neighborhood):
    # Create the rule
    rule_length = 2 ** (ca_neighborhood) 
    rule = []
    states = [0, 1]
    for i in range(rule_length):
        rule.append(np.random.choice(states))
    return rule

# Takes a string rule and returns it as an int list 
def string_rule_to_int_list(input_rule):
    new_rule = []
    for char in input_rule:
        if (char == '1'):
            new_rule.append(1)
        else: 
            new_rule.append(0)
    return new_rule


''' GA related functions '''

# The fitness function for the GA 
def fitness_func(ga_instance, solution, solution_idx):
    #Run the GA on each of the initial configurations 
    fitness = 1.0
    for start_state in start_states:
        majority_cell = start_state[0]
        start_lattice = start_state[1]
        final_state = simulate_ga(start_lattice, solution, ca_radius, ca_steps)

        result = get_state_fitness(majority_cell, final_state)
        if (result == 1):
            print(f" win: {start_state}")
        fitness += result

    #Calculate the fraction of times it makes the correct choice
    fitness = fitness / (len(start_states) + 1)
    print(f"   {solution_idx} : {fitness}")
    return fitness

def get_state_fitness(majority_cell, final_state):
    # Check if we reached 
    for final_cell in final_state:
        if (final_cell != majority_cell):
            return 0
    return 1

# Tests the GA on a given state 
def simulate_ga(start_state, rule_string, ca_radius, ca_steps):
    # Carry out the CA simulation 
    current_state = start_state 
    for i in range(ca_steps):
        next_state = next_lattice(current_state, rule_string, ca_radius)
        current_state = next_state
    return current_state
    
# Returns the lattice after 1 time step 
def next_lattice(current_state, rule_string, ca_radius):
    next_state = []
    length = len(current_state)
    # For each cell, find the next cell state
    for index in range(length):
        cell = next_cell(current_state, rule_string, index, length, ca_radius)
        next_state.append(cell)
    return next_state 

# Returns the next character that will be at the given index.
def next_cell(current_state, rule_string, index, length, ca_radius):
    rule_index = 0
    bit_value = 1  #the value the bit has towards rule
    current_index = index + ca_radius 
    # Check each cell in the neighborhood of the index cell 
    for i in range(ca_radius*2 + 1):
        # Check for valid index
        if (current_index >= length):
            current_index -= length 
        if (current_index < 0):
            current_index += length
        # Get value from that cell
        if (current_state[current_index] == 1):
            rule_index += bit_value 
        bit_value *= 2 
        current_index -= 1 
    return rule_string[rule_index]

last_fitness = 0
def print_on_gen(ga_instance):
    global last_fitness
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"    Sol    = {solution}")
    print(f"    Fitness     = {solution_fitness}")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]


''' Main '''

if __name__ == "__main__":
    # CA parameters 
    lattice_length = 100
    ca_start_state_count = 50
    ca_rule_count = 50
    ca_steps = 100   #number of time steps we will do 
    ca_radius = 3
    ca_neighborhood = (2 * ca_radius) + 1


    # Create the initial population and states
    initial_population = []
    for i in range(ca_rule_count):
        initial_population.append(random_rule(ca_neighborhood))
    normal_percents = generate_normal_1_percents(ca_start_state_count)
    for percent in normal_percents:
        lattice = random_lattice(lattice_length, percent)
        majority_cell = '1'
        if (percent < 0.5):
            majority_cell = '0'
        start_states.append([majority_cell, lattice])


    # Pygad parameters 
    num_generations = 10
    num_parents_mating = 10  # How many parents will create children 
    fitness_function = fitness_func  
    parallel_processing= None # None or ["process", 10]  # can set equal to an Int to run GA with that many threads 
    random_seed = 12345  # For reproducability 
    save_best_solutions = True

    gene_space = [0, 1]  
    mutation_type = "swap"
    mutation_percent_genes = 10
    allow_duplicate_genes=True
    keep_parents = 0  # Number of parents kept in population along side children 
    crossover_type = "single_point"


    # Running the GA 
    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       initial_population=initial_population,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       random_seed=random_seed,
                       gene_space=gene_space,
                       allow_duplicate_genes=allow_duplicate_genes,
                       on_generation=print_on_gen, 
                       save_best_solutions=save_best_solutions)
    ga_instance.run()
    
    # Getting the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")

    # Plotting fitness across generations 
    ga_instance.plot_fitness()  


