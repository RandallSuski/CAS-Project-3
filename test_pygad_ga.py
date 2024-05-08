'''
TODO
- Save the solution and fitness data in a file 
    - Could store in global variable inside the onGeneration() function 

Settings used in Paper 
- For the GA runs the chromosomes in the initial population were uniformly distributed over λ ∈ [0.0, 1.0]
- top 20% of the population—the set of elite strings—is copied without modification into
the next generation
- Should be graphing the best fitness for that generation 

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
import csv

# Global variables so that we can get values into and out of the GA instance
global start_states # [Majority Cell, Lattice]
global zero_fitness
ca_steps = 100   #number of time steps we will do 
ca_radius = 3
generation_info = [] 

''' Generating the initial states '''
# Generates a normal curve of values between 0 and 1 
def generate_normal_1_percents(n):
    # Generate normal curve
    mean = 0.5
    sd = 0.15
    normalized_values = np.random.normal(mean, sd, n)
    normalized_values = np.clip(normalized_values, 0, 1)
    
    # Takeout some values 
    count0 = 0
    count1 = 0
    half = n/2 
    values = []
    for val in normalized_values:
        if (val > 0.5 and count1 < half):
            values.append(val)
            count1 += 1
        if (val < 0.5 and count0 < half):
            values.append(val)
            count0 += 1
        
    # Get 50/50
    while (count1 < half or count0 < half):
        new_val = np.random.normal(mean, sd, 1)[0]
        if (count1 < half and new_val > 0.5):
            values.append(new_val)
            count1 += 1
        if (count0 < half and new_val < 0.5):
            values.append(new_val)
            count0 += 1
    return values

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

def generate_uniform_rules(n, ca_neighborhood):
    # Create uniform percents 
    uniform_rules= []
    for i in range(n):
        current_percent = i / (n - 1) 
        uniform_rules.append(weighted_random_rule(ca_neighborhood, current_percent))
    return uniform_rules


# Creates a random rule with a given percent of 1's 
def weighted_random_rule(ca_neighborhood, percent):    
    # Create the rule
    rule_length = 2 ** (ca_neighborhood) 
    rule = []
    states = [0, 1]

    count1 = round(rule_length * percent)
    count0 = rule_length - count1

    for i in range(rule_length):
        weights = [count0, count1]
        rule_bit = rand.choices(states, weights=weights, k=1)[0]
        rule.append(rule_bit)
        if (rule_bit == 1):
            count1 -= 1
        else:
            count0 -= 1
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

# Define your custom mutation function
def mutation_function(parent1, parent2):
    # Randomly select a single point (index)
    mutation_point = random.randint(0, len(parent1) - 1)
    child1 = np.concatenate((parent1[:mutation_point], parent2[mutation_point:]))
    child2 = np.concatenate((parent2[:mutation_point], parent1[mutation_point:]))
    return child1, child2
    
# The fitness function for the GA 
def my_fitness_func(ga_instance, solution, solution_idx):
    #Run the GA on each of the initial configurations 
    fitness = 1.0

    for start_state in start_states:
        start_lattice = start_state[1]
        final_state = simulate_ga(start_lattice, solution, ca_radius, ca_steps)

        result = get_state_fitness(start_lattice, final_state)
        fitness += result

    #Calculate the fraction of times it makes the correct choice
    fitness = fitness / (len(start_states) + 1)
    return fitness

def get_state_fitness(start_state, final_state):
    # Find the most used in first
    count1 = 0
    count0 = 0
    for i in range(len(start_state)):
        if (start_state[i] == 1):
            count1 += 1
        else:
            count0 += 1
    majority_cell = 1
    if (count0 > count1): 
        majority_cell = 0

    # Check if we reached 
    for i in range(len(final_state)):
        if (majority_cell != final_state[i]):
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

# This is the function that will determine the next state of the cell given a rule string
# With the two ending input cells being randomly chosen from the population rather than a neighborhood
def next_cell_wireless(current_state, rule_string, index, length, ca_radius):
    neighborhood_radius = ca_radius - 2
    rule_index = 0
    # The first bit is randomly chosen from the whole sample

    if (current_state[rand.randint(0, len(current_state) - 1)] == 1):
        rule_index += 1
    rule_index = rule_index << 1

    # The next bits are chosen from the neighborhood
    current_index = index - neighborhood_radius
    for i in range(neighborhood_radius * 2 + 1):
        # Check for valid index
        if (current_index >= length):
            current_index -= length
        if (current_index < 0):
            current_index += length

        if (current_state[current_index] == 1):
            rule_index += 1
        rule_index = rule_index << 1

    # Final bit is also randomly chosen
    if (current_state[rand.randint(0, len(current_state) - 1)] == 1):
        rule_index += 1
    return rule_string[rule_index]


last_fitness = 0
def on_generation(ga_instance):
    # Print the generation results
    global last_fitness
    global start_states 
    global zero_fitness
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"    Sol    = {solution}")
    print(f"    Fitness     = {solution_fitness}")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

    # Store this generation data in the array 
    gen_entry = [ga_instance.generations_completed, solution_fitness, solution]  
    generation_info.append(gen_entry)

    
    count0 = 0
    count1 = 0
    for state in start_states:
        vcount0 = 0
        vcount1 = 0
        for val in state[1]:
            if (val == 0):
                vcount0 += 1
            else: 
                vcount1 += 1
        if (vcount0 > vcount1):
            count0 += 1
        if (vcount1 > vcount0): 
            count1 += 1
    print(f"  0: {count0}  1: {count1}")

    # Create I new IC for the next generation 
    normal_percents = generate_normal_1_percents(len(start_states))
    #print(f" normal percents: {normal_percents}")
    start_states = []
    for percent in normal_percents:
        lattice = random_lattice(lattice_length, percent)
        majority_cell = 1
        if (percent < 0.5):
            majority_cell = 0
        start_states.append([majority_cell, lattice])
    
    print(len(start_states))
    count0 = 0
    count1 = 0
    for state in start_states:
        vcount0 = 0
        vcount1 = 0
        for val in state[1]:
            if (val == 0):
                vcount0 += 1
            else: 
                vcount1 += 1
        if (vcount0 > vcount1):
            count0 += 1
        if (vcount1 > vcount0): 
            count1 += 1
    print(f"  0: {count0}  1: {count1}")
    


''' Storing and graphing data '''

def write_current_generation_data():
    with open("generation-data-mut-rate-0.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(generation_info)






''' Main '''
if __name__ == "__main__":
    # CA parameters 
    zero_fitness = False
    lattice_length = 100
    ca_start_state_count = 50
    ca_rule_count = 50
    ca_steps = 100   #number of time steps we will do 
    ca_radius = 3
    ca_neighborhood = (2 * ca_radius) + 1


    # Create the initial population and states
    initial_population = generate_uniform_rules(ca_rule_count, ca_neighborhood)
    initial_population[1] = [0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,1,0,0,1,0,0,0,1,1,1,0,1,1,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1]

    normal_percents = generate_normal_1_percents(ca_start_state_count)
    start_states = []
    for percent in normal_percents:
        lattice = random_lattice(lattice_length, percent)
        majority_cell = 1
        if (percent < 0.5):
            majority_cell = 0
        start_states.append([majority_cell, lattice])


    # Pygad parameters 
    num_generations = 100
    num_parents_mating = ca_rule_count  # How many parents will create children 
    parallel_processing= None # None or ["process", 10]  # can set equal to an Int to run GA with that many threads 
    random_seed = 12345  # For reproducability 
    save_best_solutions = False
    save_solutions = False
    parent_selection_type = "rank"

    gene_space = [0, 1]  
    mutation_type = "random"
    mutation_num_genes = 1
    allow_duplicate_genes=True
    keep_elitism= round(ca_rule_count * 0.2)  # 20% are kept 
    crossover_type = "single_point"
    '''
    mutation_type, mutation_percent_genes, crossover_type, num_parents_mating
    "random", 30, single_point, 20 -> 0.29, 0.25, 0.47   <---------
    "swap", 30, single_point, 20 -> 0.06, 0.49, 0.06
    None, 30, single_point, 20 -> 0.06, 0.06

    "random", 30, single_point, 10 -> 0.49   <---------
    "random", 30, single_point, 40 -> 0.27

    '''


    # Running the GA 
    generation_info = []    
    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=my_fitness_func,
                       initial_population=initial_population,
                       keep_elitism=keep_elitism,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_num_genes=mutation_num_genes,
                       random_seed=random_seed,
                       gene_space=gene_space,
                       allow_duplicate_genes=allow_duplicate_genes,
                       on_generation=on_generation, 
                       save_best_solutions=save_best_solutions,
                       save_solutions=save_solutions,
                       parent_selection_type=parent_selection_type)
    ga_instance.run()
    write_current_generation_data()
    
    # Getting the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")


    # Plotting fitness across generations 
    ga_instance.plot_fitness()  


