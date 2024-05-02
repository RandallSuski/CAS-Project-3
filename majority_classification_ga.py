'''
Each generation

Fitness
- if final state (not reached or not all the same color) = 0 
- if final state is the majority from the start_state = 1 
- else = 0 
- total fitness = the fraction of times it produced the correct final configuration 

Init: create 100 random rules (the GA's) and 100 random initial configurations 
For each generation
- Find the fitness for each GA 
    - Run the GA on each of the 100 initial configurations 
    - Calculate the fraction of times it makes the correct choice
- Weighted randomly select 2 parents to create 2 children (50 times)
- Use these children in next generation 

textbooks_ga = "00000101000001100001010110000111000001110000010000010101010101110110010001110111000001010000000101111101111111111011011101111111"
'''
import array as array 
import random
from cellular_automata_gui import CellularAutomata2D

# Returns a random lattice with the given length 
def random_lattice(length):
    lattice = ""
    states = ['0', '1']
    for i in range(length):
        lattice += random.choice(states)
    return lattice 

def random_rule(ca_neighborhood):
    # Create the rule
    rule_length = 2 ** (ca_neighborhood) #rule length = total possible neihborhood configurations (k=2)
    ca_rule_number = random.randint(0, (2 ** rule_length)) 

    rule_string = bin(ca_rule_number)[2:] #returns string of '1's and '0's reprenting the binary conversion
    missing_length = rule_length - len(rule_string) 
    rule_string = ('0' * missing_length) + rule_string #add correct number of zeros to start of the string
    rule_string = rule_string[::-1]  #reverse the rule string to get the correct order (00..00 case at the start)
    print(ca_rule_number)
    print(len(rule_string))
    print(rule_string)
    return rule_string


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
        if (current_state[current_index] == '1'):
            rule_index += bit_value 
        bit_value *= 2 
        current_index -= 1 
    return rule_string[rule_index]


# Returns the lattice after 1 time step 
def next_lattice(current_state, rule_string, ca_radius):
    next_state = ""
    length = len(current_state)
    # For each cell, find the next cell state
    for index in range(length):
        cell = next_cell(current_state, rule_string, index, length, ca_radius)
        next_state += cell
    return next_state 

# Returns 1 if final state found the majority in the start 
def test_fitness(start_state, final_state):
    # Find the most used in first
    count1 = 0
    count0 = 0
    for i in range(len(start_state)):
        if (start_state[i] == '1'):
            count1 += 1
        else:
            count0 += 1
    majority_cell = '1'
    if (count0 > count1): 
        majority_cell = '0'

    # Check if we reached 
    for i in range(len(final_state)):
        if (majority_cell != final_state[i]):
            return 0
    return 1

def test_ga(start_state, rule_string, ca_radius, ca_steps):
    # Carry out the CA simulation 
    current_state = start_state 
    for i in range(ca_steps):
        next_state = next_lattice(current_state, rule_string, ca_radius)
        current_state = next_state
    return current_state

def create_children(parents):
    parent1 = parents[0]
    parent2 = parents[1]
    length = len(parent1)
    rand_index = random.randint(0, length - 1)

    child1 = parent1[0:rand_index] + parent2[rand_index:length]
    child2 = parent2[0:rand_index] + parent1[rand_index:length]
    return [child1, child2]

'''
Each generation

Fitness
- if final state (not reached or not all the same color) = 0 
- if final state is the majority from the start_state = 1 
- else = 0 
- total fitness = the fraction of times it produced the correct final configuration 

Init: create 100 random rules (the GA's) and 100 random initial configurations 
For each generation
- Find the fitness for each GA 
    - Run the GA on each of the 100 initial configurations 
    - Calculate the fraction of times it makes the correct choice
- Weighted randomly select 2 parents to create 2 children (50 times)
- Use these children in next generation 

textbooks_ga = "00000101000001100001010110000111000001110000010000010101010101110110010001110111000001010000000101111101111111111011011101111111"
'''

#Store each lattice as a string (where each character is a cell with state '1' or '0')
if __name__ == "__main__":
    # CA parameters 
    lattice_length = 100
    ca_steps = 100   #number of time steps we will do 
    generations = 50
    ca_radius = 3
    ca_neighborhood = (2 * ca_radius) + 1
    ca_time_space = [] #This will hold all the lattices as we move through time steps
    ca_rule_number = 110

    # Create 100 random init configurations and GA's
    init_configs = []
    for i in range(100):
        start_state = random_lattice(lattice_length)
        init_configs.append(start_state)
    ga_rules = []
    for i in range(100):
        ga_rules.append(random_rule(ca_neighborhood))
    ga_max_fitness = []
    print(f"Made init {len(ga_rules)}")
    
    # For each generation 
    for gen_index in range(generations):
        max_fitness = 0
        max_fitness_rule = ""
        all_ga_fitness = []

        # Find the fitness for each GA 
        for ga_index in range(len(ga_rules)):
            #Run the GA on each of the initial configurations 
            fitness = 1.0
            for start_index in range(len(init_configs)):
                start_state = init_configs[start_index]
                final_state = test_ga(start_state, ga_rules[ga_index], ca_radius, ca_steps)
                fitness += test_fitness(start_state, final_state)

            #Calculate the fraction of times it makes the correct choice
            fitness = fitness / (len(init_configs) + 1)
            if (fitness > max_fitness):
                max_fitness = fitness
                max_fitness_rule = ga_rules[ga_index]
            all_ga_fitness.append(fitness)
        ga_max_fitness.append([max_fitness, max_fitness_rule])

        # Weighted randomly select 2 parents to create 2 children (50 times)
        new_ga_rules = []
        for i in range(50):
            parents = random.choices(ga_rules, all_ga_fitness, k=2)
            new_children = create_children(parents)
            new_ga_rules.extend(new_children)
            print(new_children)
        ga_rules = new_ga_rules
        print(f"max fit: {gen_index} {max_fitness}")
        print(f"max rule: {max_fitness_rule}")
        #Use these children in next generation 

'''
    # Create the rule
    rule_string = "01110101100000000001010010100110111010101010011111110111001111010111011000100000111010000010101100010101000111011011001010000110"

    # Create init state 
    start_state = random_lattice(lattice_length)

    # Carry out the CA simulation 
    ca_time_space.append(start_state)
    current_state = start_state 
    for i in range(ca_steps):
        next_state = next_lattice(current_state, rule_string, ca_radius)
        ca_time_space.append(next_state)
        current_state = next_state
    print("Make Graph")
    # Feed the time space into our GUI
    ca = CellularAutomata2D(400, 400)
    ca.create_time_space_diagram(ca_time_space, len(start_state))
    ca.mainloop()
'''

