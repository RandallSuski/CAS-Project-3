import array as array 
import random
from cellular_automata_gui import CellularAutomata2D
import re

# Returns a random lattice with the given length 
def percent_random_lattice(length, percent):
    lattice = ""
    states = ['0', '1']

    count1 = round(length * percent)
    count0 = length - count1

    for i in range(length):
        weights = [count0, count1]
        newCell = random.choices(states, weights=weights, k=1)[0]
        lattice += newCell
        if (newCell == '1'):
            count1 -= 1
        else:
            count0 -= 1
    return lattice 

# Returns a random lattice with the given length 
def random_lattice(length):
    lattice = ""
    states = ['0', '1']
    for i in range(length):
        lattice += random.choice(states)
    return lattice 

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

def fitness(state):
    first_cell = state[0]
    for i in range(len(state)):
        if (first_cell != state[i]):
            return 0
    return 1

def extract_numbers(input_string):
    rule = ""
    numbers = re.findall(r'\d+', input_string)
    numbers = [int(num) for num in numbers]
    return re.sub(r'\D', '', input_string)



#Store each lattice as a string (where each character is a cell with state '1' or '0')
if __name__ == "__main__":

    # CA parameters 
    lattice_length = 200
    ca_steps = 200   #number of time steps we will do 
    ca_radius = 3
    ca_neighborhood = (2 * ca_radius) + 1
    ca_time_space = [] #This will hold all the lattices as we move through time steps
    ca_rule_number = 110

    # Create the rule
    input_string = "[1. 1. 1. 0. 0. 0. 1. 1. 1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 0. 1. 1. 1. 0. 0. 0. 1. 1. 1. 1. 0. 0. 0. 0. 1. 1. 1. 1. 0. 0. 1. 0. 1. 1. 0. 0. 1. 0. 1. 1. 1. 1. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 0. 1. 1. 0. 0. 0. 1. 1. 0. 0. 1. 1. 1. 0. 0. 0. 1. 1. 0. 0. 1. 0. 0. 1. 1. 1. 1. 0. 1. 0. 1. 1. 0. 1. 1. 1. 0. 1. 1. 0. 0. 1. 1. 0. 0. 1. 1. 0. 1. 0. 1. 1. 0. 1. 1. 1. 1. 1. 1.]"
    rule_string = extract_numbers(input_string)
    print(rule_string)
    rule_string = "00000101000001100001010110000111000001110000010000010101010101110110010001110111000001010000000101111101111111111011011101111111"

    # Create init states over λ ∈ [0.0, 1.0]
    start_states = []
    percents = [0.45] #[0.2, 0.4, 0.6, 0.8]
    for percent in percents:
        start_states.append(percent_random_lattice(lattice_length, percent))


    # Graph each start state against the rule 
    for start_state in start_states: 
        # Carry out the CA simulation 
        print(start_state)  
        ca_time_space = []
        ca_time_space.append(start_state)
        current_state = start_state 
        for i in range(ca_steps):
            next_state = next_lattice(current_state, rule_string, ca_radius)
            ca_time_space.append(next_state)
            current_state = next_state
        
        print(f"final: {next_state}")
        # Feed the time space into our GUI
        ca = CellularAutomata2D(400, 400)
        ca.create_time_space_diagram(ca_time_space, len(start_state))
        ca.mainloop()





