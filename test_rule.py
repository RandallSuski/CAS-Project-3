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
    lattice_length = 200
    ca_steps = 200   #number of time steps we will do 
    ca_radius = 3
    ca_neighborhood = (2 * ca_radius) + 1
    ca_time_space = [] #This will hold all the lattices as we move through time steps
    ca_rule_number = 110

    # Create the rule
    rule_string = "10110010111010001111101111011111000010010011111001001011010111010101110111001101101001000001001100111001010011001101100001111111"

    # Create init state 
    start_state = random_lattice(lattice_length)

    # Carry out the CA simulation 
    ca_time_space.append(start_state)
    current_state = start_state 
    for i in range(ca_steps):
        next_state = next_lattice(current_state, rule_string, ca_radius)
        ca_time_space.append(next_state)
        current_state = next_state
        
    # Feed the time space into our GUI
    ca = CellularAutomata2D(400, 400)
    ca.create_time_space_diagram(ca_time_space, len(start_state))
    ca.mainloop()





