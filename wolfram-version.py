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


#Store each lattice as a string (where each character is a cell with state '1' or '0')
if __name__ == "__main__":
    # CA parameters 
    lattice_length = 80
    ca_steps = 80   #number of time steps we will do 
    ca_radius = 1
    ca_neighborhood = (2 * ca_radius) + 1
    ca_time_space = [] #This will hold all the lattices as we move through time steps
    ca_rule_number = 110

    # Create the rule
    rule_length = 2 ** (2*ca_radius + 1) #rule length = total possible neihborhood configurations (k=2)
    rule_string = bin(ca_rule_number)[2:] #returns string of '1's and '0's reprenting the binary conversion
    missing_length = rule_length - len(rule_string) 
    rule_string = ('0' * missing_length) + rule_string #add correct number of zeros to start of the string
    rule_string = rule_string[::-1]  #reverse the rule string to get the correct order (00..00 case at the start)

    # Create init state 
    start_state = random_lattice(lattice_length)
    print("Start State: " + start_state)

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





