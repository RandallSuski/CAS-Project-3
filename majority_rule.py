import random
import matplotlib.pyplot as plt

# This is the function that will determine the next state of the cell given a rule string
# With the two ending input cells being randomly chosen from the population rather than a neighborhood
def next_cell_wireless(current_state, rule_string, index, length, ca_radius):
    neighborhood_radius = ca_radius - 2
    rule_index = 0
    # The first bit is randomly chosen from the whole sample
    if current_state[random.randint(0, len(current_state) - 1)] == 1:
        rule_index += 1
    rule_index = rule_index << 1

    # The next bits are chosen from the neighborhood
    current_index = index - neighborhood_radius
    for i in range(neighborhood_radius * 2 + 1):
        # Check for valid index
        if current_index >= length:
            current_index -= length
        if current_index < 0:
            current_index += length

        if current_state[current_index] == 1:
            rule_index += 1
        rule_index = rule_index << 1

    # Final bit is also randomly chosen
    if current_state[random.randint(0, len(current_state) - 1)] == 1:
        rule_index += 1
    return rule_string[rule_index]


def create_IC(num_ones, IC_size):
    IC = [0] * IC_size
    if num_ones / IC_size > 0.5:
        # Get a random sample from 0 - lattice_length that represent 0's
        zero_indexes = random.sample(range(IC_size + 1), IC_size - num_ones)
        for i in range(IC_size):
            if i not in zero_indexes:
                IC[i] = 1
    else:
        # Get a random sample from 0 - lattice_length that represents 1's
        one_indexes = random.sample(range(IC_size + 1), num_ones)
        for i in range(IC_size):
            if i in one_indexes:
                IC[i] = 1
    return IC


def test_CA(IC, num_ones, rule, ca_radius):
    # Determine if the IC converges to 1 or 0
    converge_value = 0
    if num_ones / len(IC) > 0.5:
        converge_value = 1
    # Run through 200 time steps
    state = IC
    for _ in range(200):
        state = next_state(state, rule, ca_radius)
    # If after 200 steps the state converged to one value and that value is correct, return 1 for fitness
    if is_state_converged(state) and state[0] == converge_value:
        return 1
    else:
        return 0


def is_state_converged(state):
    convergence_value = state[0]
    for cell in state:
        if cell != convergence_value:
            return False
    return True


def next_state(old_state, rule, ca_radius):
    new_state = [0] * len(old_state)
    for cell_index in range(len(new_state)):
        if next_cell_wireless(old_state, rule, cell_index, len(old_state), ca_radius) == '0':
            new_state[cell_index] = 0
        else:
            new_state[cell_index] = 1
    return new_state


if __name__ == "__main__":
    majority_rule = "00000000000000010000000100010111000000010001011100010111011111110000000100010111000101110111111100010111011111110111111111111111"
    lattice_length = 199
    radius = 3

    # Want to create 100 starting IC's for each rho = 0/lattice_length - lattice_length / lattice_length
    # Then run the rule on each one, calculating the fitness and storing the fitness for a particular rho
    rho_values = []
    fitness_values = []
    for i in range(lattice_length + 1):
        # i represents number of 1's
        rho_values.append(i / lattice_length)
        fitness_score = 0
        # Test 100 IC's with i number of 1's
        for j in range(100):
            # Create an IC with i number of 1's
            my_IC = create_IC(i, lattice_length)
            fitness_score += test_CA(my_IC, i, majority_rule, radius)
        fitness_values.append(fitness_score)
        print("Fitness Score for " + str(i) + " number of ones: " + str(fitness_score))

    # Plot the rho values and fitness scores
    plt.plot(rho_values, fitness_values)
    plt.xlabel('Rho-Values')
    plt.ylabel('Fitness')
    plt.title('Fitness of the Majority Rules Rule for a wireless CA')
    plt.show()