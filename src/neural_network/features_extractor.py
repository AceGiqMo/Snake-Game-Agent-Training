import numpy as np
import copy


def _get_relative_danger_loc_neurons(row_size, column_size, map, snake, current_dir):
    """
    Fills the input neurons by 0 or 1, indicating the presence of danger ahead, on the left and on the right
    """
    input_arr = np.full(shape=3, fill_value=0)

    # Fill the occupied by snake cells
    snake_arr = np.array(snake)
    occupied = copy.deepcopy(map)
    occupied[snake_arr[:, 0], snake_arr[:, 1]] = True

    left_rot_matrix = np.array([
        [0, 1],
        [-1, 0]
    ])

    right_rot_matrix = np.array([
        [0, -1],
        [1, 0]
    ])

    left_dir = left_rot_matrix @ current_dir
    right_dir = right_rot_matrix @ current_dir

    head = snake[0]

    # Danger ahead
    if (not(0 <= head[1] + current_dir[1] < row_size) or
        not(0 <= head[0] + current_dir[0] < column_size) or
            occupied[head[1] + current_dir[1], head[0] + current_dir[0]]):

        input_arr[0] = 1

    # Danger on the left
    if (not(0 <= head[1] + left_dir[1] < row_size) or
        not(0 <= head[0] + left_dir[0] < column_size) or
            occupied[head[1] + left_dir[1], head[0] + left_dir[0]]):
        input_arr[1] = 1

    # Danger on the right
    if (not(0 <= head[1] + right_dir[1] < row_size) or
        not(0 <= head[0] + right_dir[0] < column_size) or
            occupied[head[1] + right_dir[1], head[0] + right_dir[0]]):
        input_arr[2] = 1

    return input_arr


def _get_direction_neurons(current_dir):
    """
    Uses One-Hot Encoding to fill the input neurons, corresponding to UP, DOWN, LEFT, RIGHT
    """
    input_arr = np.full(shape=4, fill_value=0)

    if current_dir == [0, -1]:
        input_arr[0] = 1

    elif current_dir == [0, 1]:
        input_arr[1] = 1

    elif current_dir == [-1, 0]:
        input_arr[2] = 1

    else:
        input_arr[3] = 1

    return input_arr


def _get_relative_food_direction_neurons(snake, food_pos):
    """
    Determines the direction (UP, DOWN, LEFT, RIGHT) towards the food/super-food relative to the snake location
    """
    input_arr = np.full(shape=4, fill_value=0)

    # Relevant to superfood: if not active
    if not food_pos:
        return input_arr

    head = snake[0]
    direction = (food_pos[0] - head[0], food_pos[1] - head[1])

    if direction[1] < 0:
        input_arr[0] = 1

    elif direction[1] > 0:
        input_arr[1] = 1

    if direction[0] < 0:
        input_arr[2] = 1

    elif  direction[0] > 0:
        input_arr[3] = 1

    return input_arr

def _get_distance_to_food_neuron(row_size, column_size, snake, food_pos, superfood_pos):
    """
    Returns the normalized Manhattan distance to the food (distance / max_possible)
    """
    head = snake[0]

    distance_to_food = np.abs(food_pos[0] - head[0]) + np.abs(food_pos[1] - head[1])

    if superfood_pos:
        distance_to_superfood = np.abs(superfood_pos[0] - head[0]) + np.abs(superfood_pos[1] - head[1])
    else:
        distance_to_superfood = row_size + column_size

    max_possible_distance = row_size + column_size

    if distance_to_food < distance_to_superfood:
        return np.array([distance_to_food / max_possible_distance])

    else:
        return np.array([distance_to_superfood / max_possible_distance])



def _get_game_ticks_neuron(current_tick, max_expected_ticks):
    """
    Returns normalized number of elapsed ticks (frames), which encourages speed and prevents infinite loops
    """
    input_arr = np.array([current_tick / max_expected_ticks])

    return input_arr



def assemble_input_neurons_array(row_size, column_size, map, snake, current_dir, food_pos, superfood_pos,
                                 current_tick, max_expected_ticks):

    input_frag1 = _get_relative_danger_loc_neurons(row_size, column_size, map, snake, current_dir)
    input_frag2 = _get_direction_neurons(current_dir)
    input_frag3 = _get_relative_food_direction_neurons(snake, food_pos)         # For general food
    input_frag4 = _get_relative_food_direction_neurons(snake, superfood_pos)    # For superfood
    input_frag5 = _get_distance_to_food_neuron(row_size, column_size, snake, food_pos, superfood_pos) # The closest food
    input_frag6 = _get_game_ticks_neuron(current_tick, max_expected_ticks)

    return np.hstack([input_frag1, input_frag2, input_frag3, input_frag4, input_frag5, input_frag6])



