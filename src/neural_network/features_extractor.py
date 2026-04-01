import numpy as np
import copy


def _get_relative_danger_loc_neurons(row_size, column_size, map, snake, current_dir):
    """
    Fills the input neurons by 0 or 1, indicating the presence of danger ahead, on the left and on the right
    """
    input_arr = np.full(shape=3, fill_value=0)

    # Fill the occupied by snake cells
    snake_arr = np.array(snake)
    occupied = copy.deepcopy(map[snake_arr[:, 0], snake_arr[:, 1]])
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
    if (0 <= head[1] + current_dir[1] <= row_size and
            0 <= head[0] + current_dir[0] <= column_size and
            occupied[head[1] + current_dir[1], head[0] + current_dir[0]]):
        input_arr[0] = 1

    # Danger on the left
    if (0 <= head[1] + left_dir[1] <= row_size and
            0 <= head[0] + left_dir[0] <= column_size and
            occupied[head[1] + left_dir[1], head[0] + left_dir[0]]):
        input_arr[1] = 1

    # Danger on the right
    if (0 <= head[1] + right_dir[1] <= row_size and
            0 <= head[0] + right_dir[0] <= column_size and
            occupied[head[1] + right_dir[1], head[0] + right_dir[0]]):
        input_arr[2] = 1

    return input_arr


def _get_direction_neurons(current_dir):
    """
    Uses One-Hot Encoding to fill the input neurons, corresponding to UP, DOWN, LEFT, RIGHT
    """
    input_arr = np.full(shape=4, fill_value=0)

    if current_dir == [0, -1]:
        input_arr[3] = 1

    elif current_dir == [0, 1]:
        input_arr[4] = 1

    elif current_dir == [-1, 0]:
        input_arr[5] = 1

    else:
        input_arr[6] = 1

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

    else:
        input_arr[1] = 1

    if direction[0] < 0:
        input_arr[2] = 1

    else:
        input_arr[3] = 1

    return input_arr


def _get_snake_length_neuron(row_size, column_size, snake, map):
    """
    Returns the normalized length of snake (current_length / max_possible)
    """
    obstacles_count = np.sum(map)
    max_possible = row_size * column_size - obstacles_count

    input_arr = np.array([len(snake) / max_possible])

    return input_arr


def _get_distance_to_food_neuron(row_size, column_size, snake, food_pos):
    """
    Returns the normalized Manhattan distance to the food (distance / max_possible)
    """
    if not food_pos:
        input_arr = np.array([1.0])
        return input_arr

    head = snake[0]

    distance = np.abs(food_pos[0] - head[0]) + np.abs(food_pos[1] - head[1])
    max_possible_distance = row_size + column_size

    input_arr = np.array([distance / max_possible_distance])

    return input_arr


def _get_food_type_neuron(superfood):
    """
    Returns 0 if there is only general food on the map. Returns 1, if there is super-food as well
    """

    input_arr = np.array([0])

    if superfood:
        input_arr[0] = 1

    return input_arr


def _get_super_food_timer_neuron(time_left, max_time, superfood):
    """
    Returns normalized super-food timer. The `time_left` and `max_time` are counted in frame numbers
    """
    if not superfood:
        input_arr = np.array([0.0])
        return input_arr

    input_arr = np.array([time_left / max_time])

    return input_arr


def _get_obstacle_density_neurons(snake, map):
    """
    Returns normalized [-1.0, 1.0] obstacle density ratios (up/down, left/right)
    """
    input_arr = np.array([0.0, 0.0])

    head = snake[0]

    up_down_ratio = (np.sum(map[head[1] + 1:]) - np.sum(map[0:head[1]])) / np.sum(map)
    left_right_ratio = (np.sum(map[:, head[0] + 1:]) - np.sum(map[:, 0:head[0]])) / np.sum(map)

    input_arr[0] = up_down_ratio
    input_arr[1] = left_right_ratio

    return input_arr


def _get_game_ticks_neuron(current_tick, max_expected_ticks):
    """
    Returns normalized number of elapsed ticks (frames), which encourages speed and prevents infinite loops
    """
    input_arr = np.array([current_tick / max_expected_ticks])

    return input_arr


def assemble_input_neurons_array(row_size, column_size, map, snake, current_dir, food_pos, superfood_pos,
                                 superfood_time_left, superfood_max_time, current_tick, max_expected_ticks):

    input_frag1 = _get_relative_danger_loc_neurons(row_size, column_size, map, snake, current_dir)
    input_frag2 = _get_direction_neurons(current_dir)
    input_frag3 = _get_relative_food_direction_neurons(snake, food_pos)         # For general food
    input_frag4 = _get_relative_food_direction_neurons(snake, superfood_pos)    # For superfood
    input_frag5 = _get_snake_length_neuron(row_size, column_size, snake, map)
    input_frag6 = _get_distance_to_food_neuron(row_size, column_size, snake, food_pos)       # For general food
    input_frag7 = _get_distance_to_food_neuron(row_size, column_size, snake, superfood_pos)  # For superfood
    input_frag8 = _get_food_type_neuron(superfood_pos)
    input_frag9 = _get_super_food_timer_neuron(superfood_time_left, superfood_max_time, superfood_pos)
    input_frag10 = _get_obstacle_density_neurons(snake, map)
    input_frag11 = _get_game_ticks_neuron(current_tick, max_expected_ticks)

    return np.hstack([input_frag1, input_frag2, input_frag3, input_frag4, input_frag5, input_frag6, input_frag7,
                      input_frag8, input_frag9, input_frag10, input_frag11])



