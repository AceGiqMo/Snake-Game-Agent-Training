import sqlite3


def save_game_actions(db_cursor: sqlite3.Cursor, agent_id, actions_dict: dict, epoch_num: int):
    for (frame, action) in actions_dict.items():
        db_cursor.execute("INSERT INTO agent_actions (id, frame, movement, epoch) VALUES (?, ?, ?, ?)",
                          (agent_id, frame, action, epoch_num))


def save_food_positions(db_cursor: sqlite3.Cursor, agent_id, epoch_num, food_pos_dict: dict):
    for (frame, data) in food_pos_dict.items():
        food_x = food_pos_dict[frame].get("food_x")
        food_y = food_pos_dict[frame].get("food_y")
        super_food_x = food_pos_dict[frame].get("super_food_x")
        super_food_y = food_pos_dict[frame].get("super_food_y")

        db_cursor.execute("INSERT INTO food_pos_tracks "
                          "(agent_id, epoch, frame, food_x, food_y, super_food_x, super_food_y) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (agent_id, epoch_num, frame, food_x, food_y, super_food_x, super_food_y))


def save_used_map(db_cursor: sqlite3.Cursor, epoch_num, map_number):
    db_cursor.execute("INSERT INTO maps_used (epoch, map_number) VALUES (?, ?)",
                      (epoch_num, map_number))


def update_best_parameters(db_cursor: sqlite3.Cursor, agents_num, best_neural_networks, best_fitnesses):
    db_cursor.execute("DELETE FROM best_parameters")

    for i in range(agents_num):
        db_cursor.execute("INSERT INTO best_parameters (id, neural_network, fitness) VALUES (?, ?, ?)",
                          (i + 1, sqlite3.Binary(best_neural_networks[i]), float(best_fitnesses[i])))


def update_last_parameters(db_cursor: sqlite3.Cursor, agents_num, epoch_num, positions, velocities, fitnesses):
    db_cursor.execute("DELETE FROM last_parameters")

    for i in range(agents_num):
        db_cursor.execute("INSERT INTO last_parameters (id, epoch, position, velocity, fitness) "
                          "VALUES (?, ?, ?, ?, ?)",
                          (i + 1, epoch_num, sqlite3.Binary(positions[i]),
                           sqlite3.Binary(velocities[i]), float(fitnesses[i])))


def save_projected_parameters(db_cursor: sqlite3.Cursor, agents_num, epoch_num, positions, fitnesses):
    for i in range(agents_num):
        db_cursor.execute("INSERT INTO pso_projection (id, epoch, point_x, point_y, fitness) "
                          "VALUES (?, ?, ?, ?, ?)",
                          (i + 1, epoch_num, float(positions[i, 0]), float(positions[i, 1]),
                           float(fitnesses[i])))


def save_buffered_projected_parameters(db_cursor: sqlite3.Cursor, agents_num, positions, fitnesses):
    for i in range(len(positions)):
        db_cursor.execute("INSERT INTO pso_projection (id, epoch, point_x, point_y, fitness) "
                          "VALUES (?, ?, ?, ?, ?)",
                          ((i % agents_num) + 1, (i // agents_num) + 1,
                           float(positions[i, 0]), float(positions[i, 1]), float(fitnesses[i])))