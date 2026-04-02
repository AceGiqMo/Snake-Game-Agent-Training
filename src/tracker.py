import sqlite3


def save_game_actions(db_cursor: sqlite3.Cursor, agent_id, actions_dict: dict, epoch_num: int):
    for (frame, action) in actions_dict.items():
        db_cursor.execute("INSERT INTO game_tracks (id, frame, movement, epoch) VALUES (?, ?, ?, ?)",
                          (agent_id, frame, action, epoch_num))


def save_epoch_parameters(db_cursor, agents_num, neural_networks, velocities, epoch_num, fitnesses):
    for i in range(len(agents_num)):
        db_cursor.execute("INSERT INTO parameters (id, neural_network, velocity epoch, fitness) VALUES (?, ?, ?, ?)",
                          (i + 1, sqlite3.Binary(neural_networks[i]), sqlite3.Binary(velocities[i]),
                           epoch_num, fitnesses[i]))


def update_best_parameters(db_cursor, agents_num, best_neural_networks, best_fitnesses):
    db_cursor.execute("DELETE FROM best_parameters")

    for i in range(len(agents_num)):
        db_cursor.execute("INSERT INTO best_parameters (id, neural_network, fitness) VALUES (?, ?, ?)",
                          (i + 1, sqlite3.Binary(best_neural_networks[i]), best_fitnesses[i]))
