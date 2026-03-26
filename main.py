import os
import sqlite3

import numpy as np

# from src.snake import SnakeGame
from src.pso import PSO
from src.agent import SnakeAgent
from src.map_manager import MapManager


# Configuration
GRID_SIZE = 20
CELL_SIZE = 30
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
GAME_MODE = 1
AGENTS_NUM = 75
MAPS_NUM = 50

# Colors
FOOD_COLOR = (255, 0, 0)
SUPER_FOOD_COLOR = (0, 180, 255)


class Main:
    """
    The main class that handles the whole training process by acting as a Facade
    """
    def __init__(self, maps_num=MAPS_NUM, agents_num=AGENTS_NUM, rng=None):
        self.random_state = rng
        self.epoch = 1

        self._initialize_maps(maps_num=maps_num)
        self._initialize_agents(agents_num=agents_num)


    def _initialize_maps(self, maps_num):
        """Loads maps if they do exist, otherwise it generates them from scratch"""
        snake_start_cells = [[GRID_SIZE // 2, GRID_SIZE // 2],
                             [GRID_SIZE // 2 - 1, GRID_SIZE // 2],
                             [GRID_SIZE // 2 - 2, GRID_SIZE // 2]]

        map_manager = MapManager(GRID_SIZE, GRID_SIZE, snake_start_cells=snake_start_cells)

        self.maps = map_manager.load_maps()

        if len(self.maps) != maps_num:
            self.maps = map_manager.generate_maps(maps_num)
            map_manager.save_maps(self.maps)

    def _initialize_agents(self, agents_num):
        """Loads agents (from the latest epoch) from the table `parameters` of the database `train_tracking.db`.
        If `parameters` is empty, then it randomly initializes the parameters of the agents"""
        self.agents = []

        for _ in range(agents_num):
            self.agents.append(SnakeAgent(rng=self.random_state))

        # Extract the parameters of the latest epoch
        with sqlite3.connect(f"{os.getcwd()}/train_tracking.db") as conn:
            cursor = conn.cursor()

            # Get the latest epoch
            cursor.execute("SELECT MAX(epoch) FROM parameters")
            latest_epoch = cursor.fetchone()[0]

            # The table is empty
            if latest_epoch is None:
                return

            cursor.execute("SELECT neural_network FROM parameters WHERE epoch = ?", (latest_epoch,))
            parameters = cursor.fetchall()

            # The training configuration was changed, it is better to start from scratch
            if len(parameters) != len(self.agents):
                # TODO: Logging
                cursor.execute("DELETE FROM parameters")
                return

            self.epoch = latest_epoch + 1

            cursor.close()

        # Restore the parameters of the agents
        for i in range(agents_num):
            params = np.frombuffer(parameters[i][0])
            self.agents[i].set_weights(params)

    def train(self):
        ...













if __name__ == "__main__":
    main = Main()
    main.train()
