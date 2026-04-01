import os
import sqlite3

import numpy as np

from src.snake import SnakeGame
from src.pso import PSO
from src.map_manager import MapManager
from src.neural_network.agent import SnakeAgent
from src.neural_network import features_extractor


# Configuration
GRID_SIZE = 20
CELL_SIZE = 30
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
GAME_MODE = 1
AGENTS_NUM = 75
MAPS_NUM = 50
MAX_EXPECTED_FRAMES = 1000


# PSO configuration
PSO_LOWER_BOUND = -10
PSO_UPPER_BOUND = 10
PSO_INERTIA = 0.9
PSO_C1 = 1.5
PSO_C2 = 1.5
MAX_EPOCH_NUM = 10

# Colors
FOOD_COLOR = (255, 0, 0)
SUPER_FOOD_COLOR = (0, 180, 255)


class Main:
    """
    The main class that handles the whole training process by acting as a Facade
    """
    def __init__(self,
                 maps_num=MAPS_NUM,
                 agents_num=AGENTS_NUM,
                 game_mode=GAME_MODE,
                 row_size=GRID_SIZE,
                 column_size=GRID_SIZE,
                 pso_lower_bound=PSO_LOWER_BOUND,
                 pso_upper_bound=PSO_UPPER_BOUND,
                 pso_inertia=PSO_INERTIA,
                 pso_c1=PSO_C1,
                 pso_c2=PSO_C2,
                 rng=None):

        self.snake_start_cells = [[GRID_SIZE // 2, GRID_SIZE // 2],
                             [GRID_SIZE // 2 - 1, GRID_SIZE // 2],
                             [GRID_SIZE // 2 - 2, GRID_SIZE // 2]]

        self.game_mode = game_mode

        self.row_size = row_size
        self.column_size = column_size

        self.rng = rng or np.random.default_rng()

        self.epoch = 1

        self._initialize_maps(maps_num=maps_num)
        self._initialize_agents(agents_num=agents_num)
        self._initialize_pso(lower_bound=pso_lower_bound,
                             upper_bound=pso_upper_bound,
                             inertia=pso_inertia,
                             c1=pso_c1,
                             c2=pso_c2)

    def _initialize_maps(self, maps_num):
        """
        Loads maps if they do exist, otherwise it generates them from scratch
        """
        map_manager = MapManager(GRID_SIZE, GRID_SIZE, snake_start_cells=self.snake_start_cells)

        self.maps = map_manager.load_maps()

        if len(self.maps) != maps_num:
            self.maps = map_manager.generate_maps(maps_num)
            map_manager.save_maps(self.maps)

    def _initialize_agents(self, agents_num):
        """
        Loads agents (from the latest epoch) from the table `parameters` of the database `train_tracking.db`.
        If `parameters` is empty, then it randomly initializes the parameters of the agents
        """
        self.agents = []

        for _ in range(agents_num):
            self.agents.append(SnakeAgent(rng=self.rng))

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

    def _initialize_pso(self, lower_bound, upper_bound, inertia, c1, c2):
        """
        Initializes the PSO algorithm for neural network training. If necessary, it restores
        the progress made be previous sessions
        """

        positions = np.array([agent.get_weights() for agent in self.agents])

        self.pso = PSO(
            ndims=self.agents[0].num_parameters(),
            popsize=len(self.agents),
            positions=positions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            inertia=inertia,
            c1=c1,
            c2=c2,
            rng=self.rng
        )

        # If this is the beginning of training, we have nothing to fetch from the best stored neural networks
        if self.epoch > 1:
            return

        # Otherwise we extract them and restore the PSO history
        with sqlite3.connect(f"{os.getcwd()}/train_tracking.db") as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT neural_network, fitness FROM best_parameters")
            pso_stored_bests = cursor.fetchall()

            # The training configuration was changed, it is better to start from scratch
            if len(pso_stored_bests) != len(self.agents):
                # TODO: Logging
                cursor.execute("DELETE FROM best_parameters")
                return

            # Extract last fitnesses of agents
            cursor.execute("SELECT fitness FROM parameters WHERE epoch = ?", (self.epoch - 1,))
            last_fitnesses = cursor.fetchall()

            cursor.close()

        agent_bests = np.full(shape=(self.pso.popsize, self.pso.ndims), fill_value=0.0)
        agent_best_fitnesses = np.full(shape=self.pso.popsize, fill_value=0.0)
        agent_last_fitnesses = np.full(shape=self.pso.popsize, fill_value=0.0)

        for i in range(len(pso_stored_bests)):
            agent_bests[i] = np.frombuffer(pso_stored_bests[i][0])
            agent_best_fitnesses[i] = pso_stored_bests[i][1]
            agent_last_fitnesses[i] = last_fitnesses[i][0]

        self.pso.restore_best_points(agent_bests)
        self.pso.restore_best_fitnesses(agent_best_fitnesses)
        self.pso.restore_last_fitnesses(agent_last_fitnesses)

    def train(self, max_epoch_num=MAX_EPOCH_NUM):
        # Each agent has its own game space
        snake_games = [SnakeGame(game_mode=self.game_mode,
                               row_size=self.row_size,
                               column_size=self.column_size,
                               snake_start_cells=self.snake_start_cells,
                               map=self.rng.choice(self.maps),
                               rng=self.rng) for _ in range(len(self.agents)) for _ in range(len(self.agents))]

        try:
            for _ in range(max_epoch_num):
                map = self.rng.choice(self.maps)
                fitnesses = np.full(fill_value=-1, shape=len(self.agents))

                game_index = 0

                for (game, agent) in zip(snake_games, self.agents):
                    game.reset(map=map)

                    while not game.game_over and not game.victory:
                        input_array = features_extractor.assemble_input_neurons_array(
                            row_size=self.row_size,
                            column_size=self.column_size,
                            map=SnakeGame.map,
                            snake=game.snake,
                            current_dir=game.direction,
                            food_pos=game.food,
                            superfood_pos=game.super_food,
                            superfood_time_left=game.super_food_timer,
                            superfood_max_time=game.super_food_max_time,
                            current_tick=game.frame,
                            max_expected_ticks=MAX_EXPECTED_FRAMES
                        )

                        next_action = agent.act(observation=input_array)
                        game.handle_action(next_action)
                        game.update()

                    fitnesses[game_index] = (
                        game.score * 100 +                       # Primary: game score
                        game.frame * 0.1 +                       # Secondary: survival time
                        game.eaten_food * 50 +                   # Bonus: food collected
                        (1000 - game.frames_survived) * 0.01 +   # Small penalty for time (encourages speed)
                        game.super_food_eaten * 100              # Bonus for super food
                    )

                    game_index += 1

                # Since we CANNOT calculate fitness via an explicit function with known values,
                # the computed fitnesses are passed manually to PSO method
                self.pso.update_fitness(np.array(fitnesses))
                self.pso.update_velocities()
                self.pso.update_positions()

                self.epoch += 1

        except:
            ...

        finally:
            ...



if __name__ == "__main__":
    main = Main()
    main.train()
