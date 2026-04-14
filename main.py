import os
import sys
import logging
import sqlite3
import copy

import numpy as np

from src.snake_game import SnakeGame
from src.pso import PSO
from src.map_tools.map_manager import MapManager
from src.neural_network.agent import SnakeAgent
from src.neural_network import features_extractor

from src.tracker import save_game_actions
from src.tracker import save_food_positions
from src.tracker import save_projected_parameters
from src.tracker import save_buffered_projected_parameters
from src.tracker import save_used_map
from src.tracker import update_best_parameters
from src.tracker import update_last_parameters

import src.projection_tools.projectors as projectors

# --------------------------
# GENERAL CONFIGURATION
# --------------------------
GRID_SIZE = 20
CELL_SIZE = 30
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
GAME_MODE = 1
AGENTS_NUM = 20000
MAPS_NUM = 50
MAX_EXPECTED_FRAMES = 1000
RANDOM_SEED = 33

# --------------------------
# PSO CONFIGURATION
# --------------------------
PSO_LOWER_BOUND = -20
PSO_UPPER_BOUND = 20
PSO_INERTIA = 0.9
PSO_C1 = 1.5
PSO_C2 = 1.5
MAX_EPOCH_NUM = 30

# --------------------------
# COLORS
# --------------------------
FOOD_COLOR = (255, 0, 0)
SUPER_FOOD_COLOR = (0, 180, 255)

# --------------------------
# LOGGING CONFIGURATION
# --------------------------
session_num = len(os.listdir("logs"))

if ".DS_Store" not in os.listdir("logs"):
    session_num += 1

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

log_format = logging.Formatter("%(asctime)s [%(levelname)s] :: %(filename)s, line %(lineno)d :: %(message)s")

# Console Handler
c_handler = logging.StreamHandler(sys.stdout)
c_handler.setLevel(logging.INFO)
c_handler.setFormatter(log_format)

# File Handler
f_handler = logging.FileHandler(f"logs/session_{session_num}.log", "w")
f_handler.setLevel(logging.DEBUG)
f_handler.setFormatter(log_format)

logger.addHandler(c_handler)
logger.addHandler(f_handler)


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

        try:
            self._initialize_maps(maps_num=maps_num)
            self._initialize_agents(agents_num=agents_num)
            self._initialize_pso(lower_bound=pso_lower_bound,
                                 upper_bound=pso_upper_bound,
                                 inertia=pso_inertia,
                                 c1=pso_c1,
                                 c2=pso_c2)

            # The algorithms for projecting the high-dimensional points into 2D place for visualization of training
            projectors.load_fitted_pca()
            projectors.load_fitted_umap()

            # Points buffer for projection algorithms training
            self.__points_buffer = np.zeros(shape=(20 * len(self.agents), self.agents[0].num_parameters()))
            self.__fitness_buffer = np.zeros(shape=20 * len(self.agents))

        except:
            logger.exception("Initialization failed")
            sys.exit()

    def _initialize_maps(self, maps_num):
        """
        Loads maps if they do exist, otherwise it generates them from scratch
        """
        logger.info("Initialization of maps...")
        map_manager = MapManager(GRID_SIZE, GRID_SIZE, rng=self.rng, snake_start_cells=self.snake_start_cells)

        self.maps = map_manager.load_maps()

        if len(self.maps) != maps_num:
            logger.warning("The required map amount does coincide with the existing ones. "
                           "New maps are getting generated...")
            self.maps = map_manager.generate_maps(maps_num)
            map_manager.save_maps(self.maps)

    def _initialize_agents(self, agents_num):
        """
        Loads agents (from the latest epoch) from the table `parameters` of the database `train_tracking.db`.
        If `parameters` is empty, then it randomly initializes the parameters of the agents
        """
        logger.info("Initialization of agents...")
        self.agents = []

        for _ in range(agents_num):
            self.agents.append(SnakeAgent(rng=self.rng))

        # Extract the parameters of the latest epoch
        with sqlite3.connect(f"{os.getcwd()}/train_tracking.db") as conn:
            cursor = conn.cursor()

            # Get the latest epoch
            cursor.execute("SELECT DISTINCT epoch FROM last_parameters")
            latest_epoch = cursor.fetchone()

            # The table is empty
            if latest_epoch is None:
                return

            cursor.execute("SELECT position FROM last_parameters")
            parameters = cursor.fetchall()

            # The training configuration was changed, we must start from scratch
            if len(parameters) != len(self.agents):
                cursor.execute("DELETE FROM agent_actions")
                cursor.execute("DELETE FROM food_pos_tracks")
                cursor.execute("DELETE FROM maps_used")
                cursor.execute("DELETE FROM last_parameters")
                cursor.execute("DELETE FROM best_parameters")
                cursor.execute("DELETE FROM pso_projection")
                logger.warning("Training configuration was changed, "
                                "the tables, related to training monitoring, were truncated")

                return

            self.epoch = latest_epoch[0] + 1

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

        logger.info("Initialization of PSO algorithm...")
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
        # We need to synchronize the PSO positions with the parameters of agents
        if self.epoch == 1:
            positions = np.full(shape=(self.pso.popsize, self.pso.ndims), fill_value=0.0)

            for i in range(len(self.agents)):
                positions[i] = self.agents[i].get_weights()

            self.pso.restore_positions(positions)

            return

        # Otherwise we extract them and restore the PSO history
        with sqlite3.connect(f"{os.getcwd()}/train_tracking.db") as conn:
            cursor = conn.cursor()

            # Neural network is a synonym for position
            cursor.execute("SELECT neural_network, fitness FROM best_parameters")
            pso_stored_bests = cursor.fetchall()

            # Extract last fitnesses of agents
            cursor.execute("SELECT position, velocity, fitness FROM last_parameters")
            last_parameters = cursor.fetchall()

            cursor.close()

        agent_bests = np.full(shape=(self.pso.popsize, self.pso.ndims), fill_value=0.0)
        agent_best_fitnesses = np.full(shape=self.pso.popsize, fill_value=0.0)
        agent_last_positions = np.full(shape=(self.pso.popsize, self.pso.ndims), fill_value=0.0)
        agent_last_velocities = np.full(shape=(self.pso.popsize, self.pso.ndims), fill_value=0.0)
        agent_last_fitnesses = np.full(shape=self.pso.popsize, fill_value=0.0)

        for i in range(len(pso_stored_bests)):
            try:
                # Sometimes, the array is converted from float32 to bytes, and sometimes, float64 to bytes
                agent_bests[i] = np.frombuffer(pso_stored_bests[i][0], dtype=np.float32)
            except ValueError:
                agent_bests[i] = np.frombuffer(pso_stored_bests[i][0], dtype=np.float64)

            agent_best_fitnesses[i] = pso_stored_bests[i][1]
            agent_last_positions[i] = np.frombuffer(last_parameters[i][0], dtype=np.float64)
            agent_last_velocities[i] = np.frombuffer(last_parameters[i][1], dtype=np.float64)
            agent_last_fitnesses[i] = last_parameters[i][2]

        self.pso.restore_best_fitnesses(agent_best_fitnesses)
        self.pso.restore_best_points(agent_bests)
        self.pso.restore_positions(agent_last_positions)
        self.pso.restore_velocities(agent_last_velocities)
        self.pso.restore_last_fitnesses(agent_last_fitnesses)

    def _update_database(self, snake_games, map_number):
        """
        Fills the `parameters` table with the current epoch data.
        Fills the `game_tracks` with the tracked actions made by each agent in the games of the current epoch
        Updates the `best_parameters` table with relevant data.
        """
        with sqlite3.connect(f"{os.getcwd()}/train_tracking.db") as conn:
            cursor = conn.cursor()
            # cursor.execute("PRAGMA foreign_keys = on")

            for i in range(len(snake_games)):
                save_game_actions(db_cursor=cursor, agent_id=i + 1, actions_dict=snake_games[i]._actions,
                                  epoch_num=self.epoch)
                save_food_positions(db_cursor=cursor, agent_id=i + 1, food_pos_dict=snake_games[i]._food_history,
                                    epoch_num=self.epoch)

            logger.debug("All game actions of each agent and food positions were saved into the database")

            save_used_map(db_cursor=cursor, epoch_num=self.epoch, map_number=map_number)
            logger.debug("The used map was saved into the database")

            update_best_parameters(db_cursor=cursor, agents_num=len(self.agents),
                                   best_neural_networks=self.pso.local_bests,
                                   best_fitnesses=self.pso.local_best_fitnesses)
            logger.debug("The table of best agents was updated in the database")

            update_last_parameters(db_cursor=cursor, agents_num=len(self.agents), epoch_num=self.epoch,
                                   positions=self.pso.positions, velocities=self.pso.velocities,
                                   fitnesses=self.pso.fitnesses)
            logger.debug("Last agents' parameters were updated in the database")


            if projectors.projectors_ready:
                # Pre-projected via PCA points onto 50-100 dimensional subspace for accelerating the UMAP projection
                pca_projected_points = projectors.pca.transform(self.pso.positions)

                # Projection of 50-100 dimensional points onto 2D plane via UMAP
                umap_projected_points = projectors.umap.transform(pca_projected_points)

                save_projected_parameters(db_cursor=cursor, agents_num=len(self.agents), epoch_num=self.epoch,
                                          positions=umap_projected_points, fitnesses=self.pso.fitnesses)
                logger.debug("Each agent's positions were projected onto 2D plane for further visualization")

            else:
                idx_start = (self.epoch - 1) * len(self.agents)
                idx_end = self.epoch * len(self.agents)

                self.__points_buffer[idx_start:idx_end] = copy.deepcopy(self.pso.positions)
                self.__fitness_buffer[idx_start:idx_end] = copy.deepcopy(self.pso.fitnesses)

                if self.epoch == 10:
                    projectors.pca.fit(self.__points_buffer[:10 * len(self.agents)])
                    logger.debug(f"PCA was fitted by {10 * len(self.agents)} samples")

                if self.epoch == 20:
                    pca_projected_points = projectors.pca.transform(self.__points_buffer)
                    umap_projected_points = projectors.umap.fit_transform(pca_projected_points)

                    logger.debug(f"UMAP was fitted by {20 * len(self.agents)} samples")

                    save_buffered_projected_parameters(db_cursor=cursor, agents_num=len(self.agents),
                                                       positions=umap_projected_points, fitnesses=self.__fitness_buffer)

                    logger.debug("Buffered positions were projected onto 2D plane and saved into database")

                    projectors.save_fitted_pca()
                    projectors.save_fitted_umap()

                    projectors.projectors_ready = True

            cursor.close()

    def train(self, max_epoch_num=MAX_EPOCH_NUM):
        # Each agent has its own game space
        snake_games = [SnakeGame(game_mode=self.game_mode,
                               row_size=self.row_size,
                               column_size=self.column_size,
                               snake_start_cells=self.snake_start_cells,
                               map=self.rng.choice(self.maps),
                               rng=self.rng) for _ in range(len(self.agents))]

        try:
            for _ in range(max_epoch_num):
                logger.info(f"EPOCH {self.epoch} => CURRENT_BEST_FITNESS = {self.pso.global_best_fitness}")

                map_index = self.rng.integers(low=0, high=len(self.maps) - 1)
                map = self.maps[map_index]

                logger.debug(f"The map_{map_index + 1} was chosen for the epoch")

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
                        (1000 - game.frame) * 0.01 +   # Small penalty for time (encourages speed)
                        game.super_food_eaten * 100              # Bonus for super food
                    )

                    status = "VICTORY" if game.victory else "LOSS"
                    logger.debug(f"Agent {game_index + 1} completed the game with {status} "
                                  f"and fitness = {fitnesses[game_index]} | "
                                 f"[food_overall = {game.eaten_food}, super_food = {game.super_food_eaten}]")

                    game_index += 1

                # Since we CANNOT calculate fitness via an explicit function with known values,
                # the computed fitnesses are passed manually to PSO method
                self.pso.update_fitness(np.array(fitnesses))

                self._update_database(snake_games=snake_games, map_number=int(map_index + 1))

                self.pso.update_velocities()
                self.pso.update_positions()

                self.epoch += 1

        except:
            logger.exception("Fatal error occurred during the training")
            sys.exit()

        else:
            logger.info("The training session was successfully completed")


if __name__ == "__main__":
    main = Main(rng=np.random.default_rng(seed=RANDOM_SEED))
    main.train()
