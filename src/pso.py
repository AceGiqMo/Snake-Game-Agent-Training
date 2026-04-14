import numpy as np
import copy
import logging

logger = logging.getLogger("main.pso")

class PSO:
    def __init__(self,
                 ndims: int,
                 popsize: int,
                 positions: np.ndarray,
                 lower_bound: float,
                 upper_bound: float,
                 inertia=1,
                 c1=1,
                 c2=1,
                 rng=None):

        self.ndims = ndims
        self.popsize = popsize
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.rgen = rng or np.random.default_rng()

        self.positions = positions
        self.velocities = self.rgen.normal(loc=0.0, scale=0.1, size=(self.popsize, self.ndims))

        self.fitnesses = np.full(fill_value=-1, shape=self.popsize)

        self.global_best = self.rgen.choice(self.positions, axis=0)
        self.local_bests = copy.deepcopy(self.positions)
        self.global_best_fitness = -1
        self.local_best_fitnesses = np.full(fill_value=-1, shape=self.popsize)

    def update_fitness(self, new_fitnesses):
        mask = self.local_best_fitnesses < new_fitnesses

        self.local_bests[mask] = copy.deepcopy(self.positions[mask])
        self.local_best_fitnesses[mask] = copy.deepcopy(new_fitnesses[mask])

        if np.any(self.global_best_fitness < self.local_best_fitnesses):
            self.global_best = copy.deepcopy(self.local_bests[np.argmax(self.local_best_fitnesses)])
            self.global_best_fitness = float(self.local_best_fitnesses[np.argmax(self.local_best_fitnesses)])

        self.fitnesses = new_fitnesses


    def update_velocities(self):
        r1 = self.rgen.random()
        r2 = self.rgen.random()

        self.velocities = (self.inertia * self.velocities +
                           self.c1 * r1 * (self.local_bests - self.positions) +
                           self.c2 * r2 * (self.global_best - self.positions))

    def update_positions(self):
        self.positions += self.velocities
        self.positions = np.clip(self.positions, a_min=self.lower_bound, a_max=self.upper_bound)

    def restore_best_points(self, best_points):
        self.local_bests = best_points
        self.global_best = copy.deepcopy(self.local_bests[np.argmax(self.local_best_fitnesses)])

        logger.info("Best points were successfully restored from the previous sessions")

    def restore_best_fitnesses(self, best_fitnesses):
        self.local_best_fitnesses = best_fitnesses
        self.global_best_fitness = float(self.local_best_fitnesses[np.argmax(self.local_best_fitnesses)])

        logger.info("Best fitnesses were successfully restored from the previous sessions")

    def restore_positions(self, positions):
        self.positions = positions

        logger.info("Last positions were successfully restored from the previous session")

    def restore_velocities(self, velocities):
        self.velocities = velocities

        logger.info("Last velocities were successfully restored from the previous session")

    def restore_last_fitnesses(self, last_fitnesses):
        self.fitnesses = last_fitnesses

        logger.info("Last fitnesses were successfully restored from the previous session")
