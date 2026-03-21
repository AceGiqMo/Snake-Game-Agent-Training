import numpy as np
import types
import copy


class PSO:
    def __init__(self,
                 ndims: int,
                 popsize: int,
                 positions: np.ndarray,
                 fitness_calculator: types.FunctionType | type,
                 lower_bound: float,
                 upper_bound: float,
                 inertia=1,
                 c1=1,
                 c2=1,
                 rng=None):

        self.ndims = ndims
        self.popsize = popsize
        self.fitness_calculator = fitness_calculator
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.rgen = np.random.default_rng(seed=rng)

        self.positions = positions
        self.velocities = self.rgen.normal(loc=0.0, scale=0.1, size=(self.popsize, self.ndims))

        self.fitnesses = self.fitness_calculator(self.positions)

        self.global_best = self.positions[np.argmax(self.fitnesses)]
        self.local_bests = copy.deepcopy(self.positions)
        self.global_best_fitness = -1
        self.local_best_fitnesses = np.full(fill_value=-1, shape=self.popsize)


    def update_velocities(self):
        r1 = self.rgen.random()
        r2 = self.rgen.random()

        self.velocities = (self.inertia * self.velocities +
                           self.c1 * r1 * (self.local_bests - self.positions) +
                           self.c2 * r2 * (self.global_best - self.positions))

    def update_positions(self):
        self.positions += self.velocities
        self.positions = np.clip(self.positions, a_min=self.lower_bound, a_max=self.upper_bound)

        new_fitnesses = self.fitness_calculator()
        mask = self.fitnesses < new_fitnesses

        self.local_bests[mask] = copy.deepcopy(self.positions[mask])
        self.local_best_fitnesses[mask] = np.copy(new_fitnesses[mask])

        if np.any(self.global_best_fitness < self.local_best_fitnesses):
            self.global_best = self.local_bests[np.argmax(new_fitnesses)]
            self.global_best_fitness = self.local_best_fitnesses[np.argmax(new_fitnesses)]

        self.fitnesses = new_fitnesses









