import numpy as np
import copy
import logging

SUPER_FOOD_MAX_TIME = 20       # Lasts 20 frames


class SnakeGame:
    """
    Main game class managing state, logic, food generation.
    Encapsulates all game components: snake, food, obstacles, and rules.
    """
    map = None

    def __init__(self, game_mode, row_size, column_size, snake_start_cells, map, rng):
        self.game_mode = game_mode  # 1 = collision, 2 = teleport
        self.row_size = row_size
        self.column_size = column_size
        self.snake_start_cells = snake_start_cells
        self.rng = rng or np.random.default_rng()

        self.super_food_max_time = SUPER_FOOD_MAX_TIME

        self.reset(map=map)

    def reset(self, map):
        """Reset the entire game state to initial values."""
        SnakeGame.map = map        # The map is class attribute, since all agents share the same map for epoch

        self.snake = self.snake_start_cells
        self.direction = [1, 0]  # Start moving right
        self.food = self._generate_food()
        self.super_food = None
        self.super_food_timer = 0
        self.super_food_eaten = 0
        self.frame = 0
        self.eaten_food = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self._actions = {}

    def _generate_food(self):
        """Generate food in a random free cell (not on snake or obstacles)."""
        snake_arr = np.array(self.snake)

        occupied = copy.deepcopy(SnakeGame.map[snake_arr[:, 0], snake_arr[:, 1]])
        occupied[snake_arr[:, 0], snake_arr[:, 1]] = True
        free_mask = np.where(~occupied)

        return self.rng.choice(np.array(free_mask[0], free_mask[1]).T, axis=0).tolist()

    def _move_snake(self):
        """Move the snake one step in current direction."""
        head = self.snake[0].copy()
        head[0] += self.direction[0]
        head[1] += self.direction[1]

        # Check collision with obstacles or boundaries (mode 1)
        if self.game_mode == 1:
            if not (0 <= head[0] < self.column_size and 0 <= head[1] < self.row_size):
                self.game_over = True
                return

        # Teleport through walls (mode 2 only)
        elif self.game_mode == 2:
            head[0] %= self.column_size
            head[1] %= self.row_size

        # Always check obstacle- and self-collision
        if self.map[head[1], head[0]] or head in self.snake[1:]:
            self.game_over = True
            return

        # Add new head
        self.snake.insert(0, head)

    def _handle_food_eating(self):
        head = self.snake[0]

        # Eat food
        if head == self.food:
            self.food = self._generate_food()
            self.eaten_food += 1
            self.score += 1

        # Eat super-food
        elif self.super_food and head == self.super_food:
            self.super_food = None
            self.eaten_food += 1
            self.super_food_eaten += 1
            self.score += 2

        else:
            # Remove tail if no food eaten
            self.snake.pop()

        # Check victory
        if self.score >= 20:
            self.victory = True


    def _update_super_food(self):
        """Generate or expire super food based on timer."""
        if not self.super_food and self.frame % 50 == 0:
            self.super_food = self._generate_food()

            if self.super_food:
                self.super_food_timer = self.super_food_max_time

        if self.super_food:
            self.super_food_timer -= 1

            if self.super_food_timer <= 0:
                self.super_food = None

    def handle_action(self, action):
        """Process only the first valid keyboard input per frame (no multiple commands)."""
        if action == "UP" and self.direction != [0, 1] and self.direction != [0, -1]:
            self.direction = [0, -1]
            self._actions[self.frame] = action

        elif action == "DOWN" and self.direction != [0, -1] and self.direction != [0, 1]:
            self.direction = [0, 1]
            self._actions[self.frame] = action

        elif action == "LEFT" and self.direction != [1, 0] and self.direction != [-1, 0]:
            self.direction = [-1, 0]
            self._actions[self.frame] = action

        elif action == "RIGHT" and self.direction != [-1, 0] and self.direction != [1, 0]:
            self.direction = [1, 0]
            self._actions[self.frame] = action
        # Any other action is ignored

    def update(self):
        """Update game state: move snake, handle super food, increment counter."""
        self.frame += 1
        self._move_snake()
        self._handle_food_eating()
        self._update_super_food()
