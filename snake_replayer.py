import pygame
import pandas as pd
import os
import sys
import sqlite3

# ----------------------
# CHOOSE THE AGENT AND THE EPOCH (BY DEFAULT THE BEST AGENT IS CHOSEN)
# ----------------------
AGENT_ID = None
EPOCH_NUM = None

# Game Configuration
GRID_SIZE = 20
CELL_SIZE = 30
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
FPS = 4

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
GRAY = (40, 40, 40)
DARK_GRAY = (60, 60, 60)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
WHITE = (255, 255, 255)
EYE_COLOR = (128, 0, 0)
FOOD_COLOR = (255, 0, 0)
SUPER_FOOD_COLOR = (0, 180, 255)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Replayer")
clock = pygame.time.Clock()

# Fonts
font_large = pygame.font.SysFont("Arial", 32)
font_small = pygame.font.SysFont("Arial", 18)
font_score = pygame.font.SysFont("Arial", 24)

# Sprites
replay_sprite = pygame.image.load("src/sprites/replay_on.bmp")
replay_sprite_scaled = pygame.transform.scale(replay_sprite, (50, 50))


class SnakeReplayer:
    """
    Main replay class managing state, logic, and rendering.
    Encapsulates all game components: snake, food, obstacles, and rules.
    """

    def __init__(self, agent_id=AGENT_ID, epoch_num=EPOCH_NUM):
        self.snake = [[GRID_SIZE // 2, GRID_SIZE // 2],
                      [GRID_SIZE // 2 - 1, GRID_SIZE // 2],
                      [GRID_SIZE // 2 - 2, GRID_SIZE // 2]]

        self.direction = [1, 0]
        self.score = 0

        self.eaten_food = 0
        self.food = None
        self.super_food = None
        self.super_food_active = False
        self.super_food_timer = 0

        self.frame = 0

        self.agent_id = agent_id
        self.from_epoch = epoch_num

        self.game_over = False
        self.victory = False

        with sqlite3.connect(f"{os.getcwd()}/train_tracking.db") as conn:
            cursor = conn.cursor()

            if not(self.agent_id and self.from_epoch):
                self.agent_id, self.from_epoch = self._load_best_agent(db_cursor=cursor)

            self.agent_actions = self._load_actions(db_cursor=cursor)
            self.food_positions = self._load_food_pos(db_cursor=cursor)
            self.map = self._load_map(db_cursor=cursor)

        self.food = [self.food_positions[self.frame]["food_x"], self.food_positions[self.frame]["food_y"]]
        self.__show_replay_button = False

    def _load_best_agent(self, db_cursor):
        db_cursor.execute("SELECT id, fitness FROM best_parameters ORDER BY fitness DESC LIMIT 1")
        best_agent_info = db_cursor.fetchone()

        best_agent_id = best_agent_info[0]
        best_agent_fitness = best_agent_info[1]

        db_cursor.execute("SELECT epoch FROM pso_projection WHERE id = ? AND fitness = ? LIMIT 1",
                          (best_agent_id, best_agent_fitness))
        best_agent_epoch = db_cursor.fetchone()[0]

        return best_agent_id, best_agent_epoch

    def _load_actions(self, db_cursor):
        db_cursor.execute("SELECT frame, movement FROM agent_actions WHERE id = ? AND epoch = ?",
                       (self.agent_id, self.from_epoch))

        agent_actions_raw = db_cursor.fetchall()
        agent_actions = {}

        for act in agent_actions_raw:
            agent_actions[act[0]] = act[1]

        print(agent_actions)

        return agent_actions

    def _load_food_pos(self, db_cursor):
        db_cursor.execute("SELECT frame, food_x, food_y, super_food_x, super_food_y FROM food_pos_tracks "
                          "WHERE agent_id = ? AND epoch = ?",
                          (self.agent_id, self.from_epoch))

        food_positions_raw = db_cursor.fetchall()
        food_positions = {}

        for notify in food_positions_raw:
            food_positions[notify[0]] = {"food_x": notify[1], "food_y": notify[2],
                                              "super_food_x": notify[3], "super_food_y": notify[4]}

        return food_positions


    def _load_map(self, db_cursor):
        db_cursor.execute("SELECT map_number FROM maps_used WHERE epoch = ?",
                          (self.from_epoch,))

        map_number = db_cursor.fetchone()[0]

        df = pd.read_parquet(f"{os.getcwd()}/maps/map_{map_number}.parquet")
        map = df.to_numpy().tolist()

        return map

    def _move_snake(self):
        """Move the snake one step in current direction."""
        head = self.snake[0].copy()
        head[0] += self.direction[0]
        head[1] += self.direction[1]

        # Check collision with obstacles or boundaries (mode 1)
        if not (0 <= head[0] < GRID_SIZE and 0 <= head[1] < GRID_SIZE):
            self.game_over = True
            return

        # Always check obstacle- and self-collision
        if self.map[head[1]][head[0]] or head in self.snake[1:]:
            self.game_over = True
            return

        # Add new head
        self.snake.insert(0, head)

        # Eat food
        if head == self.food:
            self.food = [self.food_positions[self.frame]["food_x"], self.food_positions[self.frame]["food_y"]]
            self.eaten_food += 1
            self.score += 1

        # Eat super food
        elif self.super_food_active and head == self.super_food:
            self.super_food = None
            self.super_food_active = False
            self.eaten_food += 1
            self.score += 2
        else:
            # Remove tail if no food eaten
            self.snake.pop()

        # Check victory
        if self.eaten_food >= 20:
            self.victory = True

    def _update_super_food(self):
        """Generate or expire super food based on timer."""
        if not self.super_food_active and self.frame % 50 == 0:
            self.super_food = [self.food_positions[self.frame]["super_food_x"],
                               self.food_positions[self.frame]["super_food_y"]]
            if self.super_food:
                self.super_food_active = True
                self.super_food_timer = 20  # Lasts 20 moves

        if self.super_food_active:
            self.super_food_timer -= 1

            if self.super_food_timer <= 0:
                self.super_food_active = False
                self.super_food = None


    def _make_movement(self):
        if not (turn := self.agent_actions.get(self.frame - 1)):
            return

        if turn == "UP":
            self.direction = [0, -1]
        elif turn == "DOWN":
            self.direction = [0, 1]
        elif turn == "LEFT":
            self.direction = [-1, 0]
        elif turn == "RIGHT":
            self.direction = [1, 0]


    def update(self):
        """Update game state: move snake, handle super food, increment counter."""
        if not self.game_over and not self.victory:
            self.frame += 1
            self._make_movement()
            self._move_snake()
            self._update_super_food()

    def render(self):
        """Render the entire game state to screen."""
        screen.fill(BLACK)

        # Draw grid
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, GRAY, rect, 1)

        # Draw obstacles
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.map[y][x]:
                    obs_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, DARK_GRAY, obs_rect)

        # Draw snake body
        for segment in self.snake[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREEN, rect)

        # Draw snake head with eyes
        if self.snake:
            self._draw_head(self.snake[0][0], self.snake[0][1], self.direction)

        # Draw food
        if self.food:
            food_rect = pygame.Rect(self.food[0] * CELL_SIZE, self.food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, FOOD_COLOR, food_rect)

        # Draw super food
        if self.super_food_active:
            super_food_rect = pygame.Rect(self.super_food[0] * CELL_SIZE, self.super_food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, SUPER_FOOD_COLOR, super_food_rect)
            pygame.draw.rect(screen, WHITE, super_food_rect, 2)  # Border

        # Draw UI
        score_text = font_score.render(f"Eaten: {self.eaten_food}/20 | Score: {self.score}", True, WHITE)
        screen.blit(score_text, (5, 5))

        obstacle_count = sum(row.count(True) for row in self.map)
        obs_text = font_small.render(f"Obstacles: {obstacle_count}", True, WHITE)
        screen.blit(obs_text, (WIDTH - obs_text.get_width() - 5, HEIGHT - 25))

        if self.super_food_active:
            timer_text = font_small.render(f"Super Food: {self.super_food_timer} moves left", True,
                                           SUPER_FOOD_COLOR)
            screen.blit(timer_text, (WIDTH // 2 - timer_text.get_width() // 2, HEIGHT - 25))

        # Game over or victory screen
        if self.game_over:
            title = font_large.render("GAME OVER!", True, RED)
            text = font_small.render("The replay is OVER! The agent have lost!", True, WHITE)
            screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 60))
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2))

        elif self.victory:
            self._show_victory_screen()

        else:
            if self.frame > 0 and self.frame % (FPS // 2) == 0:
                self.__show_replay_button ^= True

            if self.__show_replay_button:
                screen.blit(replay_sprite_scaled,
                      (GRID_SIZE * CELL_SIZE - replay_sprite_scaled.get_width() - 10, 10))


    def _draw_head(self, x, y, direction):
        """Draw the snake head with eyes (no mouth)."""
        rect_x = x * CELL_SIZE
        rect_y = y * CELL_SIZE
        pygame.draw.rect(screen, GREEN, (rect_x, rect_y, CELL_SIZE, CELL_SIZE))

        eye_size = CELL_SIZE // 5
        eye_offset = CELL_SIZE // 4

        if direction == [1, 0]:  # Right
            left_eye = (rect_x + CELL_SIZE - eye_offset, rect_y + eye_offset)
            right_eye = (rect_x + CELL_SIZE - eye_offset, rect_y + CELL_SIZE - eye_offset)

        elif direction == [-1, 0]:  # Left
            left_eye = (rect_x + eye_offset, rect_y + eye_offset)
            right_eye = (rect_x + eye_offset, rect_y + CELL_SIZE - eye_offset)

        elif direction == [0, -1]:  # Up
            left_eye = (rect_x + eye_offset, rect_y + eye_offset)
            right_eye = (rect_x + CELL_SIZE - eye_offset, rect_y + eye_offset)

        else:  # Down
            left_eye = (rect_x + eye_offset, rect_y + CELL_SIZE - eye_offset)
            right_eye = (rect_x + CELL_SIZE - eye_offset, rect_y + CELL_SIZE - eye_offset)

        pygame.draw.circle(screen, EYE_COLOR, left_eye, eye_size)
        pygame.draw.circle(screen, EYE_COLOR, right_eye, eye_size)

    def _show_victory_screen(self):
        """Display victory screen and wait for SPACE to restart."""
        screen.fill(BLACK)
        title = font_large.render("VICTORY!", True, GREEN)
        subtitle = font_small.render("The replay is OVER! The agent ate 20 pieces of food!", True, WHITE)
        restart = font_small.render("Press SPACE to restart", True, BLUE)

        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 60))
        screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, HEIGHT // 2))
        screen.blit(restart, (WIDTH // 2 - restart.get_width() // 2, HEIGHT // 2 + 60))
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False

    def render_welcome_text(self):
        agent_info = font_small.render(f"This is agent {self.agent_id} from epoch {self.from_epoch}",
                                       True, WHITE)
        title = font_large.render("PRESS SPACE to LAUNCH REPLAY!", True, GREEN)

        screen.blit(agent_info, (WIDTH // 2 - agent_info.get_width() // 2, HEIGHT // 2 - 120))
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 60))


def main():
    """Main game loop using OOP SnakeReplayer class."""
    game = SnakeReplayer()
    running = True
    replay_on = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and (game.game_over or game.victory):
                    # Restart replay
                    game = SnakeReplayer()

                elif event.key == pygame.K_SPACE and not replay_on:
                    replay_on = True

        if replay_on:
            game.update()

        game.render()

        if not replay_on:
            game.render_welcome_text()

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()



if __name__ == "__main__":
    main()
