import pygame
import sys
import random

# Configuration
GRID_SIZE = 20
CELL_SIZE = 30
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
FPS = 6

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
GRAY = (40, 40, 40)
DARK_GRAY = (60, 60, 60)
BLUE = (0, 100, 255)
WHITE = (255, 255, 255)
EYE_COLOR = (128, 0, 0)
FOOD_COLOR = (255, 0, 0)
SUPER_FOOD_COLOR = (0, 180, 255)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()

# Fonts
font_large = pygame.font.SysFont("Arial", 32)
font_small = pygame.font.SysFont("Arial", 18)
font_score = pygame.font.SysFont("Arial", 24)


class SnakeGame:
    """
    Main game class managing state, logic, and rendering.
    Encapsulates all game components: snake, food, obstacles, and rules.
    """

    def __init__(self, game_mode=1, num_obstacles_ratio=0.12):
        self.game_mode = game_mode  # 1 = collision, 2 = teleport
        self.num_obstacles = max(1, int(GRID_SIZE * GRID_SIZE * num_obstacles_ratio))
        self.reset()

    def reset(self):
        """Reset the entire game state to initial values."""
        self.snake = [[GRID_SIZE // 2, GRID_SIZE // 2], [GRID_SIZE // 2 - 1, GRID_SIZE // 2], [GRID_SIZE // 2 - 2, GRID_SIZE // 2]]
        self.direction = [1, 0]  # Start moving right
        self.obstacles = self._generate_obstacles()
        self.food = self._generate_food()
        self.super_food = None
        self.super_food_active = False
        self.super_food_timer = 0
        self.move_counter = 0
        self.eaten_food = 0
        self.score = 0
        self.game_over = False
        self.victory = False

    def _generate_obstacles(self):
        """Generate obstacle grid, avoiding safe zone around head."""
        obstacles = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
        all_cells = {(x, y) for x in range(1, GRID_SIZE - 1) for y in range(1, GRID_SIZE - 1)}
        occupied = {tuple(segment) for segment in self.snake}
        safe_zone = self._get_safe_zone_around_head(self.snake[0], radius=3)
        available = all_cells - occupied - safe_zone

        if len(available) < self.num_obstacles:
            self.num_obstacles = len(available)

        selected = random.sample(list(available), self.num_obstacles) if available else []
        for x, y in selected:
            obstacles[x][y] = True

        return obstacles

    def _get_safe_zone_around_head(self, head, radius=3):
        """Return set of all cells within radius around head (safe spawn zone)."""
        safe_zone = set()
        hx, hy = head
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = hx + dx, hy + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    safe_zone.add((nx, ny))
        return safe_zone

    def _generate_food(self):
        """Generate food in a random free cell (not on snake or obstacles)."""
        all_cells = {(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)}
        occupied = {tuple(segment) for segment in self.snake}
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.obstacles[x][y]:
                    occupied.add((x, y))
        available = all_cells - occupied
        return list(random.choice(list(available))) if available else None

    def _move_snake(self):
        """Move the snake one step in current direction."""
        head = self.snake[0].copy()
        head[0] += self.direction[0]
        head[1] += self.direction[1]

        # Teleport through walls (mode 2 only)
        if self.game_mode == 2:
            head[0] %= GRID_SIZE
            head[1] %= GRID_SIZE

        # Check collision with obstacles or boundaries (mode 1)
        if self.game_mode == 1:
            if not (0 <= head[0] < GRID_SIZE and 0 <= head[1] < GRID_SIZE):
                self.game_over = True
                return
        # Always check obstacle collision
        if 0 <= head[0] < GRID_SIZE and 0 <= head[1] < GRID_SIZE:
            if self.obstacles[head[0]][head[1]]:
                self.game_over = True
                return
        else:
            # Out of bounds in mode 1 = game over
            if self.game_mode == 1:
                self.game_over = True
                return

        # Self-collision
        if head in self.snake[:-1]:
            self.game_over = True
            return

        # Add new head
        self.snake.insert(0, head)

        # Eat food
        if head == self.food:
            self.food = self._generate_food()
            self.eaten_food += 1
            self.score += 1
        # Eat super food
        elif self.super_food_active and head == self.super_food:
            self.super_food = None
            self.super_food_active = False
            self.eaten_food += 2
            self.score += 2
        else:
            # Remove tail if no food eaten
            self.snake.pop()

        # Check victory
        if self.eaten_food >= 20:
            self.victory = True

    def _update_super_food(self):
        """Generate or expire super food based on timer."""
        if not self.super_food_active and self.move_counter % 50 == 0:
            self.super_food = self._generate_food()
            if self.super_food:
                self.super_food_active = True
                self.super_food_timer = 20  # Lasts 20 moves

        if self.super_food_active:
            self.super_food_timer -= 1
            if self.super_food_timer <= 0:
                self.super_food_active = False
                self.super_food = None

    def handle_input(self, event):
        """Process only the first valid keyboard input per frame (no multiple commands)."""
        if self.game_over or self.victory:
            return

        # Only process the first valid key press in this frame
        if event.type == pygame.KEYDOWN:
            # Define priority order: Up > Down > Left > Right
            # This ensures deterministic behavior when multiple keys are pressed
            if event.key == pygame.K_UP and self.direction != [0, 1]:
                self.direction = [0, -1]
            elif event.key == pygame.K_DOWN and self.direction != [0, -1]:
                self.direction = [0, 1]
            elif event.key == pygame.K_LEFT and self.direction != [1, 0]:
                self.direction = [-1, 0]
            elif event.key == pygame.K_RIGHT and self.direction != [-1, 0]:
                self.direction = [1, 0]
            # Any other key press in this frame is ignored


    def update(self):
        """Update game state: move snake, handle super food, increment counter."""
        if not self.game_over and not self.victory:
            self.move_counter += 1
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
                if self.obstacles[x][y]:
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

        mode_text = "Mode: Collision" if self.game_mode == 1 else "Mode: Teleport"
        mode_render = font_small.render(mode_text, True, BLUE)
        screen.blit(mode_render, (5, HEIGHT - 25))

        obstacle_count = sum(row.count(True) for row in self.obstacles)
        obs_text = font_small.render(f"Obstacles: {obstacle_count}", True, WHITE)
        screen.blit(obs_text, (WIDTH - obs_text.get_width() - 5, HEIGHT - 25))

        if self.super_food_active:
            timer_text = font_small.render(f"Super Food: {self.super_food_timer} moves left", True, SUPER_FOOD_COLOR)
            screen.blit(timer_text, (WIDTH // 2 - timer_text.get_width() // 2, HEIGHT - 25))

        # Game over or victory screen
        if self.game_over:
            text = font_small.render("GAME OVER! Press SPACE to restart", True, WHITE)
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2))

        elif self.victory:
            self._show_victory_screen()

        pygame.display.flip()

    def _draw_head(self, x, y, direction):
        """Draw the snake head with eyes (no mouth)."""
        rect_x = x * CELL_SIZE
        rect_y = y * CELL_SIZE
        pygame.draw.rect(screen, GREEN, (rect_x, rect_y, CELL_SIZE, CELL_SIZE))

        eye_size = CELL_SIZE // 5
        eye_offset = CELL_SIZE // 4

        left_eye = None
        right_eye = None

        if direction == [1, 0]:  # Right
            left_eye = (rect_x + CELL_SIZE - eye_offset, rect_y + eye_offset)
            right_eye = (rect_x + CELL_SIZE - eye_offset, rect_y + CELL_SIZE - eye_offset)
        elif direction == [-1, 0]:  # Left
            left_eye = (rect_x + eye_offset, rect_y + eye_offset)
            right_eye = (rect_x + eye_offset, rect_y + CELL_SIZE - eye_offset)
        elif direction == [0, -1]:  # Up
            left_eye = (rect_x + eye_offset, rect_y + eye_offset)
            right_eye = (rect_x + CELL_SIZE - eye_offset, rect_y + eye_offset)
        elif direction == [0, 1]:  # Down
            left_eye = (rect_x + eye_offset, rect_y + CELL_SIZE - eye_offset)
            right_eye = (rect_x + CELL_SIZE - eye_offset, rect_y + CELL_SIZE - eye_offset)

        pygame.draw.circle(screen, EYE_COLOR, left_eye, eye_size)
        pygame.draw.circle(screen, EYE_COLOR, right_eye, eye_size)

    def _show_victory_screen(self):
        """Display victory screen and wait for SPACE to restart."""
        screen.fill(BLACK)
        title = font_large.render("VICTORY!", True, GREEN)
        subtitle = font_small.render("You ate 20 pieces of food!", True, WHITE)
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


def show_menu():
    """Display mode selection menu and return selected game mode."""
    screen.fill(BLACK)
    title = font_large.render("SNAKE", True, GREEN)
    subtitle = font_small.render("Choose Mode:", True, WHITE)
    option1 = font_small.render("1 - Collision (Press 1)", True, BLUE)
    option2 = font_small.render("2 - Teleport (Press 2)", True, BLUE)

    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 100))
    screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, HEIGHT // 2 - 40))
    screen.blit(option1, (WIDTH // 2 - option1.get_width() // 2, HEIGHT // 2))
    screen.blit(option2, (WIDTH // 2 - option2.get_width() // 2, HEIGHT // 2 + 40))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return 1
                elif event.key == pygame.K_2:
                    return 2


def main():
    """Main game loop using OOP SnakeGame class."""
    game_mode = show_menu()
    game = SnakeGame(game_mode=game_mode)
    running = True

    while running:
        # Process all events, but only take the first valid direction change
        direction_changed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and (game.game_over or game.victory):
                    # Restart game
                    game_mode = show_menu()
                    game = SnakeGame(game_mode=game_mode)
                elif not direction_changed:  # Only allow one direction change per frame
                    game.handle_input(event)
                    direction_changed = True  # Mark as processed

        game.update()
        game.render()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()