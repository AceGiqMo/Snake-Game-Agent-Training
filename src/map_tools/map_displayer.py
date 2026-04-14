import pygame
import pandas as pd
import os
import sys

from pathlib import Path

# Configuration
GRID_SIZE = 20
CELL_SIZE = 30
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
MAP_NUMBER = 1

# Colors
BLACK = (0, 0, 0)
GRAY = (40, 40, 40)
DARK_GRAY = (60, 60, 60)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Map Displayer")


class MapDisplayer:
    """
    The class that launches the window for displaying the chosen map.
    Made for testing purposes.
    """

    def __init__(self, map_number):
        self._load_map(map_number)

    def _load_map(self, map_number):
        project_folder = Path(os.getcwd()).parent.parent

        df = pd.read_parquet(f"{str(project_folder)}/maps/map_{map_number}.parquet")
        self.map = df.to_numpy().tolist()

    def render(self):
        """Render the environment"""
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

        pygame.display.flip()


def main():
    map = MapDisplayer(map_number=MAP_NUMBER)
    running = True

    while running:
        # Process all events, but only take the first valid direction change
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        map.render()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()