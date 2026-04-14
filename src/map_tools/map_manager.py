import os
import shutil

import numpy as np
import pandas as pd

from collections import deque

import logging

logger = logging.getLogger("my_project.map_manager")


class MapManager:
    def __init__(self, row_size, column_size, snake_start_cells, rng, *, num_obstacles_ratio=0.05):
        self.row_size = row_size
        self.column_size = column_size
        self.snake_cells = snake_start_cells
        self.num_obstacles = max(1, int(self.row_size * self.column_size * num_obstacles_ratio))
        self.rng = rng or np.random.default_rng()

    def _get_safe_zone_around_head(self, head, radius=3) -> set[tuple[int, int]]:
        """Return set of all cells within radius around head (safe spawn zone)."""
        safe_zone = set()
        hx, hy = head

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = hx + dx, hy + dy
                if 0 <= nx < self.column_size and 0 <= ny < self.row_size:
                    safe_zone.add((nx, ny))

        return safe_zone

    def generate_map(self) -> np.ndarray:
        while True:
            obstacles = [[False] * self.column_size for _ in range(self.row_size)]

            all_cells = {
                (x, y)
                for x in range(1, self.column_size - 1)
                for y in range(1, self.row_size - 1)
            }

            occupied = {tuple(segment) for segment in self.snake_cells}
            safe_zone = self._get_safe_zone_around_head(self.snake_cells[0], radius=3)
            available = all_cells - occupied - safe_zone

            num_obstacles = min(self.num_obstacles, len(available))
            selected = self.rng.choice(list(available), size=num_obstacles, axis=0,
                                       replace=False).tolist() if available else []

            for x, y in selected:
                obstacles[y][x] = True

            if self._is_map_connected_bfs(obstacles):
                self._eliminate_dead_ends(obstacles)

                return np.array(obstacles, dtype=np.bool)

    def generate_maps(self, num) -> np.ndarray:
        maps = []

        for _ in range(num):
            maps.append(self.generate_map())

        logger.info(f"New {num} maps were generated")
        return np.array(maps, dtype=np.bool)

    def save_maps(self, maps):
        logger.info("The maps are getting saved into the ./maps/ directory...")

        shutil.rmtree(f"{os.getcwd()}/maps")
        os.mkdir(f"{os.getcwd()}/maps")

        for i in range(len(maps)):
            df = pd.DataFrame(maps[i])
            df.to_parquet(f'{os.getcwd()}/maps/map_{i + 1}.parquet', engine='pyarrow')

        logger.info("The maps were successfully saved")

    def load_maps(self) -> np.ndarray:
        maps = []

        for filename in sorted(os.listdir(f"{os.getcwd()}/maps"),
                               key=(lambda x: int(x.lstrip("map_").rstrip(".parquet")))):
            if not filename.endswith(".parquet"):
                continue

            df = pd.read_parquet(f"{os.getcwd()}/maps/{filename}")
            maps.append(df.to_numpy().tolist())

        if maps:
            logger.info("The existing maps were successfully loaded")

        return np.array(maps)

    def _is_map_connected_bfs(self, obstacles) -> bool:
        snake_cells = {tuple(segment) for segment in self.snake_cells}
        free_cells = []

        for x in range(self.column_size):
            for y in range(self.row_size):
                if not obstacles[y][x] and (x, y) not in snake_cells:
                    free_cells.append((x, y))

        if not free_cells:
            return True

        start = free_cells[0]
        queue = deque([start])
        visited = {start}

        while queue:
            cx, cy = queue.popleft()

            for nx, ny in self._get_free_neighbors(cx, cy, obstacles):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return len(visited) == len(free_cells)

    def _get_free_neighbors(self, x, y, obstacles) -> list[tuple[int, int]]:
        neighbors = []
        snake_cells = {tuple(segment) for segment in self.snake_cells}

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < self.column_size and 0 <= ny < self.row_size:
                if not obstacles[ny][nx] and (nx, ny) not in snake_cells:
                    neighbors.append((nx, ny))

        return neighbors

    def _eliminate_dead_ends(self, obstacles) -> bool:
        """
        This method checks if the map has dead ends, and if it does, then the code eliminates it by destroying
        one of blocks, constructing the dead end
        """
        for y in range(self.row_size):
            for x in range(self.column_size):

                if obstacles[y][x]:
                    continue

                # Scan the Manhattan neighborhood of the cell: if it contains 3 obstacles, then it is the dead end
                manhattan_neighborhood = []

                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy

                    if not(0 <= nx < self.column_size and 0 <= ny < self.row_size):
                        continue

                    if obstacles[ny][nx]:
                        manhattan_neighborhood.append([nx, ny])

                if len(manhattan_neighborhood) == 3:
                    obstacle_to_eliminate = self.rng.choice(manhattan_neighborhood, axis=0)

                    obstacles[obstacle_to_eliminate[1]][obstacle_to_eliminate[0]] = False

        return False
