import random
import os
import shutil

import numpy as np
import pandas as pd

from collections import deque


class MapManager:
    def __init__(self, row_size, column_size, snake_start_cells, *, num_obstacles_ratio=0.12):
        self.row_size = row_size
        self.column_size = column_size
        self.snake_cells = snake_start_cells
        self.num_obstacles = max(1, int(self.row_size * self.column_size * num_obstacles_ratio))

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
            selected = random.sample(list(available), num_obstacles) if available else []

            for x, y in selected:
                obstacles[y][x] = True

            if self._is_map_connected_bfs(obstacles) and not self._has_dead_ends(obstacles):
                return np.array(obstacles, dtype=np.bool)

    def generate_maps(self, num) -> np.ndarray:
        maps = []

        for _ in range(num):
            maps.append(self.generate_map())

        return np.array(maps, dtype=np.bool)

    def save_maps(self, maps):
        shutil.rmtree(f"{os.getcwd()}/maps")
        os.mkdir(f"{os.getcwd()}/maps")

        for i in range(len(maps)):
            df = pd.DataFrame(maps[i])
            df.to_parquet(f'{os.getcwd()}/maps/map_{i + 1}.parquet', engine='pyarrow')

    def load_maps(self) -> np.ndarray:
        maps = []

        for filename in os.listdir(f"{os.getcwd()}/maps"):
            if not filename.endswith(".parquet"):
                continue

            df = pd.read_parquet(f"{os.getcwd()}/maps/{filename}")
            maps.append(df.to_numpy().tolist())

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

    def _has_dead_ends(self, obstacles) -> bool:
        snake_cells = {tuple(segment) for segment in self.snake_cells}
        safe_zone = self._get_safe_zone_around_head(self.snake_cells[0], radius=2)
        checked = set()

        for x in range(self.column_size):
            for y in range(self.row_size):
                if obstacles[y][x]:
                    continue
                if (x, y) in snake_cells:
                    continue
                if (x, y) in safe_zone:
                    continue
                if (x, y) in checked:
                    continue

                neighbors = self._get_free_neighbors(x, y, obstacles)

                # Corridor found
                if len(neighbors) == 1:
                    corridor = []
                    prev = None
                    current = (x, y)

                    while True:
                        corridor.append(current)
                        checked.add(current)

                        current_neighbors = self._get_free_neighbors(current[0], current[1], obstacles)
                        next_neighbors = [n for n in current_neighbors if n != prev]

                        # Dead end
                        if len(next_neighbors) == 0:
                            return True

                        # Diverging ways are found (each of them is checked later)
                        elif len(next_neighbors) >= 2:
                            break

                        prev = current
                        current = next_neighbors[0]

                        if current in corridor:
                            break

        return False
