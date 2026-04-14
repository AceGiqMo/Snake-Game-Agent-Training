import sqlite3
import os

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

BEST_AGENTS_NUM = 50

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
agent_scatter = ax.scatter([], [], color="orange", s=200, zorder=5, edgecolors="black")


def load_training_history():
    with sqlite3.connect(f"{os.getcwd()}/train_tracking.db") as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT id FROM pso_projection")
        agents_num = len(cursor.fetchall())

        cursor.execute("SELECT MAX(epoch) FROM pso_projection")
        epochs = cursor.fetchone()[0]

        cursor.execute("SELECT point_x, point_y, fitness FROM pso_projection")
        pso_data = np.array(cursor.fetchall()).astype(float)

        points = pso_data[:, [0, 1]]
        fitnesses = pso_data[:, 2]

        cursor.execute(f"SELECT point_x, point_y, fitness FROM pso_projection "
                       f"WHERE id IN (SELECT id FROM best_parameters ORDER BY fitness DESC LIMIT {BEST_AGENTS_NUM})")
        best_pso_data = np.array(cursor.fetchall()).astype(float)

        best_points = best_pso_data[:, [0, 1]]
        best_fitnesses = best_pso_data[:, 2]

        cursor.close()

    return points, best_points, fitnesses, best_fitnesses, agents_num, epochs


def update(frame):
    print(frame)
    current_points = best_points[frame * (BEST_AGENTS_NUM) : (frame + 1) * BEST_AGENTS_NUM]
    current_fitnesses = best_fitnesses[frame * (BEST_AGENTS_NUM) : (frame + 1) * BEST_AGENTS_NUM]

    agent_scatter.set_offsets(current_points)

    ax.set_title(f"Epoch {frame}\nFitness: avg={np.mean(current_fitnesses):.2f}, best={np.max(fitnesses):.2f}\n"
                 f"Best 50 agents",
                 fontsize=12, fontweight="bold")

    return agent_scatter, ax.get_title()


if __name__ == "__main__":
    points, best_points, fitnesses, best_fitnesses, agents_num, epochs = load_training_history()

    # Duplicate points are removed
    filtered_points = np.unique(np.round(np.hstack([points, fitnesses.reshape(-1, 1)]), decimals=1), axis=0)

    fitness_scatter = ax.scatter(
        filtered_points[:, 0],
        filtered_points[:, 1],
        c=filtered_points[:, 2],
        cmap="viridis",
        alpha=0.3,
        s=50,
        vmin=np.min(fitnesses),
        vmax=np.max(fitnesses),
        zorder=2
    )

    plt.colorbar(fitness_scatter, ax=ax, label="Fitness")

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")

    ax.set_xlim(best_points[:, 0].min(), best_points[:, 0].max())
    ax.set_ylim(best_points[:, 1].min(), best_points[:, 1].max())

    ax.grid(True, alpha=0.3)

    ani = FuncAnimation(fig, update, frames=epochs, interval=100, blit=False)
    ani.save(f"training_visualization/PSO_{agents_num}_{int(np.max(best_fitnesses))}.gif", writer="pillow")
    plt.close()
