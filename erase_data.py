"""
The module erases the existing progress of training
"""

import sqlite3
import os

if __name__ == "__main__":
    # Clean the database
    with sqlite3.connect(f"{os.getcwd()}/train_tracking.db") as conn:
        cursor = conn.cursor()

        cursor.execute("DELETE FROM agent_actions")
        cursor.execute("DELETE FROM food_pos_tracks")
        cursor.execute("DELETE FROM maps_used")
        cursor.execute("DELETE FROM last_parameters")
        cursor.execute("DELETE FROM best_parameters")
        cursor.execute("DELETE FROM pso_projection")

        cursor.close()

    # Remove trained algorithms for projecting high-dimensional points onto 2D plane for visualization
    if os.path.exists(f"{os.getcwd()}/src/projection_tools/fitted_pca.pkl"):
        os.remove(f"{os.getcwd()}/src/projection_tools/fitted_pca.pkl")

    if os.path.exists(f"{os.getcwd()}/src/projection_tools/fitted_umap.pkl"):
        os.remove(f"{os.getcwd()}/src/projection_tools/fitted_umap.pkl")