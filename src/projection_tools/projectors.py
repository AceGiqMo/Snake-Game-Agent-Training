from sklearn.decomposition import PCA
from umap import UMAP

import joblib
import os


projectors_ready = False

pca: PCA | None = None
umap: UMAP | None = None


def load_fitted_pca():
    global pca, projectors_ready

    if not os.path.exists(f"{os.getcwd()}/src/projection_tools/fitted_pca.pkl"):
        pca = PCA(n_components=75)
        return

    with open(f"{os.getcwd()}/src/projection_tools/fitted_pca.pkl", "rb") as f:
        pca = joblib.load(f)

    if pca and umap:
        projectors_ready = True


def load_fitted_umap():
    global umap, projectors_ready

    if not os.path.exists(f"{os.getcwd()}/src/projection_tools/fitted_umap.pkl"):
        umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="euclidean", verbose=False)
        return

    with open(f"{os.getcwd()}/src/projection_tools/fitted_umap.pkl", "rb") as f:
        umap = joblib.load(f)

    if pca and umap:
        projectors_ready = True


def save_fitted_pca():
    with open(f"{os.getcwd()}/src/projection_tools/fitted_pca.pkl", "wb") as f:
        joblib.dump(pca, f)


def save_fitted_umap():
    with open(f"{os.getcwd()}/src/projection_tools/fitted_umap.pkl", "wb") as f:
        joblib.dump(umap, f)