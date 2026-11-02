"""
PCA Dimensionality Reducer for KisanQ.
Reduces CNN 1280-D embeddings to exactly 4 numerical angles for 4-qubit Quantum Circuit encoding.
"""

import sys
import pickle
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class QuantumPCAReducer:
    """
    Fits and applies PCA to reduce high-dimensional CNN features to 4 values
    and normalizes them to the range [-pi, pi] for RY rotation gates.
    """
    def __init__(self, n_components: int = config.NUM_PCA_FEATURES):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
        self.scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        self.is_fitted = False

    def fit(self, features: np.ndarray) -> "QuantumPCAReducer":
        """
        Fit PCA and Scaler on training CNN features.
        features: numpy array of shape (N, 1280)
        """
        print(f"[PCA] Fitting PCA with {self.n_components} components on input shape {features.shape}...")
        reduced = self.pca.fit_transform(features)
        self.scaler.fit(reduced)
        self.is_fitted = True
        
        explained_variance = np.sum(self.pca.explained_variance_ratio_) * 100
        print(f"[PCA] Explained variance by {self.n_components} principal components: {explained_variance:.2f}%")
        for i, var in enumerate(self.pca.explained_variance_ratio_):
            print(f"  Component {i+1}: {var*100:.2f}%")
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform 1280-D features to 4 normalized values in [-pi, pi].
        """
        if not self.is_fitted:
            raise RuntimeError("QuantumPCAReducer has not been fitted or loaded yet.")
        reduced = self.pca.transform(features)
        scaled = self.scaler.transform(reduced)
        # Clip to ensure numerical bounds stay strictly within [-pi, pi]
        scaled = np.clip(scaled, -np.pi, np.pi)
        return scaled

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Persist fitted PCA and scaler."""
        if filepath is None:
            filepath = config.SAVED_MODELS_DIR / "pca_reducer.pkl"
        with open(filepath, "wb") as f:
            pickle.dump({"pca": self.pca, "scaler": self.scaler, "is_fitted": self.is_fitted}, f)
        print(f"[PCA] Saved fitted PCA reducer to {filepath}")
        return Path(filepath)

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "QuantumPCAReducer":
        """Load fitted PCA reducer."""
        if filepath is None:
            filepath = config.SAVED_MODELS_DIR / "pca_reducer.pkl"
        if not Path(filepath).exists():
            raise FileNotFoundError(f"PCA checkpoint not found at {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        reducer = cls(n_components=data["pca"].n_components)
        reducer.pca = data["pca"]
        reducer.scaler = data["scaler"]
        reducer.is_fitted = data["is_fitted"]
        print(f"[PCA] Loaded fitted PCA reducer from {filepath}")
        return reducer


if __name__ == "__main__":
    dummy_feats = np.random.randn(50, 1280)
    reducer = QuantumPCAReducer()
    out = reducer.fit_transform(dummy_feats)
    print("Reduced shape:", out.shape)
    print("Values min/max:", out.min(), out.max())
    saved_path = reducer.save()
    loaded = QuantumPCAReducer.load(saved_path)
    print("Loaded test match:", np.allclose(out, loaded.transform(dummy_feats)))
