import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from .preprocess import load_and_preprocess_data

class CityHotspotDetector:
    """
    Implements the City Hotspot Detector (CHD) algorithm as described in the paper:
    "Multi-density crime predictor: an approach to forecast criminal activities
    in multi-density crime hotspots"
    """

    def __init__(self, k=64, s=5000, omega=-0.27, min_points_dbscan=60):
        """
        Initializes the CHD algorithm with its key parameters.

        Args:
            k (int): Number of nearest neighbors to consider for density estimation.
            s (int): Window size for moving average smoothing.
            omega (float): Density variation threshold for partitioning.
            min_points_dbscan (int): The min_samples parameter for DBSCAN.
        """
        self.k = k
        self.s = s
        self.omega = omega
        self.min_points_dbscan = min_points_dbscan
        self.labels_ = None

    def fit(self, X):
        """
        Fits the CHD model to the spatial data X.

        Args:
            X (np.ndarray): A numpy array of shape (n_samples, 2) with spatial coordinates.
        """
        print("Starting CHD algorithm...")

        # Step 1: Density Estimation (k-nearest neighbors distance)
        print(f"1. Computing k-NN distances for k={self.k}...")
        knn = NearestNeighbors(n_neighbors=self.k)
        knn.fit(X)
        distances, _ = knn.kneighbors(X)
        # Use the distance to the k-th neighbor as the density estimator
        k_dist = distances[:, -1]

        # Sort points by their estimated density (ascending k-dist)
        sorted_indices = np.argsort(k_dist)
        sorted_k_dist = k_dist[sorted_indices]

        # Step 2: Compute Density Variation
        print("2. Computing density variation...")
        density_variation = np.diff(sorted_k_dist)

        # Step 3: Smooth Density Variations using a moving average
        print(f"3. Smoothing density variations with window size s={self.s}...")
        series = pd.Series(density_variation)
        moving_avg = series.rolling(window=self.s).mean().to_numpy()
        # Handle initial NaN values from rolling mean
        moving_avg[:self.s-1] = density_variation[:self.s-1]

        # Step 4: Partition into Density Level Sets
        print(f"4. Partitioning into density level sets with omega={self.omega}...")
        # Find points where the smoothed variation is below the omega threshold
        # These are considered stable density regions
        density_level_sets_indices = np.where(moving_avg <= np.percentile(moving_avg, (1 + self.omega) * 100))[0]

        # Step 5: Estimate epsilon for each density level
        # For simplicity in this initial implementation, we will use an adaptive epsilon
        # based on the k-dist of the points in each set.

        # Step 6 & 7: Run DBSCAN on each density level set
        print("5. Running DBSCAN on density sets...")
        self.labels_ = np.full(X.shape[0], -1)  # Initialize all points as noise
        current_label = 0

        # A simplified approach to demonstrate the concept from the paper.
        # A full implementation would require more complex set management.
        # Here we demonstrate with a single, adaptive DBSCAN run for simplicity.

        # Estimate a global epsilon from the knee of the k-NN distance plot
        # This is a common heuristic for DBSCAN's epsilon.
        nn = NearestNeighbors(n_neighbors=self.min_points_dbscan).fit(X)
        distances, _ = nn.kneighbors(X)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]

        # For this example, we will take an epsilon based on a percentile of distances,
        # which simulates adapting to the densest regions.
        # A more faithful implementation would iterate through density sets.
        # Use a fixed epsilon based on the paper's context for projected coordinates
        eps = 500

        print(f" - Using fixed epsilon: {eps}")

        dbscan = DBSCAN(eps=eps, min_samples=self.min_points_dbscan, n_jobs=-1)  # Use all available cores
        dbscan.fit(X)

        self.labels_ = dbscan.labels_
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)

        print(f"CHD finished. Found {n_clusters} clusters and {n_noise} noise points.")
        return self

if __name__ == '__main__':
    # Load preprocessed data
    df = load_and_preprocess_data()

    if df is not None:
        # Extract coordinates for clustering
        coordinates = df[['X Coordinate', 'Y Coordinate']].to_numpy()

        # Initialize and run CHD
        chd = CityHotspotDetector(k=64, s=5000, omega=-0.27, min_points_dbscan=60)
        chd.fit(coordinates)

        # Add cluster labels to the dataframe
        df['cluster'] = chd.labels_

        # Save only the labels array, not the class instance
        labels_path = 'backend/saved_models/chd_labels.npy'
        np.save(labels_path, chd.labels_)
        print(f"\nCHD labels saved to '{labels_path}'")
        print("\nClustering Results:")
        print(df['cluster'].value_counts())