import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedClustering:
    def __init__(self, data=None, random_state=42):
        """
        Initialize the clustering analyzer
        
        Args:
            data: numpy array or pandas DataFrame
            random_state: random seed for reproducibility
        """
        self.data = data
        self.original_data = data.copy() if data is not None else None
        self.scaled_data = None
        self.labels = None
        self.random_state = random_state
        self.results = {}
        
    def generate_sample_data(self, dataset_type='blobs', n_samples=300):
        """
        Generate sample datasets for demonstration
        
        Args:
            dataset_type: 'blobs', 'moons', 'circles', 'anisotropic'
            n_samples: number of data points
        """
        np.random.seed(self.random_state)
        
        if dataset_type == 'blobs':
            self.data, true_labels = make_blobs(n_samples=n_samples, centers=3, 
                                              cluster_std=0.8, random_state=self.random_state)
            self.data = pd.DataFrame(self.data, columns=['Feature_1', 'Feature_2'])
            
        elif dataset_type == 'moons':
            self.data, true_labels = make_moons(n_samples=n_samples, noise=0.1, 
                                              random_state=self.random_state)
            self.data = pd.DataFrame(self.data, columns=['Feature_1', 'Feature_2'])
            
        elif dataset_type == 'circles':
            self.data, true_labels = make_circles(n_samples=n_samples, noise=0.05, 
                                                factor=0.5, random_state=self.random_state)
            self.data = pd.DataFrame(self.data, columns=['Feature_1', 'Feature_2'])
            
        elif dataset_type == 'anisotropic':
            # Anisotropicly distributed data
            X, true_labels = make_blobs(n_samples=n_samples, centers=3, 
                                      random_state=self.random_state)
            transformation = [[0.6, -0.6], [-0.4, 0.8]]
            X = np.dot(X, transformation)
            self.data = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
            
        elif dataset_type == 'varied_variance':
            # Clusters with different variance
            self.data, true_labels = make_blobs(n_samples=n_samples, centers=3, 
                                              cluster_std=[1.0, 2.5, 0.5],
                                              random_state=self.random_state)
            self.data = pd.DataFrame(self.data, columns=['Feature_1', 'Feature_2'])
        
        self.original_data = self.data.copy()
        print(f"Generated {dataset_type} dataset with {n_samples} samples")
        return self.data
    
    def preprocess_data(self, method='standard'):
        """
        Preprocess the data for clustering
        
        Args:
            method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        """
        if self.data is None:
            raise ValueError("No data provided. Load data first.")
            
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
            
        self.scaled_data = scaler.fit_transform(self.data)
        print(f"Data scaled using {method} scaling")
        return self.scaled_data
    
    def find_optimal_k(self, max_k=10):
        """
        Find optimal number of clusters using Elbow method and Silhouette analysis
        
        Args:
            max_k: maximum number of clusters to test
        """
        if self.scaled_data is None:
            self.preprocess_data()
            
        wcss = []  # Within-Cluster Sum of Squares
        silhouette_scores = []
        
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(self.scaled_data)
            wcss.append(kmeans.inertia_)
            
            if len(set(labels)) > 1:  # Silhouette score requires at least 2 clusters
                silhouette_scores.append(silhouette_score(self.scaled_data, labels))
            else:
                silhouette_scores.append(0)
        
        # Plot Elbow method and Silhouette scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        ax1.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis for Optimal k')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal k (elbow point and max silhouette)
        optimal_k_elbow = self._find_elbow_point(k_range, wcss)
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        
        print(f"Optimal k (Elbow method): {optimal_k_elbow}")
        print(f"Optimal k (Silhouette): {optimal_k_silhouette}")
        
        return {
            'k_range': list(k_range),
            'wcss': wcss,
            'silhouette_scores': silhouette_scores,
            'optimal_k_elbow': optimal_k_elbow,
            'optimal_k_silhouette': optimal_k_silhouette
        }
    
    def _find_elbow_point(self, k_range, wcss):
        """Helper method to find elbow point in WCSS curve"""
        # Calculate the angle between consecutive segments
        angles = []
        for i in range(1, len(wcss) - 1):
            v1 = np.array([k_range[i-1] - k_range[i], wcss[i-1] - wcss[i]])
            v2 = np.array([k_range[i+1] - k_range[i], wcss[i+1] - wcss[i]])
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angles.append(angle)
        
        return k_range[np.argmax(angles) + 1]
    
    def apply_kmeans(self, n_clusters=3):
        """Apply K-means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(self.scaled_data)
        
        # Calculate metrics
        metrics = self._calculate_metrics(labels)
        
        self.results['kmeans'] = {
            'model': kmeans,
            'labels': labels,
            'metrics': metrics,
            'n_clusters': n_clusters,
            'centers': kmeans.cluster_centers_
        }
        
        print(f"K-means clustering completed with {n_clusters} clusters")
        self._print_metrics(metrics, "K-means")
        
        return labels
    
    def apply_dbscan(self, eps=0.5, min_samples=5):
        """Apply DBSCAN clustering"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.scaled_data)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        metrics = self._calculate_metrics(labels) if n_clusters > 1 else None
        
        self.results['dbscan'] = {
            'model': dbscan,
            'labels': labels,
            'metrics': metrics,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'parameters': {'eps': eps, 'min_samples': min_samples}
        }
        
        print(f"DBSCAN clustering: {n_clusters} clusters, {n_noise} noise points")
        if metrics:
            self._print_metrics(metrics, "DBSCAN")
        
        return labels
    
    def apply_hierarchical(self, n_clusters=3, linkage='ward'):
        """Apply Hierarchical clustering"""
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = hierarchical.fit_predict(self.scaled_data)
        
        metrics = self._calculate_metrics(labels)
        
        self.results['hierarchical'] = {
            'model': hierarchical,
            'labels': labels,
            'metrics': metrics,
            'n_clusters': n_clusters,
            'linkage': linkage
        }
        
        print(f"Hierarchical clustering completed with {n_clusters} clusters")
        self._print_metrics(metrics, "Hierarchical")
        
        return labels
    
    def apply_gmm(self, n_components=3):
        """Apply Gaussian Mixture Model"""
        gmm = GaussianMixture(n_components=n_components, random_state=self.random_state)
        labels = gmm.fit_predict(self.scaled_data)
        probabilities = gmm.predict_proba(self.scaled_data)
        
        metrics = self._calculate_metrics(labels)
        
        self.results['gmm'] = {
            'model': gmm,
            'labels': labels,
            'probabilities': probabilities,
            'metrics': metrics,
            'n_components': n_components
        }
        
        print(f"GMM clustering completed with {n_components} components")
        self._print_metrics(metrics, "GMM")
        
        return labels
    
    def _calculate_metrics(self, labels):
        """Calculate clustering metrics"""
        if len(set(labels)) <= 1:
            return None
            
        return {
            'silhouette_score': silhouette_score(self.scaled_data, labels),
            'calinski_harabasz_score': calinski_harabasz_score(self.scaled_data, labels),
            'davies_bouldin_score': davies_bouldin_score(self.scaled_data, labels)
        }
    
    def _print_metrics(self, metrics, algorithm_name):
        """Print clustering metrics"""
        if metrics:
            print(f"{algorithm_name} Metrics:")
            print(f"  Silhouette Score: {metrics['silhouette_score']:.3f}")
            print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.3f}")
            print(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
    
    def compare_algorithms(self, n_clusters=3):
        """Compare all clustering algorithms"""
        algorithms = {
            'K-means': self.apply_kmeans(n_clusters),
            'Hierarchical': self.apply_hierarchical(n_clusters),
            'GMM': self.apply_gmm(n_clusters)
        }
        
        # Compare metrics
        comparison_data = []
        for algo_name, result_key in zip(['K-means', 'Hierarchical', 'GMM'], 
                                       ['kmeans', 'hierarchical', 'gmm']):
            if result_key in self.results and self.results[result_key]['metrics']:
                metrics = self.results[result_key]['metrics']
                comparison_data.append({
                    'Algorithm': algo_name,
                    'Silhouette': metrics['silhouette_score'],
                    'Calinski-Harabasz': metrics['calinski_harabasz_score'],
                    'Davies-Bouldin': metrics['davies_bouldin_score']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nAlgorithm Comparison:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def visualize_clusters(self, algorithm_names=None, figsize=(15, 10)):
        """Visualize clustering results"""
        if algorithm_names is None:
            algorithm_names = list(self.results.keys())
        
        n_algorithms = len(algorithm_names)
        n_cols = min(3, n_algorithms)
        n_rows = (n_algorithms + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_algorithms == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Color cycle for clusters
        colors = cycle(plt.cm.tab10.colors)
        
        for idx, algo_name in enumerate(algorithm_names):
            if algo_name not in self.results:
                continue
                
            result = self.results[algo_name]
            labels = result['labels']
            
            ax = axes[idx]
            unique_labels = set(labels)
            
            # Create color map
            color_map = {label: next(colors) for label in unique_labels}
            
            for label in unique_labels:
                if label == -1:  # Noise points for DBSCAN
                    color = 'black'
                    marker = 'x'
                    label_text = 'Noise'
                else:
                    color = color_map[label]
                    marker = 'o'
                    label_text = f'Cluster {label}'
                
                mask = labels == label
                ax.scatter(self.scaled_data[mask, 0], self.scaled_data[mask, 1],
                          c=[color], marker=marker, label=label_text, alpha=0.7, s=50)
            
            # Plot centroids for K-means
            if algo_name == 'kmeans' and 'centers' in result:
                centers = result['centers']
                ax.scatter(centers[:, 0], centers[:, 1], marker='*', 
                          c='red', s=200, label='Centroids', edgecolors='black')
            
            ax.set_title(f'{algo_name.upper()} Clustering\n'
                        f'Silhouette: {result["metrics"]["silhouette_score"]:.3f}' 
                        if result['metrics'] else f'{algo_name.upper()} Clustering')
            ax.set_xlabel('Feature 1 (scaled)')
            ax.set_ylabel('Feature 2 (scaled)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(algorithm_names), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_dendrogram(self, method='ward'):
        """Plot dendrogram for hierarchical clustering"""
        if self.scaled_data is None:
            self.preprocess_data()
            
        plt.figure(figsize=(12, 8))
        
        # Calculate linkage matrix
        linkage_matrix = sch.linkage(self.scaled_data, method=method)
        
        # Plot dendrogram
        sch.dendrogram(linkage_matrix, truncate_mode='level', p=10)
        plt.title(f'Hierarchical Clustering Dendrogram ({method} linkage)')
        plt.xlabel('Sample index or (cluster size)')
        plt.ylabel('Distance')
        plt.axhline(y=linkage_matrix[-10, 2], color='r', linestyle='--', 
                   label='Suggested cut for 3 clusters')
        plt.legend()
        plt.show()
    
    def dimensionality_reduction(self, method='pca', n_components=2):
        """Apply dimensionality reduction for visualization"""
        if self.scaled_data is None:
            self.preprocess_data()
            
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=self.random_state)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=self.random_state)
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
            
        reduced_data = reducer.fit_transform(self.scaled_data)
        
        print(f"{method.upper()} explained variance ratio: "
              f"{reducer.explained_variance_ratio_.sum():.3f}" if method == 'pca' else "")
        
        return reduced_data

def main():
    """Main function to demonstrate the clustering analysis"""
    print("UNSUPERVISED CLUSTERING ANALYSIS")
    print("=" * 60)
    
    # Initialize clustering analyzer
    cluster_analyzer = UnsupervisedClustering(random_state=42)
    
    # Generate sample data
    print("\n1. Generating sample dataset...")
    data = cluster_analyzer.generate_sample_data('blobs', n_samples=300)
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    cluster_analyzer.preprocess_data('standard')
    
    # Find optimal number of clusters
    print("\n3. Finding optimal number of clusters...")
    optimal_k_results = cluster_analyzer.find_optimal_k(max_k=10)
    
    # Apply different clustering algorithms
    print("\n4. Applying clustering algorithms...")
    n_clusters = optimal_k_results['optimal_k_silhouette']
    
    # K-means
    kmeans_labels = cluster_analyzer.apply_kmeans(n_clusters=n_clusters)
    
    # DBSCAN
    dbscan_labels = cluster_analyzer.apply_dbscan(eps=0.3, min_samples=5)
    
    # Hierarchical clustering
    hierarchical_labels = cluster_analyzer.apply_hierarchical(n_clusters=n_clusters)
    
    # GMM
    gmm_labels = cluster_analyzer.apply_gmm(n_components=n_clusters)
    
    # Compare algorithms
    print("\n5. Comparing algorithms...")
    comparison_df = cluster_analyzer.compare_algorithms(n_clusters=n_clusters)
    
    # Visualize results
    print("\n6. Visualizing clustering results...")
    cluster_analyzer.visualize_clusters()
    
    # Plot dendrogram
    print("\n7. Plotting dendrogram...")
    cluster_analyzer.plot_dendrogram()
    
    # Dimensionality reduction visualization
    print("\n8. Applying dimensionality reduction...")
    pca_data = cluster_analyzer.dimensionality_reduction('pca')
    
    print("\nClustering analysis completed successfully!")
    
    return cluster_analyzer

def advanced_demo():
    """Advanced demonstration with different dataset types"""
    print("\nADVANCED DEMONSTRATION WITH DIFFERENT DATASETS")
    print("=" * 60)
    
    dataset_types = ['blobs', 'moons', 'circles', 'varied_variance']
    
    for dataset_type in dataset_types:
        print(f"\n\nANALYZING {dataset_type.upper()} DATASET")
        print("-" * 40)
        
        analyzer = UnsupervisedClustering(random_state=42)
        analyzer.generate_sample_data(dataset_type, n_samples=300)
        analyzer.preprocess_data()
        
        # Use appropriate algorithms for different datasets
        if dataset_type in ['moons', 'circles']:
            # DBSCAN works better for non-globular clusters
            analyzer.apply_dbscan(eps=0.2 if dataset_type == 'moons' else 0.1, 
                                min_samples=5)
            analyzer.apply_kmeans(n_clusters=2)
        else:
            analyzer.apply_kmeans(n_clusters=3)
            analyzer.apply_hierarchical(n_clusters=3)
        
        analyzer.visualize_clusters()

# Example of using with custom data
def use_with_custom_data():
    """Example of using the analyzer with custom data"""
    print("\nUSING WITH CUSTOM DATA")
    print("=" * 60)
    
    # Create your custom data here
    custom_data = np.random.randn(100, 2)  # Replace with your data
    custom_data = pd.DataFrame(custom_data, columns=['Feature_A', 'Feature_B'])
    
    analyzer = UnsupervisedClustering(data=custom_data)
    analyzer.preprocess_data()
    
    # Find optimal k
    optimal_k = analyzer.find_optimal_k()
    
    # Apply clustering
    analyzer.apply_kmeans(n_clusters=3)
    analyzer.visualize_clusters()
    
    return analyzer

if __name__ == "__main__":
    # Run main demonstration
    analyzer = main()
    
    # Run advanced demo
    advanced_demo()
    
    # Example with custom data (uncomment to use)
    # custom_analyzer = use_with_custom_data()