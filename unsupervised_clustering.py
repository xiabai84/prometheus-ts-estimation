import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

class MixedDataClustering:
    def __init__(self, dataframe, string_column=0, random_state=42):
        """
        Initialize the clustering analyzer for mixed data types
        
        Args:
            dataframe (pd.DataFrame): Input DataFrame with string and numerical columns
            string_column (int or str): Index or name of the string column
            random_state (int): Random seed for reproducibility
        """
        self.original_df = dataframe.copy()
        self.string_column = string_column
        self.random_state = random_state
        self.numerical_data = None
        self.scaled_data = None
        self.string_labels = None
        self.cluster_results = {}
        self.encoder = LabelEncoder()
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Extract and prepare numerical data for clustering"""
        # Identify string column
        if isinstance(self.string_column, int):
            string_col_name = self.original_df.columns[self.string_column]
        else:
            string_col_name = self.string_column
        
        # Store string labels
        self.string_labels = self.original_df[string_col_name].copy()
        
        # Extract numerical data (exclude string column)
        self.numerical_data = self.original_df.drop(columns=[string_col_name])
        
        # Encode string column for some visualizations (optional)
        self.encoded_labels = self.encoder.fit_transform(self.string_labels)
        
        print(f"Data prepared:")
        print(f"  String column: '{string_col_name}' with {len(self.string_labels.unique())} unique values")
        print(f"  Numerical columns: {list(self.numerical_data.columns)}")
        print(f"  Total samples: {len(self.original_df)}")
    
    def preprocess_data(self, method='standard'):
        """
        Preprocess numerical data for clustering
        
        Args:
            method (str): 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        self.scaled_data = scaler.fit_transform(self.numerical_data)
        self.scaler = scaler
        
        print(f"Numerical data scaled using {method} scaling")
        return self.scaled_data
    
    def find_optimal_clusters(self, max_k=10, algorithm='kmeans'):
        """
        Find optimal number of clusters using multiple methods
        
        Args:
            max_k (int): Maximum number of clusters to test
            algorithm (str): Clustering algorithm to use for evaluation
        """
        if self.scaled_data is None:
            self.preprocess_data()
        
        wcss = []  # Within-Cluster Sum of Squares
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            if algorithm == 'kmeans':
                model = KMeans(n_clusters=k, random_state=self.random_state)
            elif algorithm == 'gmm':
                model = GaussianMixture(n_components=k, random_state=self.random_state)
            else:
                raise ValueError("Algorithm must be 'kmeans' or 'gmm'")
            
            labels = model.fit_predict(self.scaled_data)
            wcss.append(model.inertia_ if algorithm == 'kmeans' else np.nan)
            
            if len(set(labels)) > 1:
                silhouette_scores.append(silhouette_score(self.scaled_data, labels))
            else:
                silhouette_scores.append(0)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve (only for K-means)
        if algorithm == 'kmeans':
            ax1.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Number of Clusters (k)')
            ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
            ax1.set_title('Elbow Method for Optimal k')
            optimal_k_elbow = self._find_elbow_point(k_range, wcss)
            ax1.axvline(x=optimal_k_elbow, color='red', linestyle='--', 
                       label=f'Optimal k: {optimal_k_elbow}')
        else:
            ax1.text(0.5, 0.5, 'Elbow method only available for K-means', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Elbow Method (K-means only)')
            optimal_k_elbow = None
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis for Optimal k')
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        ax2.axvline(x=optimal_k_silhouette, color='red', linestyle='--', 
                   label=f'Optimal k: {optimal_k_silhouette}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Optimal k (Elbow method): {optimal_k_elbow}")
        print(f"Optimal k (Silhouette): {optimal_k_silhouette}")
        
        return {
            'optimal_k_elbow': optimal_k_elbow,
            'optimal_k_silhouette': optimal_k_silhouette,
            'silhouette_scores': silhouette_scores
        }
    
    def _find_elbow_point(self, k_range, wcss):
        """Helper method to find elbow point in WCSS curve"""
        if len(wcss) < 3:
            return k_range[0]
        
        # Calculate the angle between consecutive segments
        angles = []
        for i in range(1, len(wcss) - 1):
            v1 = np.array([k_range[i-1] - k_range[i], wcss[i-1] - wcss[i]])
            v2 = np.array([k_range[i+1] - k_range[i], wcss[i+1] - wcss[i]])
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angles.append(angle)
        
        return k_range[np.argmax(angles) + 1] if angles else k_range[0]
    
    def apply_kmeans(self, n_clusters=3):
        """Apply K-means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(self.scaled_data)
        
        metrics = self._calculate_metrics(labels)
        
        self.cluster_results['kmeans'] = {
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
        
        self.cluster_results['dbscan'] = {
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
        
        self.cluster_results['hierarchical'] = {
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
        
        self.cluster_results['gmm'] = {
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
            if result_key in self.cluster_results and self.cluster_results[result_key]['metrics']:
                metrics = self.cluster_results[result_key]['metrics']
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
    
    def visualize_clusters(self, algorithm_name='kmeans', figsize=(12, 8)):
        """Visualize clustering results with multiple plot types"""
        if algorithm_name not in self.cluster_results:
            print(f"No results found for {algorithm_name}")
            return
        
        result = self.cluster_results[algorithm_name]
        labels = result['labels']
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: 2D PCA projection
        pca = PCA(n_components=2, random_state=self.random_state)
        pca_data = pca.fit_transform(self.scaled_data)
        
        unique_labels = set(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise points
                color = 'black'
                marker = 'x'
                label_text = 'Noise'
            else:
                color = colors[i]
                marker = 'o'
                label_text = f'Cluster {label}'
            
            mask = labels == label
            ax1.scatter(pca_data[mask, 0], pca_data[mask, 1], 
                       c=[color], marker=marker, label=label_text, alpha=0.7, s=50)
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title(f'{algorithm_name.upper()} - PCA Projection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cluster distribution by string labels
        cluster_string_df = pd.DataFrame({
            'String_Label': self.string_labels,
            'Cluster': labels
        })
        
        cluster_counts = cluster_string_df.groupby(['String_Label', 'Cluster']).size().unstack(fill_value=0)
        cluster_counts.plot(kind='bar', stacked=True, ax=ax2, colormap='tab10')
        ax2.set_title('Cluster Distribution by String Labels')
        ax2.set_xlabel('String Labels')
        ax2.set_ylabel('Count')
        ax2.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Feature importance for clustering (using PCA loadings)
        feature_importance = np.abs(pca.components_[0])
        features = self.numerical_data.columns
        ax3.barh(features, feature_importance)
        ax3.set_title('Feature Importance (PC1 Loadings)')
        ax3.set_xlabel('Absolute Loading Value')
        
        # Plot 4: Silhouette analysis
        from sklearn.metrics import silhouette_samples
        if len(unique_labels) > 1 and -1 not in unique_labels:
            silhouette_vals = silhouette_samples(self.scaled_data, labels)
            y_lower = 10
            
            for i, label in enumerate(sorted(unique_labels)):
                cluster_silhouette_vals = silhouette_vals[labels == label]
                cluster_silhouette_vals.sort()
                size_cluster = cluster_silhouette_vals.shape[0]
                y_upper = y_lower + size_cluster
                
                color = colors[i] if label != -1 else 'black'
                ax4.fill_betweenx(np.arange(y_lower, y_upper),
                                0, cluster_silhouette_vals,
                                facecolor=color, edgecolor=color, alpha=0.7)
                ax4.text(-0.05, y_lower + 0.5 * size_cluster, str(label))
                y_lower = y_upper + 10
            
            ax4.set_xlabel('Silhouette Coefficient Values')
            ax4.set_ylabel('Cluster Label')
            ax4.set_title('Silhouette Plot')
            ax4.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--",
                       label=f'Average: {np.mean(silhouette_vals):.3f}')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Silhouette plot requires\n2+ clusters without noise',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Silhouette Plot')
        
        plt.tight_layout()
        plt.suptitle(f'{algorithm_name.upper()} Clustering Analysis', y=1.02, fontsize=16)
        plt.show()
    
    def plot_dendrogram(self, method='ward', figsize=(12, 6)):
        """Plot dendrogram for hierarchical clustering"""
        if self.scaled_data is None:
            self.preprocess_data()
        
        plt.figure(figsize=figsize)
        
        # Calculate linkage matrix
        linkage_matrix = sch.linkage(self.scaled_data, method=method)
        
        # Plot dendrogram with labels
        sch.dendrogram(linkage_matrix, 
                      labels=self.string_labels.values,
                      leaf_rotation=90,
                      leaf_font_size=10)
        plt.title(f'Hierarchical Clustering Dendrogram\n({method} linkage)')
        plt.xlabel('String Labels')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()
    
    def extend_dataframe_with_clusters(self, algorithm_name='kmeans', cluster_column_name='cluster'):
        """
        Extend original DataFrame with cluster labels and save to CSV
        
        Args:
            algorithm_name (str): Name of clustering algorithm to use
            cluster_column_name (str): Name for the new cluster column
            
        Returns:
            pd.DataFrame: Extended DataFrame
        """
        if algorithm_name not in self.cluster_results:
            raise ValueError(f"No clustering results found for {algorithm_name}")
        
        # Get cluster labels
        cluster_labels = self.cluster_results[algorithm_name]['labels']
        
        # Create extended DataFrame
        extended_df = self.original_df.copy()
        extended_df[cluster_column_name] = cluster_labels
        
        # Add cluster information
        print(f"\nCluster distribution for {algorithm_name}:")
        cluster_counts = extended_df[cluster_column_name].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            print(f"  Cluster {cluster}: {count} samples")
        
        return extended_df
    
    def save_clustered_data(self, algorithm_name='kmeans', 
                          output_file='clustered_data.csv',
                          cluster_column_name='cluster'):
        """
        Save the clustered data to CSV file
        
        Args:
            algorithm_name (str): Clustering algorithm to use
            output_file (str): Output CSV file path
            cluster_column_name (str): Name for cluster column
        """
        extended_df = self.extend_dataframe_with_clusters(algorithm_name, cluster_column_name)
        
        # Save to CSV
        extended_df.to_csv(output_file, index=False)
        print(f"\nClustered data saved to: {output_file}")
        print(f"Total samples: {len(extended_df)}")
        print(f"Cluster column: '{cluster_column_name}'")
        
        return extended_df
    
    def generate_clustering_report(self, algorithm_name='kmeans'):
        """Generate comprehensive clustering report"""
        if algorithm_name not in self.cluster_results:
            raise ValueError(f"No results found for {algorithm_name}")
        
        result = self.cluster_results[algorithm_name]
        labels = result['labels']
        
        print(f"\n{'='*60}")
        print(f"CLUSTERING REPORT: {algorithm_name.upper()}")
        print(f"{'='*60}")
        
        # Basic information
        print(f"Dataset Information:")
        print(f"  Total samples: {len(self.original_df)}")
        print(f"  Numerical features: {len(self.numerical_data.columns)}")
        print(f"  String labels: {len(self.string_labels.unique())} unique values")
        
        # Cluster information
        unique_clusters = set(labels)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = list(labels).count(-1) if -1 in labels else 0
        
        print(f"\nCluster Information:")
        print(f"  Number of clusters: {n_clusters}")
        if n_noise > 0:
            print(f"  Noise points: {n_noise}")
        
        # Cluster sizes
        print(f"\nCluster Sizes:")
        for cluster in sorted(unique_clusters):
            count = list(labels).count(cluster)
            percentage = (count / len(labels)) * 100
            if cluster == -1:
                print(f"  Noise: {count} samples ({percentage:.1f}%)")
            else:
                print(f"  Cluster {cluster}: {count} samples ({percentage:.1f}%)")
        
        # Metrics
        if result['metrics']:
            print(f"\nClustering Metrics:")
            for metric, value in result['metrics'].items():
                print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")

def create_sample_data():
    """Create sample mixed-type dataset for demonstration"""
    np.random.seed(42)
    
    # Sample string labels (cities)
    cities = ['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 
              'Berlin', 'Toronto', 'Singapore', 'Dubai', 'Mumbai']
    
    # Generate sample data
    n_samples = 200
    data = {
        'city': np.random.choice(cities, n_samples),
        'population': np.random.normal(1000000, 300000, n_samples),
        'area': np.random.normal(500, 150, n_samples),
        'gdp': np.random.normal(50000, 15000, n_samples),
        'tourists': np.random.normal(1000000, 300000, n_samples),
        'temperature': np.random.normal(20, 10, n_samples)
    }
    
    # Add some cluster structure
    df = pd.DataFrame(data)
    
    # Create artificial clusters based on city groups
    cluster_centers = {
        'New York': [1200000, 600, 60000, 1200000, 15],
        'London': [1100000, 550, 55000, 1100000, 10],
        'Tokyo': [1300000, 700, 65000, 1300000, 18],
        'Paris': [900000, 450, 45000, 900000, 12],
        'Sydney': [800000, 400, 40000, 800000, 22]
    }
    
    for idx, row in df.iterrows():
        if row['city'] in cluster_centers:
            center = cluster_centers[row['city']]
            df.loc[idx, 'population'] = center[0] + np.random.normal(0, 50000)
            df.loc[idx, 'area'] = center[1] + np.random.normal(0, 30)
            df.loc[idx, 'gdp'] = center[2] + np.random.normal(0, 5000)
            df.loc[idx, 'tourists'] = center[3] + np.random.normal(0, 50000)
            df.loc[idx, 'temperature'] = center[4] + np.random.normal(0, 3)
    
    return df

def main():
    """Main function to demonstrate the mixed data clustering"""
    print("MIXED DATA CLUSTERING ANALYSIS")
    print("=" * 60)
    
    # Create or load your dataset
    print("\n1. Loading dataset...")
    df = create_sample_data()
    print("Sample data:")
    print(df.head())
    
    # Initialize clustering analyzer
    print("\n2. Initializing clustering analyzer...")
    cluster_analyzer = MixedDataClustering(df, string_column='city', random_state=42)
    
    # Preprocess data
    print("\n3. Preprocessing numerical data...")
    cluster_analyzer.preprocess_data('standard')
    
    # Find optimal number of clusters
    print("\n4. Finding optimal number of clusters...")
    optimal_results = cluster_analyzer.find_optimal_clusters(max_k=8, algorithm='kmeans')
    
    # Apply clustering algorithms
    print("\n5. Applying clustering algorithms...")
    n_clusters = optimal_results['optimal_k_silhouette']
    
    # K-means
    kmeans_labels = cluster_analyzer.apply_kmeans(n_clusters=n_clusters)
    
    # Hierarchical clustering
    hierarchical_labels = cluster_analyzer.apply_hierarchical(n_clusters=n_clusters)
    
    # GMM
    gmm_labels = cluster_analyzer.apply_gmm(n_components=n_clusters)
    
    # Compare algorithms
    print("\n6. Comparing algorithms...")
    comparison_df = cluster_analyzer.compare_algorithms(n_clusters=n_clusters)
    
    # Visualize results
    print("\n7. Visualizing clustering results...")
    cluster_analyzer.visualize_clusters(algorithm_name='kmeans')
    cluster_analyzer.visualize_clusters(algorithm_name='hierarchical')
    
    # Plot dendrogram
    print("\n8. Plotting dendrogram...")
    cluster_analyzer.plot_dendrogram()
    
    # Generate report
    print("\n9. Generating clustering report...")
    cluster_analyzer.generate_clustering_report('kmeans')
    
    # Save clustered data
    print("\n10. Saving clustered data...")
    final_df = cluster_analyzer.save_clustered_data(
        algorithm_name='kmeans',
        output_file='clustered_city_data.csv',
        cluster_column_name='city_cluster'
    )
    
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Cluster column added: 'city_cluster'")
    print(f"File saved: 'clustered_city_data.csv'")
    
    return cluster_analyzer, final_df

# Example of using with custom data
def use_with_custom_data(csv_file_path, string_column_name):
    """Example function for using with custom CSV data"""
    print(f"\nProcessing custom data from: {csv_file_path}")
    
    # Load your custom data
    df = pd.read_csv(csv_file_path)
    
    # Initialize analyzer
    analyzer = MixedDataClustering(df, string_column=string_column_name)
    
    # Preprocess and cluster
    analyzer.preprocess_data()
    analyzer.apply_kmeans(n_clusters=3)
    
    # Visualize and save
    analyzer.visualize_clusters('kmeans')
    result_df = analyzer.save_clustered_data(output_file='my_clustered_data.csv')
    
    return analyzer, result_df

if __name__ == "__main__":
    # Run main demonstration
    analyzer, final_df = main()
    
    # Example of using with custom data (uncomment and modify as needed)
    # custom_analyzer, custom_result = use_with_custom_data(
    #     csv_file_path='your_data.csv',
    #     string_column_name='your_string_column'
    # )