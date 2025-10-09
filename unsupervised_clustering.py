import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedClustering:
    def __init__(self, weights=None, n_clusters=3, clustering_method='kmeans'):
        """
        Initialize the clustering model
        
        Parameters:
        - weights: dict or list, feature weights
        - n_clusters: int, number of clusters
        - clustering_method: str, 'kmeans' or 'dbscan'
        """
        self.weights = weights
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.model = None
        self.feature_names = None
        
    def preprocess_data(self, df):
        """
        Preprocess the data: handle NaNs, scale, and apply weights
        """
        # Separate string column and numerical data
        self.string_column = df.iloc[:, 0]
        numerical_data = df.iloc[:, 1:]
        self.feature_names = numerical_data.columns.tolist()
        
        # Handle missing values
        numerical_data_imputed = pd.DataFrame(
            self.imputer.fit_transform(numerical_data),
            columns=self.feature_names
        )
        
        # Apply feature weights if provided
        if self.weights:
            if isinstance(self.weights, dict):
                # Apply weights from dictionary {feature_name: weight}
                for feature, weight in self.weights.items():
                    if feature in numerical_data_imputed.columns:
                        numerical_data_imputed[feature] *= weight
            elif isinstance(self.weights, list):
                # Apply weights from list (assuming same order as columns)
                if len(self.weights) == len(self.feature_names):
                    for i, weight in enumerate(self.weights):
                        numerical_data_imputed.iloc[:, i] *= weight
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(numerical_data_imputed)
        
        return pd.DataFrame(scaled_data, columns=self.feature_names)
    
    def find_optimal_clusters(self, data, max_k=10):
        """
        Find optimal number of clusters using elbow method and silhouette score
        """
        wcss = []  # Within-cluster sum of squares
        silhouette_scores = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
            
            if len(np.unique(kmeans.labels_)) > 1:
                score = silhouette_score(data, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        
        # Plot elbow curve and silhouette scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        ax1.plot(range(2, max_k + 1), wcss, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('WCSS')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        # Silhouette scores
        ax2.plot(range(2, max_k + 1), silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Scores')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal k (you can modify this logic)
        optimal_k = np.argmax(silhouette_scores) + 2  # +2 because range starts from 2
        print(f"Suggested optimal number of clusters: {optimal_k}")
        
        return optimal_k
    
    def fit_predict(self, df, auto_optimize=False):
        """
        Perform clustering on the data
        """
        # Preprocess data
        processed_data = self.preprocess_data(df)
        
        # Find optimal clusters if auto_optimize is True
        if auto_optimize:
            self.n_clusters = self.find_optimal_clusters(processed_data)
        
        # Initialize and fit clustering model
        if self.clustering_method == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        elif self.clustering_method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError("Unsupported clustering method. Use 'kmeans' or 'dbscan'")
        
        # Fit and predict
        clusters = self.model.fit_predict(processed_data)
        
        return clusters
    
    def visualize_clusters(self, df, clusters):
        """
        Create various visualizations for the clusters
        """
        processed_data = self.preprocess_data(df)
        
        # Reduce dimensions for 2D visualization
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(processed_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. PCA scatter plot
        scatter = axes[0, 0].scatter(reduced_data[:, 0], reduced_data[:, 1], 
                                    c=clusters, cmap='viridis', alpha=0.7)
        axes[0, 0].set_xlabel('PCA Component 1')
        axes[0, 0].set_ylabel('PCA Component 2')
        axes[0, 0].set_title('Cluster Visualization (PCA)')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. Cluster distribution
        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
        axes[0, 1].bar(unique_clusters, cluster_counts, color='lightblue')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Number of Points')
        axes[0, 1].set_title('Cluster Distribution')
        
        # 3. Feature importance (based on PCA)
        feature_importance = np.abs(pca.components_[0])
        features = self.feature_names
        axes[1, 0].barh(features, feature_importance)
        axes[1, 0].set_xlabel('Feature Importance (PCA Component 1)')
        axes[1, 0].set_title('Feature Importance in Clustering')
        
        # 4. Boxplot of features by cluster (first 4 features)
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = clusters
        
        # Select first 4 numerical features for boxplot
        num_features = min(4, len(self.feature_names))
        feature_subset = self.feature_names[:num_features]
        
        if num_features > 0:
            df_melted = df_with_clusters.melt(id_vars=['cluster'], 
                                            value_vars=feature_subset,
                                            var_name='feature', 
                                            value_name='value')
            
            sns.boxplot(data=df_melted, x='cluster', y='value', hue='feature', ax=axes[1, 1])
            axes[1, 1].set_title('Feature Distribution by Cluster')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # Print cluster statistics
        print("\nCluster Statistics:")
        print("===================")
        for cluster in unique_clusters:
            cluster_size = np.sum(clusters == cluster)
            print(f"Cluster {cluster}: {cluster_size} points ({cluster_size/len(clusters)*100:.1f}%)")

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    
    data = {
        'name': [f'Item_{i}' for i in range(100)],
        'feature1': np.random.normal(50, 15, 100),
        'feature2': np.random.normal(100, 25, 100),
        'feature3': np.random.normal(200, 40, 100),
        'feature4': np.random.normal(150, 30, 100)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some NaN values (10% of data)
    mask = np.random.random(df.iloc[:, 1:].shape) < 0.1
    df.iloc[:, 1:][mask] = np.nan
    
    # Create some clusters by modifying the data
    df.iloc[0:33, 1] += 20  # Cluster 0
    df.iloc[33:66, 2] -= 30  # Cluster 1
    df.iloc[66:100, 3] += 50  # Cluster 2
    
    return df

def main():
    """
    Main function to demonstrate the clustering pipeline
    """
    # Create or load your dataset
    print("Loading data...")
    # df = pd.read_csv('your_data.csv')  # Uncomment to load your own data
    df = create_sample_data()  # Using sample data for demonstration
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    
    # Configuration
    FEATURE_WEIGHTS = {
        'feature1': 2.0,  # Give more weight to feature1
        'feature2': 1.0,
        'feature3': 0.5,  # Give less weight to feature3
        'feature4': 1.0
    }
    
    # Alternative: Use list of weights (in same order as columns)
    # FEATURE_WEIGHTS = [2.0, 1.0, 0.5, 1.0]
    
    # Initialize clustering model
    clustering = UnsupervisedClustering(
        weights=FEATURE_WEIGHTS,
        n_clusters=3,
        clustering_method='kmeans'  # or 'dbscan'
    )
    
    # Perform clustering
    print("\nPerforming clustering...")
    clusters = clustering.fit_predict(df, auto_optimize=True)
    
    # Visualize results
    print("\nGenerating visualizations...")
    clustering.visualize_clusters(df, clusters)
    
    # Add clusters to original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    # Save results
    output_filename = 'clustered_data.csv'
    df_with_clusters.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")
    
    # Display sample of results
    print("\nSample of clustered data:")
    print(df_with_clusters.head(10))
    
    # Print cluster summary
    print(f"\nClustering completed successfully!")
    print(f"Number of clusters found: {len(np.unique(clusters))}")
    print(f"Cluster distribution:")
    print(df_with_clusters['cluster'].value_counts().sort_index())

if __name__ == "__main__":
    main()