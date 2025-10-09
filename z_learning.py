import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedClustering:
    def __init__(self, dataframe, string_column, feature_weights=None, outlier_fraction=0.1):
        """
        Initialize the clustering model
        
        Parameters:
        - dataframe: pandas DataFrame
        - string_column: name of the string column
        - feature_weights: dictionary of feature weights for outlier detection
        - outlier_fraction: expected proportion of outliers
        """
        self.df = dataframe.copy()
        self.string_column = string_column
        self.feature_weights = feature_weights or {}
        self.outlier_fraction = outlier_fraction
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.cluster_model = None
        self.outlier_model = None
        
    def preprocess_data(self):
        """Preprocess the data: handle NaNs and normalize numerical data"""
        # Separate string and numerical data
        self.string_data = self.df[self.string_column]
        self.numerical_data = self.df.drop(columns=[self.string_column])
        
        # Store column names for later
        self.numerical_columns = self.numerical_data.columns.tolist()
        
        # Handle NaN values
        self.numerical_clean = pd.DataFrame(
            self.imputer.fit_transform(self.numerical_data),
            columns=self.numerical_columns,
            index=self.numerical_data.index
        )
        
        # Normalize the data
        self.numerical_normalized = pd.DataFrame(
            self.scaler.fit_transform(self.numerical_clean),
            columns=self.numerical_columns,
            index=self.numerical_clean.index
        )
        
        print(f"Original data shape: {self.df.shape}")
        print(f"Numerical data shape: {self.numerical_normalized.shape}")
        print(f"Missing values handled: {self.numerical_data.isnull().sum().sum()} NaNs imputed")
        
        return self.numerical_normalized
    
    def apply_feature_weights(self, data):
        """Apply feature weights for outlier detection"""
        if not self.feature_weights:
            return data
            
        weighted_data = data.copy()
        for feature, weight in self.feature_weights.items():
            if feature in weighted_data.columns:
                weighted_data[feature] = weighted_data[feature] * weight
                print(f"Applied weight {weight} to feature '{feature}'")
        
        return weighted_data
    
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        data = self.numerical_normalized
        wcss = []  # Within-cluster sum of squares
        silhouette_scores = []
        
        for i in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
            
            if len(np.unique(kmeans.labels_)) > 1:  # Silhouette score requires at least 2 clusters
                silhouette_scores.append(silhouette_score(data, kmeans.labels_))
            else:
                silhouette_scores.append(0)
        
        # Plot elbow curve and silhouette scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        ax1.plot(range(2, max_clusters + 1), wcss, marker='o')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('WCSS')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        # Silhouette scores
        ax2.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', color='orange')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Scores')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal clusters (simple method - you can enhance this)
        optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because we started from 2 clusters
        print(f"Suggested optimal number of clusters: {optimal_clusters}")
        
        return optimal_clusters
    
    def perform_clustering(self, n_clusters=None):
        """Perform clustering and outlier detection"""
        data = self.numerical_normalized
        
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        
        # Perform K-means clustering
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.cluster_model.fit_predict(data)
        
        # Outlier detection with feature weights
        weighted_data = self.apply_feature_weights(data)
        self.outlier_model = IsolationForest(
            contamination=self.outlier_fraction, 
            random_state=42
        )
        outlier_scores = self.outlier_model.fit_predict(weighted_data)
        
        # Convert outlier scores to labels (-1 for outliers, 1 for inliers)
        outlier_labels = np.where(outlier_scores == -1, -1, 0)
        
        # Combine clustering and outlier detection
        # Outliers get label -1, others keep their cluster labels
        final_labels = np.where(outlier_labels == -1, -1, cluster_labels)
        
        self.cluster_results = {
            'cluster_labels': cluster_labels,
            'outlier_labels': outlier_labels,
            'final_labels': final_labels,
            'outlier_scores': self.outlier_model.decision_function(weighted_data)
        }
        
        print(f"Clustering completed with {n_clusters} clusters")
        print(f"Outliers detected: {np.sum(outlier_labels == -1)}")
        
        return self.cluster_results
    
    def visualize_results(self):
        """Create comprehensive visualizations of the clustering results"""
        data = self.numerical_normalized
        final_labels = self.cluster_results['final_labels']
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2, random_state=42)
        data_2d = pca.fit_transform(data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Clustering results (2D PCA)
        scatter1 = axes[0, 0].scatter(data_2d[:, 0], data_2d[:, 1], 
                                     c=final_labels, cmap='viridis', alpha=0.7)
        axes[0, 0].set_xlabel('PCA Component 1')
        axes[0, 0].set_ylabel('PCA Component 2')
        axes[0, 0].set_title('Cluster Visualization (PCA)')
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # Plot 2: Outlier scores distribution
        axes[0, 1].hist(self.cluster_results['outlier_scores'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 1].axvline(x=np.percentile(self.cluster_results['outlier_scores'], 
                                         self.outlier_fraction * 100), 
                          color='red', linestyle='--', label='Outlier threshold')
        axes[0, 1].set_xlabel('Outlier Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Outlier Scores Distribution')
        axes[0, 1].legend()
        
        # Plot 3: Cluster sizes
        unique_labels, counts = np.unique(final_labels, return_counts=True)
        colors = ['red' if label == -1 else 'skyblue' for label in unique_labels]
        bars = axes[1, 0].bar([f'Cluster {label}' if label != -1 else 'Outliers' 
                              for label in unique_labels], counts, color=colors)
        axes[1, 0].set_xlabel('Clusters')
        axes[1, 0].set_ylabel('Number of Points')
        axes[1, 0].set_title('Cluster Sizes')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           str(count), ha='center', va='bottom')
        
        # Plot 4: Feature importance in PCA
        pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        im = axes[1, 1].imshow(pca_loadings, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_xticklabels(['PC1', 'PC2'])
        axes[1, 1].set_yticks(range(len(self.numerical_columns)))
        axes[1, 1].set_yticklabels(self.numerical_columns)
        axes[1, 1].set_title('PCA Feature Loadings')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()
        
        # Print explained variance
        print(f"PCA Explained Variance: {pca.explained_variance_ratio_}")
        print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.3f}")
    
    def create_final_dataframe(self):
        """Create final dataframe with cluster labels and save to CSV"""
        # Add all results to the original dataframe
        self.df['cluster_label'] = self.cluster_results['cluster_labels']
        self.df['is_outlier'] = (self.cluster_results['outlier_labels'] == -1)
        self.df['outlier_score'] = self.cluster_results['outlier_scores']
        self.df['final_cluster'] = self.cluster_results['final_labels']
        
        # Create a meaningful cluster name
        self.df['cluster_name'] = self.df['final_cluster'].apply(
            lambda x: f"Outlier" if x == -1 else f"Cluster_{x}"
        )
        
        print("\nCluster Summary:")
        print(self.df['cluster_name'].value_counts().sort_index())
        
        return self.df
    
    def save_results(self, filename='clustered_data.csv'):
        """Save the results to a CSV file"""
        final_df = self.create_final_dataframe()
        final_df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        
        return final_df

# Example usage and demonstration with sample data
def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    data = {
        'item_name': [f'Item_{i}' for i in range(100)],
        'feature_1': np.concatenate([np.random.normal(10, 1, 40), 
                                   np.random.normal(20, 2, 40),
                                   np.random.normal(5, 1.5, 20)]),
        'feature_2': np.concatenate([np.random.normal(5, 0.5, 40), 
                                   np.random.normal(15, 1, 40),
                                   np.random.normal(20, 3, 20)]),
        'feature_3': np.concatenate([np.random.normal(100, 10, 40), 
                                   np.random.normal(200, 20, 40),
                                   np.random.normal(50, 15, 20)])
    }
    
    df = pd.DataFrame(data)
    
    # Add some NaN values
    nan_indices = np.random.choice(df.index, size=15, replace=False)
    for col in ['feature_1', 'feature_2', 'feature_3']:
        df.loc[nan_indices[:5], col] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, size=8, replace=False)
    df.loc[outlier_indices, 'feature_1'] *= 3
    df.loc[outlier_indices, 'feature_2'] *= 4
    
    return df

# Main execution
if __name__ == "__main__":
    # Create or load your dataset
    # df = pd.read_csv('your_data.csv')  # Load your actual data
    df = create_sample_data()  # Using sample data for demonstration
    
    print("Original Dataset:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum())
    
    # Define feature weights for outlier detection
    feature_weights = {
        'feature_1': 1.5,  # Give more weight to feature_1 in outlier detection
        'feature_2': 1.0,
        'feature_3': 0.8   # Give less weight to feature_3
    }
    
    # Initialize and run clustering
    clustering = UnsupervisedClustering(
        dataframe=df,
        string_column='item_name',
        feature_weights=feature_weights,
        outlier_fraction=0.1
    )
    
    # Preprocess data
    normalized_data = clustering.preprocess_data()
    
    # Perform clustering (you can specify n_clusters or let it find automatically)
    results = clustering.perform_clustering(n_clusters=3)  # or None for automatic
    
    # Visualize results
    clustering.visualize_results()
    
    # Save results
    final_df = clustering.save_results('clustered_results.csv')
    
    print("\nFinal DataFrame with clusters:")
    print(final_df[['item_name', 'cluster_name', 'is_outlier', 'outlier_score']].head(10))
    