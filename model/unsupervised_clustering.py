import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedUnsupervisedClustering:
    def __init__(self, outlier_fraction=0.1, weights=None, n_clusters=3, clustering_method='kmeans', 
                 outlier_method='isolation_forest', normalization_method='standard'):
        """
        Initialize the clustering model with advanced features
        
        Parameters:
        - weights: dict or list, feature weights
        - n_clusters: int, number of clusters
        - clustering_method: str, 'kmeans' or 'dbscan'
        - outlier_method: str, 'isolation_forest', 'zscore', or 'dbscan'
        - normalization_method: str, 'standard', 'robust', or 'minmax'
        """
        self.outlier_fraction = outlier_fraction
        self.weights = weights
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.outlier_method = outlier_method
        self.normalization_method = normalization_method
        
        # Initialize components
        if normalization_method == 'standard':
            self.scaler = StandardScaler()
        elif normalization_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()  # fallback
            
        self.imputer = SimpleImputer(strategy='median')  # More robust for outliers
        self.model = None
        self.feature_names = None
        self.outlier_labels = None
        
    def normalize_data(self, data):
        """Normalize data based on selected method"""
        if self.normalization_method == 'minmax':
            # Manual MinMax scaling
            normalized_data = (data - data.min()) / (data.max() - data.min())
            return normalized_data
        else:
            # Use sklearn scalers
            return pd.DataFrame(
                self.scaler.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
    
    def detect_outliers(self, data):
        """Detect outliers and return outlier labels"""
        outlier_mask = np.zeros(len(data), dtype=bool)
        
        if self.outlier_method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=self.outlier_fraction, random_state=42)
            outlier_labels = iso_forest.fit_predict(data)
            outlier_mask = outlier_labels == -1
            
        elif self.outlier_method == 'zscore':
            # Use Z-score method
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            outlier_mask = (z_scores > 3).any(axis=1)
            
        elif self.outlier_method == 'dbscan':
            # Use DBSCAN for outlier detection
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(data)
            outlier_mask = clusters == -1
            
        print(f"Detected {outlier_mask.sum()} outliers using {self.outlier_method} method")
        return outlier_mask
    
    def preprocess_data(self, df):
        """
        Preprocess the data: handle NaNs, normalize, and apply weights
        """
        # Separate string column and numerical data
        self.string_column = df.iloc[:, 0]
        self.original_index = df.index
        numerical_data = df.iloc[:, 1:].copy()
        self.feature_names = numerical_data.columns.tolist()
        
        print("Original data statistics:")
        print(numerical_data.describe())
        
        # Handle missing values
        numerical_data_imputed = pd.DataFrame(
            self.imputer.fit_transform(numerical_data),
            columns=self.feature_names,
            index=self.original_index
        )
        
        # Normalize data
        numerical_data_normalized = self.normalize_data(numerical_data_imputed)
        
        print("\nNormalized data statistics:")
        print(numerical_data_normalized.describe())
        
        # Apply feature weights if provided
        if self.weights:
            weighted_data = numerical_data_normalized.copy()
            if isinstance(self.weights, dict):
                # Apply weights from dictionary {feature_name: weight}
                for feature, weight in self.weights.items():
                    if feature in weighted_data.columns:
                        weighted_data[feature] *= weight
                        print(f"Applied weight {weight} to feature {feature}")
            elif isinstance(self.weights, list):
                # Apply weights from list (assuming same order as columns)
                if len(self.weights) == len(self.feature_names):
                    for i, weight in enumerate(self.weights):
                        weighted_data.iloc[:, i] *= weight
                        print(f"Applied weight {weight} to feature {self.feature_names[i]}")
            
            return weighted_data
        
        return numerical_data_normalized
    
    def find_optimal_clusters(self, data, max_k=10):
        """Find optimal number of clusters using multiple methods"""
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
        
        # Find optimal k using elbow method and silhouette score
        # optimal_k_elbow = self._find_elbow_point(wcss) + 2
        optimal_k_silhouette = np.argmax(silhouette_scores) + 2
        
        # Plot results
        # self._plot_optimization_results(range(2, max_k + 1), wcss, silhouette_scores)
        
        # Choose the best k (prefer silhouette score)
        optimal_k = optimal_k_silhouette
        print(f"Suggested optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def _find_elbow_point(self, wcss):
        """Find the elbow point in WCSS curve"""
        n = len(wcss)
        if n < 3:
            return 0
        
        # Calculate angles between consecutive segments
        angles = []
        for i in range(1, n-1):
            v1 = np.array([1, wcss[i-1] - wcss[i]])
            v2 = np.array([1, wcss[i+1] - wcss[i]])
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cosine, -1, 1))
            angles.append(angle)
        
        return np.argmax(angles)
    
    def _plot_optimization_results(self, k_range, wcss, silhouette_scores):
        """Plot optimization results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        ax1.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
        ax1.set_title('Elbow Method for Optimal Clusters')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis for Optimal Clusters')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def fit_predict(self, df, auto_optimize=False):
        """Perform clustering with outlier detection"""
        # Preprocess data
        processed_data = self.preprocess_data(df)
        
        # Detect outliers
        outlier_mask = self.detect_outliers(processed_data)
        
        # Separate outliers from main data
        main_data = processed_data[~outlier_mask]
        outlier_data = processed_data[outlier_mask]
        
        print(f"\nMain data points: {len(main_data)}")
        print(f"Outlier data points: {len(outlier_data)}")
        
        # Find optimal clusters if auto_optimize is True and we have enough data
        if auto_optimize and len(main_data) > 10:
            self.n_clusters = self.find_optimal_clusters(main_data)
        
        # Initialize and fit clustering model on main data
        if len(main_data) > 0:
            if self.clustering_method == 'kmeans':
                self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            elif self.clustering_method == 'dbscan':
                self.model = DBSCAN(eps=0.5, min_samples=5)
            else:
                raise ValueError("Unsupported clustering method. Use 'kmeans' or 'dbscan'")
            
            # Fit and predict on main data
            main_clusters = self.model.fit_predict(main_data)
            
            # Combine results: assign -1 to outliers, adjust main clusters to start from 0
            final_clusters = np.full(len(df), -1, dtype=int)
            final_clusters[main_data.index] = main_clusters
            final_clusters[outlier_data.index] = -1  # Outliers labeled as -1
            
        else:
            # If no main data, all are outliers
            final_clusters = np.full(len(df), -1, dtype=int)
        
        return final_clusters
    
    def visualize_clusters(self, df, clusters):
        """Create comprehensive visualizations for the clusters"""
        processed_data = self.preprocess_data(df)
        
        # Create subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. PCA scatter plot (main plot)
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        self._create_pca_plot(ax1, processed_data, clusters)
        
        # 2. Cluster distribution
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        self._create_cluster_distribution(ax2, clusters)
        
        # 3. Feature importance
        ax3 = plt.subplot2grid((3, 3), (1, 2))
        self._create_feature_importance(ax3, processed_data)
        
        # 4. Outlier analysis
        ax4 = plt.subplot2grid((3, 3), (2, 0))
        self._create_outlier_analysis(ax4, processed_data, clusters)
        
        # 5. Feature distribution by cluster
        ax5 = plt.subplot2grid((3, 3), (2, 1), colspan=2)
        self._create_feature_distribution(ax5, df, clusters)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed cluster statistics
        self._print_cluster_statistics(df, clusters)
    
    def _create_pca_plot(self, ax, data, clusters):
        """Create PCA scatter plot"""
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        
        # Create scatter plot with different colors for outliers
        unique_clusters = np.unique(clusters)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster in enumerate(unique_clusters):
            mask = clusters == cluster
            if cluster == -1:
                # Outliers in red
                ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                          c='red', label='Outliers', s=100, alpha=0.7, marker='X')
            else:
                ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                          c=[colors[i]], label=f'Cluster {cluster}', s=80, alpha=0.7)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('Cluster Visualization (PCA)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_cluster_distribution(self, ax, clusters):
        """Create cluster distribution bar chart"""
        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
        colors = ['red' if cluster == -1 else 'lightblue' for cluster in unique_clusters]
        
        bars = ax.bar(unique_clusters, cluster_counts, color=colors, alpha=0.7)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Points')
        ax.set_title('Cluster Distribution')
        
        # Add value labels on bars
        for bar, count in zip(bars, cluster_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{count}', ha='center', va='bottom')
    
    def _create_feature_importance(self, ax, data):
        """Create feature importance plot"""
        pca = PCA(n_components=1)
        pca.fit(data)
        feature_importance = np.abs(pca.components_[0])
        
        ax.barh(self.feature_names, feature_importance, color='orange', alpha=0.7)
        ax.set_xlabel('Feature Importance (PCA Component 1)')
        ax.set_title('Feature Importance in Clustering')
    
    def _create_outlier_analysis(self, ax, data, clusters):
        """Create outlier analysis plot"""
        outlier_mask = clusters == -1
        normal_data = data[~outlier_mask]
        outlier_data = data[outlier_mask]
        
        # Plot first two features
        if len(data.columns) >= 2:
            ax.scatter(normal_data.iloc[:, 0], normal_data.iloc[:, 1], 
                      c='blue', label='Normal', alpha=0.6, s=60)
            if len(outlier_data) > 0:
                ax.scatter(outlier_data.iloc[:, 0], outlier_data.iloc[:, 1], 
                          c='red', label='Outliers', alpha=0.8, s=100, marker='X')
            
            ax.set_xlabel(data.columns[0])
            ax.set_ylabel(data.columns[1])
            ax.set_title('Outlier Detection\n(First Two Features)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _create_feature_distribution(self, ax, df, clusters):
        """Create feature distribution by cluster"""
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = clusters
        
        # Select first 3 numerical features for boxplot
        num_features = min(3, len(self.feature_names))
        feature_subset = self.feature_names[:num_features]
        
        if num_features > 0:
            df_melted = df_with_clusters.melt(id_vars=['cluster'], 
                                            value_vars=feature_subset,
                                            var_name='feature', 
                                            value_name='value')
            
            sns.boxplot(data=df_melted, x='cluster', y='value', hue='feature', ax=ax)
            ax.set_title('Feature Distribution by Cluster')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _print_cluster_statistics(self, df, clusters):
        """Print detailed cluster statistics"""
        print("\n" + "="*50)
        print("CLUSTER STATISTICS")
        print("="*50)
        
        unique_clusters = np.unique(clusters)
        
        for cluster in unique_clusters:
            cluster_mask = clusters == cluster
            cluster_size = np.sum(cluster_mask)
            percentage = (cluster_size / len(clusters)) * 100
            
            if cluster == -1:
                print(f"\n📊 OUTLIERS (Cluster -1):")
            else:
                print(f"\n📊 CLUSTER {cluster}:")
            
            print(f"   Size: {cluster_size} points ({percentage:.1f}%)")
            
            if cluster_size > 0:
                cluster_data = df.iloc[cluster_mask, 1:]  # Numerical data only
                print(f"   Feature means:")
                for feature in cluster_data.columns[:3]:  # Show first 3 features
                    mean_val = cluster_data[feature].mean()
                    print(f"     {feature}: {mean_val:.2f}")
