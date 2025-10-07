import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class CorrelationMatrixAnalyzer:
    def __init__(self, csv_file_path):
        """
        Initialize the correlation matrix analyzer
        
        Args:
            csv_file_path (str): Path to the correlation matrix CSV file
        """
        self.csv_file_path = csv_file_path
        self.corr_matrix = None
        self.sorted_corr_matrix = None
        
    def load_correlation_matrix(self):
        """
        Load correlation matrix from CSV file
        """
        try:
            self.corr_matrix = pd.read_csv(self.csv_file_path, index_col=0)
            print(f"Correlation matrix loaded successfully. Shape: {self.corr_matrix.shape}")
            print(f"Columns: {list(self.corr_matrix.columns)}")
            return True
        except Exception as e:
            print(f"Error loading correlation matrix: {e}")
            return False
    
    def calculate_column_ranking(self, method='absolute_mean'):
        """
        Calculate ranking for columns based on correlation strength
        
        Args:
            method (str): Ranking method - 'absolute_mean', 'variance', 'max_absolute'
        
        Returns:
            pd.Series: Column rankings
        """
        if self.corr_matrix is None:
            raise ValueError("Correlation matrix not loaded. Call load_correlation_matrix() first.")
        
        # Create a copy and set diagonal to NaN to avoid self-correlation bias
        corr_copy = self.corr_matrix.copy()
        np.fill_diagonal(corr_copy.values, np.nan)
        
        if method == 'absolute_mean':
            # Rank by mean of absolute correlations (overall correlation strength)
            rankings = corr_copy.abs().mean().sort_values(ascending=False)
        
        elif method == 'variance':
            # Rank by variance of correlations (how much correlation varies)
            rankings = corr_copy.var().sort_values(ascending=False)
        
        elif method == 'max_absolute':
            # Rank by maximum absolute correlation
            rankings = corr_copy.abs().max().sort_values(ascending=False)
        
        elif method == 'interaction_strength':
            # Rank by sum of absolute correlations (total interaction strength)
            rankings = corr_copy.abs().sum().sort_values(ascending=False)
        
        else:
            raise ValueError(f"Unknown ranking method: {method}")
        
        return rankings
    
    def sort_columns_by_ranking(self, method='absolute_mean'):
        """
        Sort correlation matrix columns by ranking
        
        Args:
            method (str): Ranking method
            
        Returns:
            pd.DataFrame: Sorted correlation matrix
        """
        rankings = self.calculate_column_ranking(method)
        
        # Sort both rows and columns by the same order
        self.sorted_corr_matrix = self.corr_matrix.loc[rankings.index, rankings.index]
        
        print(f"Columns sorted by {method} ranking:")
        for i, (col, score) in enumerate(rankings.items(), 1):
            print(f"  {i:2d}. {col}: {score:.4f}")
        
        return self.sorted_corr_matrix
    
    def get_highly_correlated_pairs(self, threshold=0.7, top_n=10):
        """
        Get highly correlated variable pairs
        
        Args:
            threshold (float): Correlation threshold
            top_n (int): Number of top pairs to return
            
        Returns:
            pd.DataFrame: Highly correlated pairs
        """
        if self.sorted_corr_matrix is None:
            self.sort_columns_by_ranking()
        
        corr_matrix = self.sorted_corr_matrix
        
        # Get upper triangle of correlation matrix
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Stack and sort by absolute correlation
        correlated_pairs = upper_triangle.stack().reset_index()
        correlated_pairs.columns = ['Variable_1', 'Variable_2', 'Correlation']
        correlated_pairs['Abs_Correlation'] = correlated_pairs['Correlation'].abs()
        
        # Filter by threshold and get top N
        high_corr_pairs = correlated_pairs[
            correlated_pairs['Abs_Correlation'] >= threshold
        ].sort_values('Abs_Correlation', ascending=False).head(top_n)
        
        return high_corr_pairs
    
    def export_sorted_matrix(self, output_file=None):
        """
        Export sorted correlation matrix to CSV
        
        Args:
            output_file (str): Output file path
        """
        if self.sorted_corr_matrix is None:
            self.sort_columns_by_ranking()
        
        if output_file is None:
            input_path = Path(self.csv_file_path)
            output_file = input_path.parent / f"sorted_{input_path.name}"
        
        self.sorted_corr_matrix.to_csv(output_file)
        print(f"Sorted correlation matrix exported to: {output_file}")
        
        return output_file
    
    def visualize_sorted_correlation(self, figsize=(12, 10), output_file=None):
        """
        Create a heatmap visualization of the sorted correlation matrix
        
        Args:
            figsize (tuple): Figure size
            output_file (str): Output file path for saving the plot
        """
        if self.sorted_corr_matrix is None:
            self.sort_columns_by_ranking()
        
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(self.sorted_corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(
            self.sorted_corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap='RdBu_r',
            center=0,
            square=True,
            cbar_kws={"shrink": .8}
        )
        
        plt.title('Sorted Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Correlation heatmap saved to: {output_file}")
        
        plt.show()
    
    def generate_ranking_report(self, output_prefix="correlation_ranking"):
        """
        Generate comprehensive ranking report
        
        Args:
            output_prefix (str): Prefix for output files
        """
        report_data = []
        
        # Compare different ranking methods
        methods = ['absolute_mean', 'variance', 'max_absolute', 'interaction_strength']
        
        for method in methods:
            rankings = self.calculate_column_ranking(method)
            
            for rank, (col, score) in enumerate(rankings.items(), 1):
                report_data.append({
                    'Metric': col,
                    'Ranking_Method': method,
                    'Rank': rank,
                    'Score': score
                })
        
        report_df = pd.DataFrame(report_data)
        
        # Pivot to show all rankings together
        pivot_df = report_df.pivot_table(
            index='Metric', 
            columns='Ranking_Method', 
            values='Rank'
        ).reset_index()
        
        # Calculate average rank
        pivot_df['Average_Rank'] = pivot_df[methods].mean(axis=1)
        pivot_df = pivot_df.sort_values('Average_Rank')
        
        # Export reports
        report_df.to_csv(f"{output_prefix}_detailed.csv", index=False)
        pivot_df.to_csv(f"{output_prefix}_summary.csv", index=False)
        
        print(f"Detailed ranking report exported to: {output_prefix}_detailed.csv")
        print(f"Summary ranking report exported to: {output_prefix}_summary.csv")
        
        return pivot_df

# Example usage and demonstration
def demonstrate_analyzer():
    """
    Demonstrate the correlation matrix analyzer with sample data
    """
    # Create sample correlation matrix for demonstration
    np.random.seed(42)
    
    # Generate sample metrics
    metrics = ['cpu_usage', 'memory_usage', 'disk_io', 'network_bytes', 
               'http_requests', 'response_time', 'error_rate', 'queue_length']
    
    # Create a realistic correlation matrix
    base_corr = np.random.uniform(-0.8, 0.8, (len(metrics), len(metrics)))
    corr_matrix = np.dot(base_corr, base_corr.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Create DataFrame
    sample_corr_df = pd.DataFrame(corr_matrix, index=metrics, columns=metrics)
    sample_corr_df.to_csv('sample_correlation_matrix.csv')
    
    print("Sample correlation matrix created: sample_correlation_matrix.csv")
    
    # Initialize analyzer
    analyzer = CorrelationMatrixAnalyzer('sample_correlation_matrix.csv')
    
    if analyzer.load_correlation_matrix():
        print("\n" + "="*60)
        print("ORIGINAL CORRELATION MATRIX")
        print("="*60)
        print(analyzer.corr_matrix.round(3))
        
        # Demonstrate different ranking methods
        methods = ['absolute_mean', 'variance', 'max_absolute', 'interaction_strength']
        
        for method in methods:
            print(f"\n" + "="*60)
            print(f"SORTED BY {method.upper()} RANKING")
            print("="*60)
            
            sorted_matrix = analyzer.sort_columns_by_ranking(method)
            print(f"\nSorted Matrix (first 5x5):")
            print(sorted_matrix.iloc[:5, :5].round(3))
        
        # Get highly correlated pairs
        print(f"\n" + "="*60)
        print("HIGHLY CORRELATED PAIRS (threshold: 0.7)")
        print("="*60)
        high_corr_pairs = analyzer.get_highly_correlated_pairs(threshold=0.7)
        if not high_corr_pairs.empty:
            print(high_corr_pairs.round(3))
        else:
            print("No highly correlated pairs found above threshold")
        
        # Export sorted matrix
        analyzer.export_sorted_matrix()
        
        # Generate comprehensive ranking report
        analyzer.generate_ranking_report()
        
        # Create visualization
        analyzer.visualize_sorted_correlation(output_file='correlation_heatmap.png')
        
        return analyzer

def main():
    """
    Main function to run the correlation matrix analyzer
    """
    # For demonstration with sample data
    analyzer = demonstrate_analyzer()
    
    # For your actual data, use:
    # analyzer = CorrelationMatrixAnalyzer('your_correlation_matrix.csv')
    # analyzer.load_correlation_matrix()
    # analyzer.sort_columns_by_ranking('absolute_mean')
    # analyzer.export_sorted_matrix()

if __name__ == "__main__":
    main()