import pandas as pd
import numpy as np
from pathlib import Path

class ColumnSorterByMax:
    def __init__(self, df=None, csv_file=None):
        """
        Initialize the column sorter
        
        Args:
            df (pd.DataFrame): Input DataFrame
            csv_file (str): Path to CSV file (alternative to df)
        """
        if df is not None:
            self.df = df.copy()
        elif csv_file is not None:
            self.df = self.load_from_csv(csv_file)
        else:
            self.df = None
        
        self.sorted_df = None
        self.column_max_values = None
    
    def load_from_csv(self, csv_file):
        """Load DataFrame from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            print(f"Data loaded successfully from {csv_file}. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise
    
    def calculate_column_maxes(self):
        """Calculate maximum values for each column"""
        if self.df is None:
            raise ValueError("No DataFrame loaded. Please provide df or csv_file.")
        
        # Calculate max values, ignoring NaN
        self.column_max_values = self.df.max(axis=0, skipna=True)
        
        print("Column maximum values:")
        for col, max_val in self.column_max_values.items():
            print(f"  {col}: {max_val}")
        
        return self.column_max_values
    
    def sort_columns_by_max(self, ascending=False, exclude_columns=None):
        """
        Sort DataFrame columns by their maximum values
        
        Args:
            ascending (bool): Sort order (False = descending by max values)
            exclude_columns (list): Columns to exclude from sorting
            
        Returns:
            pd.DataFrame: DataFrame with sorted columns
        """
        if self.column_max_values is None:
            self.calculate_column_maxes()
        
        # Identify columns to sort
        all_columns = self.df.columns.tolist()
        
        if exclude_columns:
            sort_columns = [col for col in all_columns if col not in exclude_columns]
            fixed_columns = exclude_columns
        else:
            sort_columns = all_columns
            fixed_columns = []
        
        # Sort columns by their max values
        sorted_columns = sorted(
            sort_columns, 
            key=lambda x: self.column_max_values[x], 
            reverse=not ascending
        )
        
        # Combine fixed columns with sorted columns
        final_columns = fixed_columns + sorted_columns
        
        # Create sorted DataFrame
        self.sorted_df = self.df[final_columns]
        
        print(f"\nColumns sorted by maximum values ({'descending' if not ascending else 'ascending'}):")
        for i, col in enumerate(final_columns, 1):
            max_val = self.column_max_values.get(col, 'N/A')
            print(f"  {i:2d}. {col}: {max_val}")
        
        return self.sorted_df
    
    def sort_columns_by_abs_max(self, ascending=False):
        """
        Sort columns by absolute maximum values (considers both positive and negative extremes)
        
        Args:
            ascending (bool): Sort order
            
        Returns:
            pd.DataFrame: DataFrame with sorted columns
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded.")
        
        # Calculate absolute maximum values
        abs_max_values = self.df.abs().max(axis=0, skipna=True)
        
        # Sort columns by absolute max values
        sorted_columns = sorted(
            self.df.columns, 
            key=lambda x: abs_max_values[x], 
            reverse=not ascending
        )
        
        self.sorted_df = self.df[sorted_columns]
        self.column_max_values = abs_max_values
        
        print(f"\nColumns sorted by absolute maximum values ({'descending' if not ascending else 'ascending'}):")
        for i, col in enumerate(sorted_columns, 1):
            abs_max_val = abs_max_values[col]
            print(f"  {i:2d}. {col}: {abs_max_val}")
        
        return self.sorted_df
    
    def get_top_columns_by_max(self, top_n=10, ascending=False):
        """
        Get top N columns with highest maximum values
        
        Args:
            top_n (int): Number of top columns to return
            ascending (bool): Sort order
            
        Returns:
            list: List of top column names
        """
        if self.column_max_values is None:
            self.calculate_column_maxes()
        
        sorted_columns = sorted(
            self.column_max_values.items(),
            key=lambda x: x[1],
            reverse=not ascending
        )[:top_n]
        
        print(f"\nTop {top_n} columns by maximum value:")
        for i, (col, max_val) in enumerate(sorted_columns, 1):
            print(f"  {i:2d}. {col}: {max_val}")
        
        return [col for col, _ in sorted_columns]
    
    def export_sorted_dataframe(self, output_file=None, include_max_info=True):
        """
        Export sorted DataFrame to CSV
        
        Args:
            output_file (str): Output file path
            include_max_info (bool): Whether to include max values in separate file
        """
        if self.sorted_df is None:
            self.sort_columns_by_max()
        
        if output_file is None:
            output_file = "sorted_dataframe_by_max.csv"
        
        # Export sorted DataFrame
        self.sorted_df.to_csv(output_file, index=False)
        print(f"Sorted DataFrame exported to: {output_file}")
        
        # Export max values information
        if include_max_info and self.column_max_values is not None:
            max_info_file = Path(output_file).stem + "_max_values.csv"
            max_df = pd.DataFrame({
                'column': self.column_max_values.index,
                'max_value': self.column_max_values.values
            }).sort_values('max_value', ascending=False)
            
            max_df.to_csv(max_info_file, index=False)
            print(f"Column maximum values exported to: {max_info_file}")
        
        return output_file
    
    def visualize_max_values(self, top_n=15, figsize=(12, 6)):
        """
        Create visualization of column maximum values
        
        Args:
            top_n (int): Number of top columns to show
            figsize (tuple): Figure size
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.column_max_values is None:
            self.calculate_column_maxes()
        
        # Get top N columns
        top_columns = self.get_top_columns_by_max(top_n=top_n, ascending=False)
        top_max_values = [self.column_max_values[col] for col in top_columns]
        
        plt.figure(figsize=figsize)
        
        # Create bar plot
        bars = plt.bar(range(len(top_columns)), top_max_values, color='skyblue', alpha=0.7)
        plt.xlabel('Columns')
        plt.ylabel('Maximum Values')
        plt.title(f'Top {top_n} Columns by Maximum Values')
        plt.xticks(range(len(top_columns)), top_columns, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, top_max_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(top_max_values),
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()

# Example usage and demonstration
def create_sample_dataframe():
    """Create a sample DataFrame for demonstration"""
    np.random.seed(42)
    
    data = {
        'low_range_col': np.random.uniform(0, 10, 100),
        'medium_range_col': np.random.uniform(0, 50, 100),
        'high_range_col': np.random.uniform(0, 100, 100),
        'very_high_col': np.random.uniform(0, 200, 100),
        'negative_col': np.random.uniform(-50, 50, 100),
        'sparse_col': np.concatenate([np.random.uniform(0, 5, 95), np.random.uniform(100, 200, 5)]),
        'binary_col': np.random.choice([0, 1], 100),
        'constant_low': np.ones(100) * 5,
        'outlier_col': np.concatenate([np.random.uniform(0, 10, 99), [1000]])
    }
    
    df = pd.DataFrame(data)
    return df

def demonstrate_column_sorter():
    """Demonstrate the column sorter functionality"""
    print("Creating sample DataFrame...")
    df = create_sample_dataframe()
    
    print("\nOriginal DataFrame info:")
    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Initialize sorter
    sorter = ColumnSorterByMax(df=df)
    
    # Calculate max values
    max_values = sorter.calculate_column_maxes()
    
    print("\n" + "="*60)
    print("SORTING BY MAXIMUM VALUES (DESCENDING)")
    print("="*60)
    
    # Sort columns by max values (descending - default)
    sorted_df_desc = sorter.sort_columns_by_max(ascending=False)
    print(f"\nSorted DataFrame shape: {sorted_df_desc.shape}")
    print("Sorted columns:", list(sorted_df_desc.columns))
    
    print("\n" + "="*60)
    print("SORTING BY MAXIMUM VALUES (ASCENDING)")
    print("="*60)
    
    # Sort columns by max values (ascending)
    sorted_df_asc = sorter.sort_columns_by_max(ascending=True)
    print("Sorted columns (ascending):", list(sorted_df_asc.columns))
    
    print("\n" + "="*60)
    print("SORTING BY ABSOLUTE MAXIMUM VALUES")
    print("="*60)
    
    # Sort by absolute max values
    abs_sorted_df = sorter.sort_columns_by_abs_max(ascending=False)
    print("Columns sorted by absolute max:", list(abs_sorted_df.columns))
    
    print("\n" + "="*60)
    print("TOP COLUMNS ANALYSIS")
    print("="*60)
    
    # Get top columns
    top_columns = sorter.get_top_columns_by_max(top_n=5)
    print(f"Top 5 columns: {top_columns}")
    
    # Export results
    sorter.export_sorted_dataframe("sample_sorted_by_max.csv")
    
    # Visualize
    sorter.visualize_max_values(top_n=8)
    
    return sorter

def process_csv_file(csv_file_path):
    """Process an actual CSV file"""
    print(f"Processing CSV file: {csv_file_path}")
    
    sorter = ColumnSorterByMax(csv_file=csv_file_path)
    
    # Sort by max values (descending)
    sorted_df = sorter.sort_columns_by_max(ascending=False)
    
    # Export results
    output_file = sorter.export_sorted_dataframe()
    
    # Show top columns
    sorter.get_top_columns_by_max(top_n=10)
    
    return sorter, sorted_df

# Main execution
if __name__ == "__main__":
    # Demonstration with sample data
    print("DEMONSTRATION WITH SAMPLE DATA")
    print("="*60)
    sorter = demonstrate_column_sorter()
    
    # Example for actual CSV file (uncomment to use)
    # sorter, sorted_df = process_csv_file("your_data.csv")