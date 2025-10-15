import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class PrometheusDataProfiler:
    def __init__(self, csv_file_path):
        """
        Initialize the profiler with Prometheus CSV data
        
        Args:
            csv_file_path (str): Path to the Prometheus CSV file
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.numeric_columns = []
        
    def load_data(self):
        """
        Load and preprocess Prometheus CSV data
        """
        try:
            # Read CSV file
            self.df = pd.read_csv(self.csv_file_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            
            # Identify timestamp column (usually first column)
            timestamp_col = self.df.columns[0]
            
            # Convert timestamp to datetime if it's numeric
            if pd.api.types.is_numeric_dtype(self.df[timestamp_col]):
                self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col], unit='s')
            else:
                self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
            
            # Set timestamp as index
            self.df.set_index(timestamp_col, inplace=True)
            # Identify numeric columns (excluding timestamp)
            self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            print(f"Found {len(self.numeric_columns)} numeric metrics")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def filter_non_zero(self, series):
        """
        Filter out zero values from a series
        
        Args:
            series (pd.Series): Input series
            
        Returns:
            pd.Series: Series with zero values removed
        """
        return series[series != 0]
    
    def generate_metric_profile(self, metric_name):
        """
        Generate profiling statistics for a single metric
        
        Args:
            metric_name (str): Name of the metric column
            
        Returns:
            dict: Dictionary containing profiling statistics
        """
        if metric_name not in self.numeric_columns:
            return None
            
        series = self.df[metric_name].dropna()
        non_zero_series = self.filter_non_zero(series)
        
        if len(non_zero_series) == 0:
            return None
            
        profile = {
            'metric_name': metric_name,
            'mean': non_zero_series.mean(),
            'median': non_zero_series.median(),
            'std_dev': non_zero_series.std(),
            'skewness': non_zero_series.skew(),
            'kurtosis': non_zero_series.kurtosis(),
            'min': non_zero_series.min(),
            'max': non_zero_series.max(),
            'p05': np.percentile(non_zero_series, 5, method="linear"),
            'p25': np.percentile(non_zero_series, 25, method="linear"),
            'p75': np.percentile(non_zero_series, 75, method="linear"),
            'p95': np.percentile(non_zero_series, 95, method="linear"),
            'iqr': non_zero_series.quantile(0.75) - non_zero_series.quantile(0.25),
            'null_count': self.df[metric_name].isnull().sum(),
            'null_percentage': (self.df[metric_name].isnull().sum() / len(self.df)) * 100,
            'non_zero_records': len(non_zero_series),
            'zero_count': len(series) - len(non_zero_series),
            'zero_percentage': ((len(series) - len(non_zero_series)) / len(series)) * 100,
            'total_records': len(series)
        }
        
        return profile
    
    def generate_correlation_matrix(self):
        """
        Generate correlation matrix for non-zero values
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Create a copy with non-zero values
        non_zero_df = self.df[self.numeric_columns].copy()
        
        # Replace zeros with NaN for correlation calculation
        for col in self.numeric_columns:
            non_zero_df[col] = non_zero_df[col].replace(0, np.nan)
        
        return non_zero_df.corr()
    
    def generate_profiling_report(self):
        """
        Generate comprehensive data profiling report
        
        Returns:
            dict: Dictionary containing all report components
        """
        report = {
            'overview': {},
            'metric_profiles': [],
            'correlation_matrix': None,
            'summary_statistics': {}
        }
        
        # Overview statistics
        report['overview'] = {
            'total_metrics': len(self.numeric_columns),
            'total_timestamps': len(self.df),
            'data_range': {
                'start': self.df.index.min(),
                'end': self.df.index.max()
            },
            'time_span_days': (self.df.index.max() - self.df.index.min()).days
        }
        
        # Individual metric profiles
        for metric in self.numeric_columns:
            profile = self.generate_metric_profile(metric)
            if profile:
                report['metric_profiles'].append(profile)
        
        # Correlation matrix
        if len(self.numeric_columns) > 1:
            report['correlation_matrix'] = self.generate_correlation_matrix()
        
        # Summary statistics
        total_metrics = len(report['metric_profiles'])
        metrics_with_zeros = sum(1 for p in report['metric_profiles'] if p['zero_count'] > 0)
        
        report['summary_statistics'] = {
            'total_metrics_analyzed': total_metrics,
            'metrics_with_zeros': metrics_with_zeros,
            'avg_zero_percentage': np.mean([p['zero_percentage'] for p in report['metric_profiles']]),
            'avg_null_percentage': np.mean([p['null_percentage'] for p in report['metric_profiles']])
        }
        print("\n" + "="*80)
        print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*80)
        return report
    
    def export_report_to_csv(self, report, output_prefix="prometheus_profile"):
        """
        Export the profiling report to CSV files
        
        Args:
            report (dict): The generated report
            output_prefix (str): Prefix for output files
        """
        # Export metric profiles
        if report['metric_profiles']:
            profiles_df = pd.DataFrame(report['metric_profiles'])
            profiles_df.to_csv(f"{output_prefix}_metric_profiles.csv", index=False)
            print(f"Metric profiles exported to {output_prefix}_metric_profiles.csv")
        
        # Export correlation matrix
        if report['correlation_matrix'] is not None:
            report['correlation_matrix'].to_csv(f"{output_prefix}_correlations.csv")
            print(f"Correlation matrix exported to {output_prefix}_correlations.csv")
        
        # Export summary statistics
        summary_df = pd.DataFrame([report['summary_statistics']])
        summary_df.to_csv(f"{output_prefix}_summary.csv", index=False)
        print(f"Summary statistics exported to {output_prefix}_summary.csv")
        
        # Export overview
        overview_df = pd.DataFrame([report['overview']])
        overview_df.to_csv(f"{output_prefix}_overview.csv", index=False)
        print(f"Overview exported to {output_prefix}_overview.csv")
    
    def print_report_summary(self, report):
        """
        Print a summary of the report to console
        """
        # print("\n" + "="*80)
        # print("PROMETHEUS DATA PROFILING REPORT SUMMARY")
        # print("="*80)
        
        # overview = report['overview']
        # summary = report['summary_statistics']
        
        # print(f"\nOVERVIEW:")
        # print(f"  Time Range: {overview['data_range']['start']} to {overview['data_range']['end']}")
        # print(f"  Total Timestamps: {overview['total_timestamps']:,}")
        # print(f"  Time Span: {overview['time_span_days']} days")
        # print(f"  Metrics Analyzed: {overview['total_metrics']}")
        
        # print(f"\nSUMMARY STATISTICS:")
        # print(f"  Metrics with zero values: {summary['metrics_with_zeros']}/{summary['total_metrics_analyzed']}")
        # print(f"  Average zero percentage: {summary['avg_zero_percentage']:.2f}%")
        # print(f"  Average null percentage: {summary['avg_null_percentage']:.2f}%")
        
        # print(f"\nTOP METRICS BY ZERO PERCENTAGE:")
        # profiles_sorted = sorted(report['metric_profiles'], 
        #                        key=lambda x: x['zero_percentage'], 
        #                        reverse=True)[:5]
        
        # for profile in profiles_sorted:
        #     print(f"  {profile['metric_name']}: {profile['zero_percentage']:.2f}% zeros "
        #           f"({profile['zero_count']:,} out of {profile['total_records']:,})")

    def interpret_skewness(skew_value):
        if abs(skew_value) < 0.5:
            return "Fairly Symmetric"
        elif abs(skew_value) < 1:
            return "Moderately Skewed"
        else:
            return "Highly Skewed"
        
    def interpret_kurtosis(kurt_value):
        if kurt_value < -1:
            return "Platykurtic (Flat)"
        elif kurt_value > 1:
            return "Leptokurtic (Peaked)"
        else:
            return "Mesokurtic (Normal)"
        
    def virtualize_skewness_kurtosis(self, columns: list, figsize=(15,10), save_path=None):
        """
        Visualize skewness and kurtosis for each numerical column in a Dataframe.
        """
        dataframe = self.df[columns].copy()
        numerical_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        n_cols = len(numerical_cols)
        if n_cols == 0:
            print("No numeriacal columns found in the Dataframe")
            return None, None
        
        n_rows = int(np.ceil(n_cols / 3))
        n_subplot_cols = min(3, n_cols)
        fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=figsize, constrained_layout=True)
        axes = axes.flatten() if n_cols > 1 else [axes]
        skewness_values = dataframe[numerical_cols].skew()
        kurtosis_values = dataframe[numerical_cols].kurtosis()

        for i, col in enumerate(numerical_cols):
            ax = axes[i]
            data_col = dataframe[col].dropna()
            median_val = dataframe[col].median()
            p75 = np.percentile(dataframe[col], 75, method="linear")
            ax.hist(data_col, bins=50, alpha=0.7, density=True, color='lightblue', edgecolor='black', linewidth=0.5)
            if len(data_col) > 1:
                kde_x = np.linspace(data_col.min(), data_col.max(), 100)
                kde = stats.gaussian_kde(data_col)
                ax.plot(kde_x, kde(kde_x), color='red', linewidth=2, label='KDE')
                ax.axvline(median_val, label="Median")
                ax.axvline(p75, label="P75")
            skew_val = skewness_values[col]
            kurt_val = kurtosis_values[col]
            ax.set_title(col, fontsize=5, fontweight='bold', pad=15)
            ax.set_xlabel('Value', fontsize=4)
            ax.set_ylabel('Density', fontsize=4)
            stats_text = f'Skewness: {skew_val:.3f}\nKurtosis: {kurt_val:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), fontsize=5)
            ax.legend()
            ax.grid(True, alpha=0.3)
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        return fig, axes
