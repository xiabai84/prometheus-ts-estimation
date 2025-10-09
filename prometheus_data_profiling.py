import pandas as pd
import numpy as np
import warnings
from generate_sample_data import generate_sample_prometheus_data
warnings.filterwarnings('ignore')

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
    
    def generate_comprehensive_report(self):
        """
        Generate comprehensive data profiling report
        
        Returns:
            dict: Dictionary containing all report components
        """
        print("Generating comprehensive data profiling report...")
        
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


if __name__ == "__main__":
    # Generate sample data (comment out if using real data)
    # csv_file = generate_sample_prometheus_data()
    csv_file = "file-name.csv"
    
    # Initialize profiler
    profiler = PrometheusDataProfiler(csv_file)
    
    try:
        # Load data
        profiler.load_data()
        
        # Generate comprehensive report
        report = profiler.generate_comprehensive_report()
        
        # Print summary to console
        profiler.print_report_summary(report)
        
        # Export to CSV files
        profiler.export_report_to_csv(report, "prometheus_data_profile")
        
        print("\n" + "="*80)
        print("REPORT GENERATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        df = pd.read_csv("prometheus_data_profile_metric_profiles.csv")
        sorted_column = "p25"
        threshold_sec = 0.5

        filtered_report = df.loc[df[sorted_column] >= threshold_sec]
        sorted_report = filtered_report.sort_values(by[sorted_column], ascending=False)
        print(sorted_report)
        sorted_report.to_csv("z_filtered_metric_profile.csv", index=False)

        affected_microflows = set()
        for row in sorted_report.metric_name.values:
            category = row.split(".")[0]
            affected_microflows.add(category)

        print("\n" + "="*80)
        print(f"AFFECTED MICROFLOWS FROM TOTAL {len(profiler.numeric_columns)} SORT BY {sorted_column}")
        print("="*80)

        for microflow in affected_microflows:
            print(microflow)

    except Exception as e:
        print(f"Error during profiling: {e}")
    