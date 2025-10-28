import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from datetime import datetime

class MetricComparator:
    def __init__(self):
        self.comparison_results = {}
        self.analysis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def prepare_data(self, normal_df, fault_df, timestamp_col='timestamp', value_col='value', 
                   metric_col=None, min_data_points=5, handle_zeros='drop', handle_nulls='drop'):
        """
        Prepare data format with enhanced data quality controls
        
        Parameters:
        -----------
        normal_df : DataFrame
            Normal period data
        fault_df : DataFrame
            Fault period data  
        timestamp_col : str, default='timestamp'
            Name of timestamp column
        value_col : str, default='value'
            Name of value column
        metric_col : str, optional
            Name of metric column. If None, treats all data as single metric
        min_data_points : int, default=5
            Minimum number of data points required for each metric
        handle_zeros : str, default='keep'
            How to handle zero values: 'keep', 'drop', or 'replace'
        handle_nulls : str, default='drop'
            How to handle null values: 'drop', 'fill_mean', 'fill_median'
        """
        self.normal_df = normal_df.copy()
        self.fault_df = fault_df.copy()
        self.timestamp_col = timestamp_col
        self.value_col = value_col
        self.metric_col = metric_col
        self.min_data_points = min_data_points
        
        print(f"Initial data shapes - Normal: {self.normal_df.shape}, Fault: {self.fault_df.shape}")
        
        # Validate required columns exist
        self._validate_columns()
        
        # Ensure timestamp format if timestamp column exists
        if timestamp_col and timestamp_col in self.normal_df.columns:
            self.normal_df[timestamp_col] = pd.to_datetime(self.normal_df[timestamp_col])
            if timestamp_col in self.fault_df.columns:
                self.fault_df[timestamp_col] = pd.to_datetime(self.fault_df[timestamp_col])
        
        # Handle data quality issues
        self._handle_data_quality(handle_nulls, handle_zeros)
        
        # Process metric column and identify metrics
        self._process_metric_column()
        
        # Filter metrics by data point count
        self._filter_metrics_by_data_points()
        
        return self.normal_df, self.fault_df
    
    def _validate_columns(self):
        """Validate that required columns exist in both DataFrames"""
        if self.value_col not in self.normal_df.columns:
            raise ValueError(f"Value column '{self.value_col}' not found in normal DataFrame. Available columns: {list(self.normal_df.columns)}")
        if self.value_col not in self.fault_df.columns:
            raise ValueError(f"Value column '{self.value_col}' not found in fault DataFrame. Available columns: {list(self.fault_df.columns)}")
        
        if self.metric_col is not None:
            if self.metric_col not in self.normal_df.columns:
                print(f"Warning: Metric column '{self.metric_col}' not found in normal DataFrame. Available columns: {list(self.normal_df.columns)}")
            if self.metric_col not in self.fault_df.columns:
                print(f"Warning: Metric column '{self.metric_col}' not found in fault DataFrame. Available columns: {list(self.fault_df.columns)}")
    
    def _handle_data_quality(self, handle_nulls, handle_zeros):
        """Handle null values and zero values based on specified strategies"""
        
        # Handle null values
        for df_name, df in [('normal', self.normal_df), ('fault', self.fault_df)]:
            initial_count = len(df)
            null_count = df[self.value_col].isnull().sum()
            
            if handle_nulls == 'drop':
                df.dropna(subset=[self.value_col], inplace=True)
            elif handle_nulls == 'fill_mean':
                mean_val = df[self.value_col].mean()
                df[self.value_col].fillna(mean_val, inplace=True)
            elif handle_nulls == 'fill_median':
                median_val = df[self.value_col].median()
                df[self.value_col].fillna(median_val, inplace=True)
            
            final_count = len(df)
            if null_count > 0:
                print(f"{df_name.capitalize()} data: {null_count} null values handled ({handle_nulls}), "
                      f"records: {initial_count} -> {final_count}")
        
        # Handle zero values
        for df_name, df in [('normal', self.normal_df), ('fault', self.fault_df)]:
            initial_count = len(df)
            zero_count = (df[self.value_col] == 0).sum()
            
            if handle_zeros == 'drop':
                df = df[df[self.value_col] != 0]
            elif handle_zeros == 'replace':
                # Replace zeros with a small positive value (1% of mean)
                non_zero_data = df[df[self.value_col] != 0][self.value_col]
                if len(non_zero_data) > 0:
                    mean_val = non_zero_data.mean()
                    replacement = mean_val * 0.01 if mean_val > 0 else 0.01
                    df[self.value_col] = df[self.value_col].replace(0, replacement)
            
            final_count = len(df)
            if zero_count > 0:
                print(f"{df_name.capitalize()} data: {zero_count} zero values handled ({handle_zeros}), "
                      f"records: {initial_count} -> {final_count}")
    
    def _process_metric_column(self):
        """Process metric column and identify common metrics, show metrics only in fault data"""
        if self.metric_col is None:
            self.common_metrics = ['single_metric']
            self.normal_df['_single_metric'] = 'single_metric'
            self.fault_df['_single_metric'] = 'single_metric'
            self.metric_col = '_single_metric'
            self.metrics_only_in_normal = []
            self.metrics_only_in_fault = []
            print("No metric column specified - treating all data as single metric")
            
        elif self.metric_col not in self.normal_df.columns or self.metric_col not in self.fault_df.columns:
            print(f"Warning: Metric column '{self.metric_col}' missing in one or both DataFrames")
            print("Falling back to single metric mode")
            self.common_metrics = ['single_metric']
            self.normal_df['_single_metric'] = 'single_metric'
            self.fault_df['_single_metric'] = 'single_metric'
            self.metric_col = '_single_metric'
            self.metrics_only_in_normal = []
            self.metrics_only_in_fault = []
            
        else:
            # Metric column exists in both DataFrames
            normal_metrics = set(self.normal_df[self.metric_col].unique())
            fault_metrics = set(self.fault_df[self.metric_col].unique())
            
            self.common_metrics = list(normal_metrics.intersection(fault_metrics))
            self.metrics_only_in_normal = list(normal_metrics - fault_metrics)
            self.metrics_only_in_fault = list(fault_metrics - normal_metrics)
            
            print(f"\n=== Metric Analysis ===")
            print(f"Common metrics (will be compared): {len(self.common_metrics)}")
            print(f"Metrics only in normal data: {len(self.metrics_only_in_normal)}")
            print(f"Metrics only in fault data: {len(self.metrics_only_in_fault)}")
            
            # Show metrics only in fault data (potential new issues)
            if self.metrics_only_in_fault:
                print(f"\n⚠️  Metrics found ONLY in fault data (potential new issues):")
                for i, metric in enumerate(self.metrics_only_in_fault[:10]):
                    fault_count = len(self.fault_df[self.fault_df[self.metric_col] == metric])
                    print(f"  {i+1}. {metric} (data points: {fault_count})")
                if len(self.metrics_only_in_fault) > 10:
                    print(f"  ... and {len(self.metrics_only_in_fault) - 10} more")
            
            # Show metrics only in normal data (potential missing metrics)
            if self.metrics_only_in_normal:
                print(f"\n📊 Metrics found ONLY in normal data (missing in fault data):")
                for i, metric in enumerate(self.metrics_only_in_normal[:5]):  # Show first 5
                    normal_count = len(self.normal_df[self.normal_df[self.metric_col] == metric])
                    print(f"  {i+1}. {metric} (data points: {normal_count})")
                if len(self.metrics_only_in_normal) > 5:
                    print(f"  ... and {len(self.metrics_only_in_normal) - 5} more")

            if len(self.common_metrics) == 0:
                print("\n❌ Warning: No common metrics between normal and fault datasets!")
                print("Available metrics in normal data:", list(normal_metrics)[:5])
                print("Available metrics in fault data:", list(fault_metrics)[:5])
            else:
                print(f"\n✅ Will compare {len(self.common_metrics)} common metrics")
    
    def _filter_metrics_by_data_points(self):
        """Filter metrics based on minimum data point requirements"""
        if self.metric_col is None:
            return # Single metric mode, no filtering needed
        
        filtered_metrics = []
        
        for metric in self.common_metrics:
            normal_count = len(self.normal_df[self.normal_df[self.metric_col] == metric])
            fault_count = len(self.fault_df[self.fault_df[self.metric_col] == metric])
            
            if normal_count >= self.min_data_points and fault_count >= self.min_data_points:
                filtered_metrics.append(metric)
            else:
                print(f"Filtered out metric '{metric}': insufficient data (normal: {normal_count}, fault: {fault_count}, required: {self.min_data_points})")
        
        initial_count = len(self.common_metrics)
        self.common_metrics = filtered_metrics
        final_count = len(self.common_metrics)
        
        if final_count < initial_count:
            print(f"\n📉 Metrics filtered: {initial_count} -> {final_count} (min_data_points={self.min_data_points})")
        
        if final_count == 0:
            print("❌ No metrics meet the minimum data point requirements!")
    
    def get_metric_info(self):
        """Get comprehensive information about metrics in the datasets"""
        info = {
            'metric_column': self.metric_col,
            'common_metrics_count': len(self.common_metrics),
            'metrics_only_in_normal_count': len(self.metrics_only_in_normal),
            'metrics_only_in_fault_count': len(self.metrics_only_in_fault),
            'min_data_points': self.min_data_points,
            'normal_data_shape': self.normal_df.shape,
            'fault_data_shape': self.fault_df.shape,
            'value_column': self.value_col
        }
        
        if self.metric_col and self.metric_col in self.normal_df.columns:
            info['normal_metrics_count'] = self.normal_df[self.metric_col].nunique()
            info['fault_metrics_count'] = self.fault_df[self.metric_col].nunique()
            info['common_metrics_sample'] = self.common_metrics[:5] if self.common_metrics else []
            info['fault_only_metrics_sample'] = self.metrics_only_in_fault[:5] if self.metrics_only_in_fault else []
        
        return info
    
    def get_fault_only_metrics_details(self, top_n=10):
        """Get detailed information about metrics that only exist in fault data"""
        if not self.metrics_only_in_fault:
            return pd.DataFrame()
        
        details = []
        for metric in self.metrics_only_in_fault[:top_n]:
            fault_data = self.fault_df[self.fault_df[self.metric_col] == metric][self.value_col]
            details.append({
                'metric': metric,
                'data_points': len(fault_data),
                'mean_value': fault_data.mean(),
                'std_value': fault_data.std(),
                'min_value': fault_data.min(),
                'max_value': fault_data.max()
            })
        
        return pd.DataFrame(details)
    
    def _get_metric_data(self, df, metric):
        """Helper method to get metric data safely"""
        if self.metric_col in df.columns:
            return df[df[self.metric_col] == metric][self.value_col]
        else:
            return df[self.value_col]

    def basic_statistical_comparison(self, min_data_points=None):
        """Basic statistical comparison with enhanced data validation"""
        if min_data_points is None:
            min_data_points = self.min_data_points
        
        results = []
        metrics_skipped = 0
        
        print(f"\n=== Starting Statistical Comparison ===")
        print(f"Comparing {len(self.common_metrics)} metrics with min_data_points={min_data_points}")
        
        for metric in self.common_metrics:
            normal_data = self._get_metric_data(self.normal_df, metric)
            fault_data = self._get_metric_data(self.fault_df, metric)
            
            normal_data = normal_data.dropna()
            fault_data = fault_data.dropna()
            
            if len(normal_data) < min_data_points or len(fault_data) < min_data_points:
                metrics_skipped += 1
                continue
            
            if normal_data.std() == 0 and fault_data.std() == 0:
                metrics_skipped += 1
                continue
            
            # Basic statistics
            normal_mean = np.mean(normal_data)
            fault_mean = np.mean(fault_data)
            mean_change = fault_mean - normal_mean
            mean_change_pct = (mean_change / (abs(normal_mean) + 1e-10) * 100)
            
            normal_std = np.std(normal_data)
            fault_std = np.std(fault_data)
            std_change_pct = ((fault_std - normal_std) / (normal_std + 1e-10) * 100) if normal_std != 0 else 0
            
            # Percentile statistic handle zero and infinite cases
            normal_p75 = np.percentile(normal_data, 75)
            fault_p75 = np.percentile(fault_data, 75)
            p75_change = fault_p75 - normal_p75
            
            # Fix for infinite values: handle zero normal_p75 case
            if abs(normal_p75) < 1e-10:  # If normal_p75 is effectively zero
                if abs(fault_p75) < 1e-10:  # Both are zero
                    p75_change_pct = 0.0
                else:  # Normal is zero, fault is not zero
                    # Use a large but finite value to indicate significant change
                    p75_change_pct = 1000.0 if fault_p75 > 0 else -1000.0
            else:
                p75_change_pct = (p75_change / abs(normal_p75)) * 100
            
            # Additional safeguard: cap extreme values
            if abs(p75_change_pct) > 1e6:  # If still extremely large
                p75_change_pct = 1000.0 if p75_change_pct > 0 else -1000.0
            
            # Extreme values
            normal_max = np.max(normal_data)
            fault_max = np.max(fault_data)
            normal_min = np.min(normal_data)
            fault_min = np.min(fault_data)
            
            # Statistical significance test
            try:
                t_stat, p_value = stats.ttest_ind(normal_data, fault_data, equal_var=False, nan_policy='omit')
            except Exception as e:
                t_stat, p_value = 0, 1.0
            
            # Effect size
            pooled_std = np.sqrt((normal_std**2 + fault_std**2) / 2)
            cohens_d = abs(mean_change) / (pooled_std + 1e-10)
            
            result = {
                'metric': metric,
                'normal_data_points': len(normal_data),
                'fault_data_points': len(fault_data),
                'total_data_points': len(normal_data) + len(fault_data),
                'normal_mean': normal_mean,
                'fault_mean': fault_mean,
                'mean_absolute_change': mean_change,
                'mean_change_percent': mean_change_pct,
                'normal_std': normal_std,
                'fault_std': fault_std,
                'std_change_percent': std_change_pct,
                'normal_p75': normal_p75,
                'fault_p75': fault_p75,
                'p75_absolute_change': p75_change,
                'p75_change_percent': p75_change_pct,
                'normal_max': normal_max,
                'fault_max': fault_max,
                'normal_min': normal_min,
                'fault_min': fault_min,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'statistically_significant': p_value < 0.05,
                'effect_size': 'large' if cohens_d > 0.8 else 'medium' if cohens_d > 0.5 else 'small'
            }
            
            results.append(result)
        
        if metrics_skipped > 0:
            print(f"Skipped {metrics_skipped} metrics due to insufficient data or no variation")
        
        if not results:
            print("❌ Warning: No metrics with sufficient data for comparison!")
            return pd.DataFrame()
        
        stats_df = pd.DataFrame(results)
        
        # Check for any remaining infinite values
        inf_count = np.isinf(stats_df['p75_change_percent']).sum()
        if inf_count > 0:
            print(f"⚠️  Found {inf_count} metrics with infinite p75_change_percent, replacing with capped values")
            stats_df['p75_change_percent'] = stats_df['p75_change_percent'].replace(
                [np.inf, -np.inf], [1000.0, -1000.0]
            )
        
        self.comparison_results['statistical'] = stats_df
        print(f"✅ Successfully compared {len(stats_df)} metrics")
        return stats_df

    def anomaly_detection_comparison(self, contamination=0.3, use_percentile_features=True):
        """
        Use Isolation Forest algorithm to detect anomalous metrics with enhanced percentile features
        
        Parameters:
        -----------
        contamination : float, default=0.3
            Proportion of outliers in the data set (0 < contamination <= 0.5)
        use_percentile_features : bool, default=True
            Whether to include 75th percentile based features in anomaly detection
            
        Returns:
        --------
        DataFrame with anomaly detection results
        """
        print(f"\n=== Starting Anomaly Detection (contamination={contamination}) ===")
        if use_percentile_features:
            print("✅ Including 75th percentile features for enhanced anomaly detection")
        
        if 'statistical' not in self.comparison_results:
            self.basic_statistical_comparison()
        
        stats_df = self.comparison_results['statistical']
        
        if stats_df.empty:
            print("❌ No statistical data available for anomaly detection")
            return pd.DataFrame()
        
        if len(stats_df) < 3:
            print("❌ Insufficient metrics for anomaly detection (need at least 3 metrics)")
            return pd.DataFrame()
        
        # Prepare features for anomaly detection
        features = []
        metric_names = []
        
        print("Extracting features for anomaly detection...")
        
        for _, row in stats_df.iterrows():
            metric = row['metric']
            
            # Base features
            mean_change_pct = row['mean_change_percent']
            cohens_d = row['cohens_d']
            p_value = row['p_value']
            std_change_pct = row.get('std_change_percent', 0)
            
            # Calculate additional base features
            normal_std = row['normal_std']
            fault_std = row['fault_std']
            normal_mean = row['normal_mean']
            fault_mean = row['fault_mean']
            mean_absolute_change = row['mean_absolute_change']
            
            relative_std = fault_std / (normal_std + 1e-10)
            relative_change = mean_absolute_change / (abs(normal_mean) + 1e-10)
            z_score_change = (fault_mean - normal_mean) / (normal_std + 1e-10)
            
            # Start with base features
            feature_set = [
                mean_change_pct,                    # Magnitude of change
                abs(mean_change_pct),               # Absolute change magnitude
                cohens_d,                           # Effect size
                -np.log10(p_value + 1e-10),        # Significance level (-log10(p))
                std_change_pct,                     # Variability change
                abs(std_change_pct),                # Absolute variability change
                relative_std,                       # Relative standard deviation
                relative_change,                    # Relative change
                z_score_change                      # Z-score like change
            ]
            
            # Add 75th percentile features if enabled
            if use_percentile_features:
                normal_p75 = row.get('normal_p75', normal_mean)
                fault_p75 = row.get('fault_p75', fault_mean)
                p75_change = row.get('p75_absolute_change', 0)
                p75_change_pct = row.get('p75_change_percent', 0)

                # Protect against extreme values in percentile features
                p75_change_pct_safe = p75_change_pct
                if abs(p75_change_pct_safe) > 1000:  # Cap extreme values
                    p75_change_pct_safe = 1000.0 if p75_change_pct_safe > 0 else -1000.0
                
                p75_ratio = fault_p75 / (normal_p75 + 1e-10)
                if p75_ratio > 100:  # Cap extreme ratios
                    p75_ratio = 100.0
                elif p75_ratio < 0.01 and p75_ratio > 0:  # Handle very small ratios
                    p75_ratio = 0.01
                
                percentile_features = [
                    p75_change_pct,                  # 75th percentile change percentage
                    abs(p75_change_pct),             # Absolute 75th percentile change
                    p75_change,                      # Absolute 75th percentile difference
                    fault_p75 / (normal_p75 + 1e-10), # 75th percentile ratio
                    (fault_p75 - normal_p75) / (normal_std + 1e-10), # 75th percentile Z-score
                    abs(fault_p75 - normal_p75) / (normal_std + 1e-10), # Normalized 75th percentile change
                ]
                
                feature_set.extend(percentile_features)
            
            features.append(feature_set)
            metric_names.append(metric)
        
        # Feature names for interpretation
        base_feature_names = [
            'mean_change_pct', 'abs_mean_change_pct', 'cohens_d', 'neg_log_pvalue',
            'std_change_pct', 'abs_std_change_pct', 'relative_std', 
            'relative_change', 'z_score_change'
        ]
        
        percentile_feature_names = [
            'p75_change_pct', 'abs_p75_change_pct', 'p75_absolute_change',
            'p75_ratio', 'p75_z_score', 'p75_normalized_change'
        ]
        
        if use_percentile_features:
            feature_names = base_feature_names + percentile_feature_names
            print(f"✅ Generated {len(feature_names)} features ({len(base_feature_names)} base + {len(percentile_feature_names)} percentile) for {len(metric_names)} metrics")
        else:
            feature_names = base_feature_names
            print(f"✅ Generated {len(feature_names)} base features for {len(metric_names)} metrics")
        
        # Standardize features
        scaler = StandardScaler()
        try:
            features_scaled = scaler.fit_transform(features)
            print("✅ Features standardized successfully")
        except Exception as e:
            print(f"❌ Feature standardization failed: {e}")
            return pd.DataFrame()
        
        # Apply Isolation Forest
        print("Applying Isolation Forest algorithm...")
        try:
            iso_forest = IsolationForest(
                contamination=min(contamination, 0.5),
                random_state=42,
                n_estimators=100,
                max_samples='auto'
            )
            anomaly_predictions = iso_forest.fit_predict(features_scaled)
            anomaly_scores = iso_forest.decision_function(features_scaled)
            print("✅ Anomaly detection completed successfully")
        except Exception as e:
            print(f"❌ Anomaly detection failed: {e}")
            return pd.DataFrame()
        
        # Calculate feature importance
        normal_mask = anomaly_predictions == 1
        anomalous_mask = anomaly_predictions == -1
        
        if np.sum(anomalous_mask) > 0:
            normal_features = features_scaled[normal_mask]
            anomalous_features = features_scaled[anomalous_mask]
            
            # Feature importance based on difference in distributions
            feature_importance = np.std(anomalous_features, axis=0) - np.std(normal_features, axis=0)
            feature_importance = np.abs(feature_importance)
            feature_importance = feature_importance / np.sum(feature_importance)  # Normalize
        else:
            feature_importance = np.ones(len(feature_names)) / len(feature_names)
        
        # Analyze percentile feature importance
        if use_percentile_features:
            percentile_importance = sum(feature_importance[len(base_feature_names):])
            base_importance = sum(feature_importance[:len(base_feature_names)])
            print(f"📊 Feature importance: Base features: {base_importance:.3f}, Percentile features: {percentile_importance:.3f}")
        
        # Build results DataFrame
        anomaly_results = []
        for i, (metric, prediction, score) in enumerate(zip(metric_names, anomaly_predictions, anomaly_scores)):
            row = stats_df[stats_df['metric'] == metric].iloc[0]
            
            # Get top contributing features for this metric
            feature_contributions = np.abs(features_scaled[i]) * feature_importance
            top_features_idx = np.argsort(feature_contributions)[-3:][::-1]  # Top 3 features
            top_features = [(feature_names[idx], feature_contributions[idx]) for idx in top_features_idx]
            
            # Check if percentile features are among top contributors
            percentile_in_top = any('p75' in feature[0] for feature in top_features)
            
            anomaly_results.append({
                'metric': metric,
                'is_anomalous': prediction == -1,
                'anomaly_score': score,
                'anomaly_rank': len(anomaly_scores) - np.argsort(anomaly_scores).argsort()[i],
                'normal_mean': row['normal_mean'],
                'fault_mean': row['fault_mean'],
                'change_percent': row['mean_change_percent'],
                'absolute_change': row['mean_absolute_change'],
                'cohens_d': row['cohens_d'],
                'p_value': row['p_value'],
                'statistically_significant': row['statistically_significant'],
                'effect_size': row['effect_size'],
                'normal_p75': row.get('normal_p75', np.nan),
                'fault_p75': row.get('fault_p75', np.nan),
                'p75_change_percent': row.get('p75_change_percent', np.nan),
                'percentile_driven_anomaly': percentile_in_top,
                'top_contributing_features': str([f[0] for f in top_features]),
                'feature_contribution_scores': str([round(f[1], 3) for f in top_features]),
                'normal_data_points': row['normal_data_points'],
                'fault_data_points': row['fault_data_points']
            })
        
        anomaly_df = pd.DataFrame(anomaly_results)
        
        # Sort by anomaly score (most anomalous first)
        anomaly_df = anomaly_df.sort_values('anomaly_score', ascending=True)
        
        # Add some summary statistics
        n_anomalies = np.sum(anomaly_predictions == -1)
        avg_anomaly_score = np.mean(anomaly_scores)
        
        # Count percentile-driven anomalies
        n_percentile_driven = len(anomaly_df[(anomaly_df['is_anomalous']) & (anomaly_df['percentile_driven_anomaly'])])
        
        print(f"\n📊 Anomaly Detection Summary:")
        print(f"   • Total metrics analyzed: {len(anomaly_df)}")
        print(f"   • Anomalous metrics detected: {n_anomalies}")
        print(f"   • Average anomaly score: {avg_anomaly_score:.3f}")
        print(f"   • Contamination parameter: {contamination}")
        if use_percentile_features:
            print(f"   • Percentile-driven anomalies: {n_percentile_driven}")
        
        if n_anomalies > 0:
            print(f"\n🚨 Top anomalous metrics:")
            top_anomalies = anomaly_df[anomaly_df['is_anomalous']].head(5)
            for _, row in top_anomalies.iterrows():
                percentile_indicator = " 📊" if row['percentile_driven_anomaly'] else ""
                print(f"   • {row['metric']} (score: {row['anomaly_score']:.3f}, change: {row['change_percent']:.1f}%{percentile_indicator})")
        
        # Store results
        self.comparison_results['anomaly'] = anomaly_df
        self.comparison_results['anomaly_features'] = {
            'feature_names': feature_names,
            'feature_importance': feature_importance,
            'contamination': contamination,
            'use_percentile_features': use_percentile_features
        }
        
        return anomaly_df

    def get_percentile_analysis(self, top_n=10):
        """
        Get metrics with the largest 75th percentile changes
        """
        if 'statistical' not in self.comparison_results:
            self.basic_statistical_comparison()
        
        stats_df = self.comparison_results['statistical']
        
        if stats_df.empty:
            return pd.DataFrame()
        
        # Calculate absolute 75th percentile change
        stats_df['abs_p75_change'] = abs(stats_df['p75_change_percent'])
        
        # Get top metrics by 75th percentile change
        top_percentile_metrics = stats_df.nlargest(top_n, 'abs_p75_change')[
            ['metric', 'normal_p75', 'fault_p75', 'p75_change_percent', 
             'mean_change_percent', 'p_value', 'effect_size']
        ]
        
        return top_percentile_metrics

    def generate_summary_report(self):
        """Generate comparison analysis summary report"""
        if 'statistical' not in self.comparison_results:
            self.basic_statistical_comparison()
        
        stats_df = self.comparison_results['statistical']
        
        if stats_df.empty:
            return pd.DataFrame({
                'total_metrics_compared': [0],
                'message': ['No metrics available for comparison']
            })
        
        total_metrics = len(stats_df)
        significant_metrics = len(stats_df[stats_df['statistically_significant']])
        large_effect_metrics = len(stats_df[stats_df['effect_size'] == 'large'])
        
        # Add anomaly information if available
        n_anomalies = 0
        if 'anomaly' in self.comparison_results:
            anomaly_df = self.comparison_results['anomaly']
            n_anomalies = len(anomaly_df[anomaly_df['is_anomalous']])
        
        summary_data = {
            'total_metrics_compared': [total_metrics],
            'significantly_changed_metrics': [significant_metrics],
            'significance_ratio': [significant_metrics / total_metrics if total_metrics > 0 else 0],
            'metrics_with_large_effect': [large_effect_metrics],
            'largest_increase_metric': [stats_df.loc[stats_df['mean_change_percent'].idxmax(), 'metric']],
            'largest_increase_pct': [stats_df['mean_change_percent'].max()],
            'largest_decrease_metric': [stats_df.loc[stats_df['mean_change_percent'].idxmin(), 'metric']],
            'largest_decrease_pct': [stats_df['mean_change_percent'].min()],
            'avg_change_percent': [stats_df['mean_change_percent'].mean()],
            'fault_only_metrics_count': [len(self.metrics_only_in_fault)],
            'anomalous_metrics_count': [n_anomalies]
        }
        
        return pd.DataFrame(summary_data)

    def get_most_changed_metrics(self, n=10, by='change_percent'):
        """Get the most changed metrics"""
        if 'statistical' not in self.comparison_results:
            self.basic_statistical_comparison()
        
        stats_df = self.comparison_results['statistical']
        
        if stats_df.empty:
            return pd.DataFrame()
        
        if by == 'absolute_change':
            stats_df['sort_key'] = abs(stats_df['mean_absolute_change'])
        elif by == 'change_percent':
            stats_df['sort_key'] = abs(stats_df['mean_change_percent'])
        elif by == 'cohens_d':
            stats_df['sort_key'] = abs(stats_df['cohens_d'])
        elif by == 'anomaly_score' and 'anomaly' in self.comparison_results:
            anomaly_df = self.comparison_results['anomaly']
            stats_df = stats_df.merge(anomaly_df[['metric', 'anomaly_score']], on='metric')
            stats_df['sort_key'] = abs(stats_df['anomaly_score'])
        else:
            stats_df['sort_key'] = abs(stats_df['mean_absolute_change'])
        
        n = min(n, len(stats_df))
        top_metrics = stats_df.nlargest(n, 'sort_key')
        return top_metrics.drop('sort_key', axis=1)

    def get_anomalous_metrics(self, top_n=10):
        """Get the most anomalous metrics"""
        if 'anomaly' not in self.comparison_results:
            self.anomaly_detection_comparison()
        
        anomaly_df = self.comparison_results['anomaly']
        anomalous_metrics = anomaly_df[anomaly_df['is_anomalous']]
        
        if anomalous_metrics.empty:
            print("No anomalous metrics detected")
            return pd.DataFrame()
        
        return anomalous_metrics.head(top_n)

    def visualize_comparison(self, top_n=8, save_path=None, figsize=(18, 14), dpi=150):
        """
        Enhanced visualization with detailed annotations and save functionality

        Parameters:
        -----------
        top_n : int, default=8
            Number of top metrics to display in the main chart.
            This controls how many of the most changed metrics (by percentage change)
            are shown in the horizontal bar chart.
            - Smaller values (3-5): Focus on the most significant changes
            - Medium values (8-12): Balanced view of key changes
            - Larger values (15+): Comprehensive overview but may be crowded
            
        save_path : str, optional, default=None
            Path where the visualization will be saved as an image file.
            - If None: The plot is only displayed and not saved
            - If provided: Saves the plot to the specified path
            - Supported formats: .png, .jpg, .pdf, .svg
            - Examples:
            save_path="analysis_results.png"
            save_path="/path/to/folder/metric_comparison.jpg"
            save_path="reports/2024_analysis.pdf"
            
        figsize : tuple, default=(18, 14)
            Figure dimensions in inches (width, height).
            Controls the overall size and aspect ratio of the visualization.
            - Standard sizes:
            (12, 9): Compact view for reports
            (16, 12): Balanced size for presentations
            (18, 14): Large detailed view (default)
            (20, 16): Extra large for high-resolution outputs
            - Adjust based on your display needs and content density
            
        dpi : int, default=150
            Dots per inch - controls the resolution and quality of the output image.
            Higher values produce sharper images but larger file sizes.
            - 72-100 dpi: Suitable for web display
            - 150 dpi: Good balance for most uses (default)
            - 300 dpi: High quality for printing
            - 600+ dpi: Very high resolution for publications
        """
        if 'statistical' not in self.comparison_results:
            self.basic_statistical_comparison()
        
        stats_df = self.comparison_results['statistical']
        
        if stats_df.empty:
            print("No data available for visualization")
            return None
        
        # Get summary information for annotations
        summary = self.generate_summary_report()
        total_metrics = summary['total_metrics_compared'].iloc[0]
        significant_metrics = summary['significantly_changed_metrics'].iloc[0]
        significance_ratio = summary['significance_ratio'].iloc[0]
        fault_only_count = len(self.metrics_only_in_fault)
        anomalous_count = summary['anomalous_metrics_count'].iloc[0]
        
        top_metrics = self.get_most_changed_metrics(top_n, by='change_percent')
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = plt.GridSpec(3, 3, figure=fig)
        
        # Main title with analysis information
        fig.suptitle(
            f'Metric Comparison Analysis: Normal vs Fault Period\n'
            f'Total Metrics: {total_metrics} | Significant: {significant_metrics} '
            f'({significance_ratio:.1%}) | Anomalous: {anomalous_count} | Fault-Only: {fault_only_count}',
            fontsize=16, fontweight='bold', y=0.95
        )
        
        # 1. Top metrics change percentage (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        metrics_short = [name[:25] + '...' if len(name) > 25 else name for name in top_metrics['metric']]
        colors = ['#ff6b6b' if x > 0 else '#4ecdc4' for x in top_metrics['mean_change_percent']]
        bars = ax1.barh(metrics_short, top_metrics['mean_change_percent'], color=colors, alpha=0.8)
        
        ax1.set_xlabel('Change Percentage (%)', fontweight='bold')
        ax1.set_title(f'Top {top_n} Most Changed Metrics\n(by percentage change)', 
                     fontweight='bold', fontsize=12)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels and significance markers
        for i, (bar, value, p_val) in enumerate(zip(bars, top_metrics['mean_change_percent'], top_metrics['p_value'])):
            width = bar.get_width()
            sign_symbol = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            label_color = 'darkred' if width >= 0 else 'darkblue'
            ax1.text(width + (0.01 if width >= 0 else -0.01) * max(abs(top_metrics['mean_change_percent'])), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{value:+.1f}%{sign_symbol}', 
                    ha='left' if width >= 0 else 'right', 
                    va='center', fontsize=9, fontweight='bold', color=label_color)
        
        # 2. Effect size distribution (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        effect_sizes = stats_df['cohens_d']
        n, bins, patches = ax2.hist(effect_sizes, bins=15, alpha=0.7, color='#6a89cc', edgecolor='black')
        
        # Color bars based on effect size
        for i, (patch, bin_edge) in enumerate(zip(patches, bins)):
            if bin_edge > 0.8:
                patch.set_facecolor('#e55039')
            elif bin_edge > 0.5:
                patch.set_facecolor('#fad390')
            else:
                patch.set_facecolor('#78e08f')
        
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Medium (0.5)')
        ax2.axvline(x=0.8, color='darkred', linestyle='--', alpha=0.7, label='Large (0.8)')
        ax2.set_xlabel("Cohen's d Effect Size", fontweight='bold')
        ax2.set_ylabel('Number of Metrics', fontweight='bold')
        ax2.set_title('Effect Size Distribution\n(>0.8: Large, >0.5: Medium, <0.5: Small)', 
                     fontweight='bold', fontsize=12)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Add effect size statistics
        effect_stats = f"Mean: {effect_sizes.mean():.2f}\nMax: {effect_sizes.max():.2f}\n>0.8: {(effect_sizes > 0.8).sum()}"
        ax2.text(0.95, 0.95, effect_stats, transform=ax2.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
        
        # 3. P-value distribution (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        p_values = stats_df['p_value']
        n, bins, patches = ax3.hist(p_values, bins=20, alpha=0.7, color='#82ccdd', edgecolor='black')
        
        # Color significance region
        sig_cutoff = 0.05
        for i, (patch, bin_edge) in enumerate(zip(patches, bins)):
            if bin_edge < sig_cutoff:
                patch.set_facecolor('#b8e994')
        
        ax3.axvline(x=sig_cutoff, color='red', linestyle='--', alpha=0.7, 
                   label=f'Significance (p<{sig_cutoff})')
        ax3.set_xlabel('P-value', fontweight='bold')
        ax3.set_ylabel('Number of Metrics', fontweight='bold')
        ax3.set_title('P-value Distribution\n(Green: Statistically Significant)', 
                     fontweight='bold', fontsize=12)
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Add p-value statistics
        sig_count = (p_values < 0.05).sum()
        pval_stats = f"Significant: {sig_count}/{len(p_values)}\n({sig_count/len(p_values):.1%})"
        ax3.text(0.95, 0.95, pval_stats, transform=ax3.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
        
        # 4. Anomaly detection results (middle-left)
        ax4 = fig.add_subplot(gs[1, 0])
        if 'anomaly' in self.comparison_results:
            anomaly_df = self.comparison_results['anomaly']
            anomalous_metrics = anomaly_df[anomaly_df['is_anomalous']]
            
            if not anomalous_metrics.empty:
                # Show top anomalous metrics
                top_anomalous = anomalous_metrics.head(6)
                metrics_anom = [name[:20] + '...' if len(name) > 20 else name for name in top_anomalous['metric']]
                scores = top_anomalous['anomaly_score']
                
                colors_anom = ['#e55039' if score < -0.1 else '#fad390' for score in scores]
                bars_anom = ax4.barh(metrics_anom, scores, color=colors_anom, alpha=0.8)
                
                ax4.set_xlabel('Anomaly Score', fontweight='bold')
                ax4.set_title(f'Top Anomalous Metrics\n(lower score = more anomalous)', 
                             fontweight='bold', fontsize=12)
                ax4.grid(axis='x', alpha=0.3)
                
                # Add anomaly score labels
                for bar, score in zip(bars_anom, scores):
                    width = bar.get_width()
                    ax4.text(width - 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{score:.3f}', ha='right', va='center', 
                            fontsize=8, fontweight='bold', color='white')
            else:
                ax4.text(0.5, 0.5, 'No Anomalous\nMetrics Detected', 
                        ha='center', va='center', transform=ax4.transAxes,
                        fontsize=12, fontweight='bold')
                ax4.set_title('Anomaly Detection Results', fontweight='bold', fontsize=12)
        else:
            ax4.text(0.5, 0.5, 'Anomaly Detection\nNot Run', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, fontweight='bold')
            ax4.set_title('Anomaly Detection Results', fontweight='bold', fontsize=12)
        
        # 5. Effect size composition (middle-middle)
        ax5 = fig.add_subplot(gs[1, 1])
        effect_sizes = stats_df['effect_size'].value_counts()
        
        if not effect_sizes.empty:
            colors_effect = {'large': '#e55039', 'medium': '#fad390', 'small': '#78e08f'}
            colors_list = [colors_effect.get(effect, '#cccccc') for effect in effect_sizes.index]
            
            wedges, texts, autotexts = ax5.pie(effect_sizes.values, labels=effect_sizes.index, 
                                             autopct='%1.1f%%', colors=colors_list, startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax5.set_title('Effect Size Composition\n(Cohen\'s d)', fontweight='bold', fontsize=12)
        else:
            ax5.text(0.5, 0.5, 'No effect size data', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Effect Size Composition', fontweight='bold', fontsize=12)
        
        # 6. Data points distribution (middle-right)
        ax6 = fig.add_subplot(gs[1, 2])
        normal_points = stats_df['normal_data_points']
        fault_points = stats_df['fault_data_points']
        
        x_pos = np.arange(len(stats_df))
        width = 0.35
        
        ax6.bar(x_pos - width/2, normal_points, width, label='Normal', alpha=0.7, color='#4a69bd')
        ax6.bar(x_pos + width/2, fault_points, width, label='Fault', alpha=0.7, color='#e55039')
        
        ax6.set_xlabel('Metric Index', fontweight='bold')
        ax6.set_ylabel('Number of Data Points', fontweight='bold')
        ax6.set_title('Data Points Distribution\nper Metric', fontweight='bold', fontsize=12)
        ax6.legend()
        ax6.grid(alpha=0.3)
        
        # Add summary statistics
        data_stats = f"Avg Normal: {normal_points.mean():.0f}\nAvg Fault: {fault_points.mean():.0f}"
        ax6.text(0.95, 0.95, data_stats, transform=ax6.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
        
        # 7. Detailed summary text (bottom span)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Prepare detailed summary text
        if not summary.empty:
            summary_text = (
                f"ANALYSIS SUMMARY:\n"
                f"• Total metrics compared: {total_metrics}\n"
                f"• Statistically significant changes: {significant_metrics} ({significance_ratio:.1%})\n"
                f"• Metrics with large effect size: {summary['metrics_with_large_effect'].iloc[0]}\n"
                f"• Anomalous metrics detected: {anomalous_count}\n"
                f"• Largest increase: {summary['largest_increase_metric'].iloc[0]} "
                f"(+{summary['largest_increase_pct'].iloc[0]:.1f}%)\n"
                f"• Largest decrease: {summary['largest_decrease_metric'].iloc[0]} "
                f"({summary['largest_decrease_pct'].iloc[0]:.1f}%)\n"
                f"• Average change: {summary['avg_change_percent'].iloc[0]:.1f}%\n"
                f"• Metrics only in fault data: {fault_only_count}\n"
                f"• Analysis timestamp: {self.analysis_timestamp}"
            )
            
            # Add anomaly detection info if available
            if 'anomaly' in self.comparison_results:
                anomaly_df = self.comparison_results['anomaly']
                if not anomaly_df.empty:
                    top_anomaly = anomaly_df.iloc[0]
                    summary_text += f"\n• Most anomalous metric: {top_anomaly['metric']} (score: {top_anomaly['anomaly_score']:.3f})"
        else:
            summary_text = "No summary data available"
        
        ax7.text(0.02, 0.5, summary_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3),
                fontfamily='monospace')
        
        # Add footer
        fig.text(0.5, 0.02, 
                f'Metric Comparison Analysis | Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', fontsize=10, style='italic', alpha=0.7)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.08)
        
        # Save figure if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"✅ Visualization saved to: {save_path}")
        
        plt.show()
        return fig
