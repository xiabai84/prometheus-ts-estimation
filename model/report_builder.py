import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Callable

class ComprehensiveReportBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.filter_groups = {}  # Store multiple filter groups
        self.current_group = "default"
        self.filter_groups[self.current_group] = {
            'filters': [],
            'sort_columns': [],
            'sort_ascending': True,
            'selected_columns': None,
            'row_limit': None,
            'transformations': [],
            'description': ''
        }
    
    def create_filter_group(self, group_name: str, description: str = "") -> 'ComprehensiveReportBuilder':
        """Create a new filter group"""
        self.current_group = group_name
        self.filter_groups[group_name] = {
            'filters': [],
            'sort_columns': [],
            'sort_ascending': True,
            'selected_columns': None,
            'row_limit': None,
            'transformations': [],
            'description': description
        }
        return self
    
    def switch_group(self, group_name: str) -> 'ComprehensiveReportBuilder':
        """Switch to existing filter group"""
        if group_name not in self.filter_groups:
            raise ValueError(f"Filter group '{group_name}' does not exist")
        self.current_group = group_name
        return self
    
    def add_filter(self, condition: Callable[[pd.DataFrame], pd.Series]) -> 'ComprehensiveReportBuilder':
        """Add filter to current group"""
        self.filter_groups[self.current_group]['filters'].append(condition)
        return self
    
    def add_column_filter(self, column: str, operator: str, value: Any) -> 'ComprehensiveReportBuilder':
        """Add column-based filter to current group"""
        condition = self._create_condition(column, operator, value)
        return self.add_filter(condition)
    
    def add_range_filter(self, column: str, min_val: Any = None, max_val: Any = None) -> 'ComprehensiveReportBuilder':
        """Add range filter to current group"""
        def range_condition(df):
            mask = pd.Series(True, index=df.index)
            if min_val is not None:
                mask = mask & (df[column] >= min_val)
            if max_val is not None:
                mask = mask & (df[column] <= max_val)
            return mask
        return self.add_filter(range_condition)
    
    def set_sorting(self, columns: List[str], ascending: bool = True) -> 'ComprehensiveReportBuilder':
        """Set sorting columns and order for current group"""
        self.filter_groups[self.current_group]['sort_columns'] = columns
        self.filter_groups[self.current_group]['sort_ascending'] = ascending
        return self

    def add_sort_column(self, column: str, ascending: bool = True) -> 'ComprehensiveReportBuilder':
        """Add a sort column to current group (multiple columns supported)"""
        # For multiple sort columns, we store as list of tuples
        if not self.filter_groups[self.current_group]['sort_columns']:
            self.filter_groups[self.current_group]['sort_columns'] = []
        
        if isinstance(self.filter_groups[self.current_group]['sort_columns'], list):
            self.filter_groups[self.current_group]['sort_columns'].append((column, ascending))
        return self

    def add_multiple_filters(self, filters: List[Callable]) -> 'ComprehensiveReportBuilder':
        """Add multiple filters at once to current group"""
        self.filter_groups[self.current_group].extend(filters)
        return self
    
    def select_columns(self, columns: List[str]) -> 'ComprehensiveReportBuilder':
        """Select specific columns to include in current group"""
        self.filter_groups[self.current_group]['selected_columns'] = columns
        return self
    
    def limit_rows(self, limit: int) -> 'ComprehensiveReportBuilder':
        """Limit number of rows in result for current group"""
        self.filter_groups[self.current_group]['row_limit'] = limit
        return self
    
    def add_transformation(self, transform_func: Callable[[pd.DataFrame], pd.DataFrame]) -> 'ComprehensiveReportBuilder':
        """Add data transformation to current group"""
        self.filter_groups[self.current_group]['transformations'].append(transform_func)
        return self

    def _create_condition(self, column: str, operator: str, value: Any) -> Callable:
        """Create condition function based on operator"""
        if operator == "==":
            return lambda df: df[column] == value
        elif operator == "!=":
            return lambda df: df[column] != value
        elif operator == ">":
            return lambda df: df[column] > value
        elif operator == "<":
            return lambda df: df[column] < value
        elif operator == ">=":
            return lambda df: df[column] >= value
        elif operator == "<=":
            return lambda df: df[column] <= value
        elif operator == "in":
            return lambda df: df[column].isin(value)
        elif operator == "not in":
            return lambda df: ~df[column].isin(value)
        elif operator == "contains":
            return lambda df: df[column].str.contains(value, na=False)
        elif operator == "startswith":
            return lambda df: df[column].str.startswith(value, na=False)
        elif operator == "endswith":
            return lambda df: df[column].str.endswith(value, na=False)
        elif operator == "isna":
            return lambda df: df[column].isna()
        elif operator == "notna":
            return lambda df: df[column].notna()
        else:
            raise ValueError(f"Unsupported operator: {operator}")
    
    def build_single(self, group_name: str = None) -> pd.DataFrame:
        """Build DataFrame for a specific filter group"""
        group_name = group_name or self.current_group
        if group_name not in self.filter_groups:
            raise ValueError(f"Filter group '{group_name}' does not exist")
        
        result = self.df.copy()
        group_config = self.filter_groups[group_name]

        for transform_func in group_config['transformations']:
            result = transform_func(result)
        
        for filter_func in group_config['filters']:
            result = result[filter_func(result)]

        # Apply sorting
        if group_config['sort_columns']:
            if isinstance(group_config['sort_columns'][0], tuple):
                # Multiple sort columns with individual directions
                sort_columns = [col for col, _ in group_config['sort_columns']]
                ascending = [asc for _, asc in group_config['sort_columns']]
                result = result.sort_values(by=sort_columns, ascending=ascending)
            else:
                # Single sort column or multiple with same direction
                result = result.sort_values(
                    by=group_config['sort_columns'], 
                    ascending=group_config['sort_ascending']
                )
        # Select specific columns
        if group_config['selected_columns']:
            result = result[group_config['selected_columns']]
        
        # Apply row limit
        if group_config['row_limit']:
            result = result.head(group_config['row_limit'])

        return result
    
    def build_all(self) -> Dict[str, pd.DataFrame]:
        """Build DataFrames for all filter groups with their respective sorting"""
        results = {}
        for group_name in self.filter_groups.keys():
            results[group_name] = self.build_single(group_name)
        return results
    
    def get_group_configuration(self, group_name: str = None) -> Dict[str, Any]:
        """Get configuration for a specific group"""
        group_name = group_name or self.current_group
        if group_name not in self.filter_groups:
            raise ValueError(f"Filter group '{group_name}' does not exist")
        return self.filter_groups[group_name].copy()
    
    def build_combined(self, group_names: List[str], operation: str = "intersection") -> pd.DataFrame:
        """Build DataFrame combining multiple filter groups"""
        if operation not in ["intersection", "union"]:
            raise ValueError("Operation must be 'intersection' or 'union'")
        
        masks = {}
        for group_name in group_names:
            if group_name not in self.filter_groups:
                raise ValueError(f"Filter group '{group_name}' does not exist")
            
            mask = pd.Series(True, index=self.df.index)
            
            group_config = self.filter_groups[group_name]

            for filter_func in group_config['filters']:
                mask = mask & filter_func(self.df)

            masks[group_name] = mask
        
        if operation == "intersection":
            combined_mask = pd.Series(True, index=self.df.index)
            for mask in masks.values():
                combined_mask = combined_mask & mask
        else:  # union
            combined_mask = pd.Series(False, index=self.df.index)
            for mask in masks.values():
                combined_mask = combined_mask | mask
        
        return self.df[combined_mask]
    
    def get_summary_report(self) -> pd.DataFrame:
        """Generate a summary report of all filter groups"""
        summary_data = []
        
        for group_name, config in self.filter_groups.items():
            filtered_df = self.build_single(group_name)
            
            summary_data.append({
                'group_name': group_name,
                'description': config['description'],
                'filter_count': len(config['filters']),
                'sort_columns': str(config['sort_columns']),
                'sort_ascending': config['sort_ascending'],
                'selected_columns': len(config['selected_columns']) if config['selected_columns'] else 'All',
                'row_limit': config['row_limit'] or 'No limit',
                'transformations_count': len(config['transformations']),
                'original_rows': len(self.df),
                'filtered_rows': len(filtered_df),
                'remaining_percentage': round(len(filtered_df) / len(self.df) * 100, 2)
            })
        
        return pd.DataFrame(summary_data)

    def save_csv(self, file_prefix: str, reports: pd.DataFrame, dir="", use_iso_suffix=True):
        if use_iso_suffix:
            now = datetime.now()
            time_iso_format = now.strftime("%Y-%m-%d_%H-%M-%S")
            dir = os.path.join(dir, time_iso_format)
            
        if dir:
            directory = Path(dir)
            directory.mkdir(exist_ok=True)

        for group_name, report in reports.items():
            file_name = f"{file_prefix}_{group_name}"
            file_path = os.path.join(dir, f"{file_name}.csv")
            if group_name != "default":
                
                report.to_csv(file_path, index=False, float_format='%.2f')
                
                affected_microflows = set()
                for row in report.metric_name.values:
                    category = row.split(".")[0]
                    affected_microflows.add(category)

                print("\n" + "="*120)
                print(f"AFFECTED METRICS for GROUP {group_name}. File stored under {file_path}")
                print("="*120)
                for microflow in affected_microflows:
                    print(microflow)
                print("="*120)
        print(f"All files are stored under the direcotry: {dir}")
        return dir