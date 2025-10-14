from utils.generate_sample_data import create_sample_sales_data
from model.report_builder import ComprehensiveReportBuilder

# Comprehensive Usage Examples
print("=" * 60)
print("COMPREHENSIVE USAGE EXAMPLES")
print("=" * 60)

# Example 1: Basic Multi-Filter Builder
print("\n1. BASIC MULTI-FILTER BUILDER")

# Usage Examples
df = create_sample_sales_data()
print("Original Data Shape:", df.shape)
print("\nSample Data:")
print(df.head())

basic_builder = ComprehensiveReportBuilder(df)

# Create multiple filter groups
(basic_builder
 .create_filter_group("high_sales_north")
 .add_column_filter('sales_amount', '>', 2000)
 .add_column_filter('region', '==', 'North')
 .add_column_filter('profit', '>', 0)
 
 .create_filter_group("electronics_vip")
 .add_column_filter('product_category', '==', 'Electronics')
 .add_column_filter('customer_type', '==', 'VIP')
 .add_column_filter('rating', '>=', 4)
 
 .create_filter_group("profitable_products")
 .add_range_filter('sales_amount', min_val=1000, max_val=4000)
 .add_column_filter('profit', '>', 200)
 .add_column_filter('quantity', '>', 10))

# Build all filtered DataFrames
filtered_dfs = basic_builder.build_all()

print("Filter Groups Summary:")
for group_name, filtered_df in filtered_dfs.items():
    print(f"  {group_name}: {len(filtered_df)} rows")

# Build specific group
high_sales_north = basic_builder.build_single("high_sales_north")
print(f"\nHigh Sales North sample ({len(high_sales_north)} rows):")
print(high_sales_north.head()[['region', 'product', 'sales_amount', 'profit']])

# Example 2: Dynamic Constraint Combinations
print("\n2. DYNAMIC CONSTRAINT COMBINATIONS")

# Create a builder for complex business scenarios
business_builder = ComprehensiveReportBuilder(df)

# Define various business constraints
(business_builder
 .create_filter_group("q1_targets")
 .add_column_filter('sales_amount', '>=', 2000)
 .add_column_filter('profit', '>=', 300)
 .add_column_filter('rating', '>=', 3)
 
 .create_filter_group("high_risk_orders") 
 .add_column_filter('profit', '<', 0)
 .add_column_filter('rating', '<=', 2)
 
 .create_filter_group("premium_products")
 .add_column_filter('product_category', 'in', ['Electronics', 'Sports'])
 .add_column_filter('sales_amount', '>=', 1500)
 
 .create_filter_group("volume_deals")
 .add_column_filter('quantity', '>=', 20)
 .add_column_filter('sales_amount', '>=', 1000))

# Build intersection of multiple constraints
print("\nCombined Constraints:")
combined_high_value = business_builder.build_combined(
    ['q1_targets', 'premium_products'], 
    operation='intersection'
)
print(f"Q1 Targets + Premium Products: {len(combined_high_value)} rows")

# Build union of constraints
combined_issues = business_builder.build_combined(
    ['high_risk_orders', 'volume_deals'],
    operation='union'  
)
print(f"High Risk OR Volume Deals: {len(combined_issues)} rows")

# Get comprehensive summary
summaries = business_builder.get_summary_report()
print("\nDetailed Summary:")
print(summaries)

# Example 3: Real-time Filter Application
print("\n3. REAL-TIME FILTER APPLICATION")

def create_dynamic_filters():
    """Create filters based on dynamic conditions"""
    dynamic_builder = ComprehensiveReportBuilder(df)
    
    # These could come from user input, configuration, etc.
    dynamic_constraints = [
        {'group': 'user_defined_1', 'column': 'sales_amount', 'op': '>', 'value': 2500},
        {'group': 'user_defined_1', 'column': 'region', 'op': 'in', 'value': ['North', 'South']},
        {'group': 'user_defined_2', 'column': 'product_category', 'op': '==', 'value': 'Clothing'},
        {'group': 'user_defined_2', 'column': 'profit', 'op': '>', 'value': 100},
    ]
    
    for constraint in dynamic_constraints:
        dynamic_builder.create_filter_group(constraint['group'])
        dynamic_builder.add_column_filter(
            constraint['column'], 
            constraint['op'], 
            constraint['value']
        )
    
    return dynamic_builder.build_all()

dynamic_results = create_dynamic_filters()
print("Dynamic Filter Results:")
for group, result in dynamic_results.items():
    print(f"  {group}: {len(result)} rows")