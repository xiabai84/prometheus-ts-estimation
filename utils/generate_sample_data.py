import pandas as pd
import numpy as np

# Example usage and test data generation
def generate_sample_prometheus_data(samples=2000, file_name="sample_prometheus_data.csv"):
    """
    Generate sample Prometheus-like data for testing
    """
    dates = pd.date_range('2024-01-01', periods=samples, freq='h')
    
    data = {
        'timestamp': dates,
        'http_requests_total': np.random.exponential(10, samples),
        'cpu_usage': np.random.normal(50, 20, samples),
        'memory_usage': np.random.gamma(2, 2, samples),
        'disk_io': np.random.poisson(5, samples),
        'network_bytes': np.random.lognormal(3, 1, samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some zeros and nulls for realism
    df.loc[df.sample(100).index, 'http_requests_total'] = 0
    df.loc[df.sample(50).index, 'cpu_usage'] = 0
    df.loc[df.sample(30).index, 'memory_usage'] = np.nan
    
    df.to_csv(file_name, index=False)
    print(f"{samples} sample data generated: {file_name}")
    return file_name

# Create sample marketing sales data
def create_sample_sales_data(samples=5000):
    np.random.seed(42)
    data = {
        'order_id': range(1, (samples+1)),
        'date': pd.date_range('2024-01-01', periods=samples, freq='D'),
        'region': np.random.choice(['North', 'South', 'East', 'West'], samples),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], samples),
        'product': np.random.choice(['Laptop', 'Phone', 'Shirt', 'Shoes', 'Furniture', 'Equipment'], samples),
        'sales_amount': np.random.randint(100, 5000, samples),
        'quantity': np.random.randint(1, 50, samples),
        'profit': np.random.randint(-200, 1000, samples),
        'customer_type': np.random.choice(['New', 'Returning', 'VIP'], samples),
        'rating': np.random.choice([1, 2, 3, 4, 5], samples, p=[0.1, 0.2, 0.3, 0.3, 0.1])
    }
    return pd.DataFrame(data)