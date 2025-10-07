# Example usage and test data generation
def generate_sample_prometheus_data():
    """
    Generate sample Prometheus-like data for testing
    """
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    
    data = {
        'timestamp': dates,
        'http_requests_total': np.random.exponential(10, 1000),
        'cpu_usage': np.random.normal(50, 20, 1000),
        'memory_usage': np.random.gamma(2, 2, 1000),
        'disk_io': np.random.poisson(5, 1000),
        'network_bytes': np.random.lognormal(3, 1, 1000)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some zeros and nulls for realism
    df.loc[df.sample(100).index, 'http_requests_total'] = 0
    df.loc[df.sample(50).index, 'cpu_usage'] = 0
    df.loc[df.sample(30).index, 'memory_usage'] = np.nan
    
    df.to_csv('sample_prometheus_data.csv', index=False)
    print("Sample data generated: sample_prometheus_data.csv")
    return 'sample_prometheus_data.csv'