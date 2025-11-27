from utils.load_json_configs import JSONConfig

# Example usage of the advanced version
if __name__ == "__main__":
    # Create config from dictionary for demonstration
    config_data = {
        "app": {
            "name": "MyApp",
            "version": "1.0.0"
        },
        "settings": {
            "max_workers": 4,
            "timeout": 30
        }
    }
    
    config = JSONConfig(config_dict=config_data)
    
    # Access configuration
    print(f"App name: {config.app.name}")
    print(f"Max workers: {config.settings.max_workers}")
    
    # Set new values
    config.settings.set("timeout", 60)
    print(f"Updated timeout: {config.settings.timeout}")
    
    # Convert to dictionary
    print(f"As dict: {config.to_dict()}")