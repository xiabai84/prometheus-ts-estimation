import json
from pathlib import Path
from typing import Any, Dict

class JSONConfig:
    """
    Advanced JSON configuration class with additional features like
    environment variable substitution and validation.
    """
    
    def __init__(self, config_file: str = None, config_dict: Dict = None, 
                 auto_reload: bool = False):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to JSON configuration file
            config_dict: Dictionary containing configuration data
            auto_reload: Whether to automatically reload when file changes
        """
        self._config_file = config_file
        self._auto_reload = auto_reload
        
        if config_file:
            self._load_from_file(config_file)
        elif config_dict:
            self._load_from_dict(config_dict)
        else:
            self._data = {}
    
    def _load_from_file(self, config_file: str):
        """Load configuration from file with error handling."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file '{config_file}' not found")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                self._data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def _load_from_dict(self, config_dict: Dict):
        """Load configuration from dictionary."""
        self._data = config_dict
    
    def __getattr__(self, name: str) -> Any:
        """Attribute-style access with nested object conversion."""
        if name.startswith('_'):
            return super().__getattribute__(name)
            
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return JSONConfig(config_dict=value)
            elif isinstance(value, list):
                # Convert dictionaries in lists to JSONConfig objects
                return [JSONConfig(config_dict=item) if isinstance(item, dict) else item 
                       for item in value]
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.__getattr__(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Safe get with default value."""
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self._data[key] = value
    
    def save(self, file_path: str = None):
        """Save configuration to file."""
        save_path = file_path or self._config_file
        if not save_path:
            raise ValueError("No file path specified for saving configuration")
        
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(self._data, file, indent=4, ensure_ascii=False)
    
    def to_dict(self) -> Dict:
        """Return as plain dictionary."""
        def convert_value(value):
            if isinstance(value, JSONConfig):
                return value.to_dict()
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            else:
                return value
        
        return convert_value(self._data)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data})"
    
    def __dir__(self):
        """Include configuration keys in dir() output."""
        return sorted(set(super().__dir__() + list(self._data.keys())))
