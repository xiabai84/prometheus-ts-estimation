import re
from collections import Counter
import argparse

def extract_exceptions_from_file(file_path):
    """
    Extract all different types of exceptions from a txt file
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        list: List containing different exception types
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        # If utf-8 decoding fails, try other encodings
        with open(file_path, 'r', encoding='latin-1') as file:
            content = file.read()
    
    return extract_exceptions_from_text(content)

def extract_exceptions_from_text(text):
    """
    Extract all different types of exceptions from text
    
    Args:
        text (str): Input text content
        
    Returns:
        list: List containing different exception types
    """
    # Define regex patterns to match exceptions
    # Match common exception formats like: Exception, ValueError, FileNotFoundError, etc.
    patterns = [
        r'Exception:\s*([^\n]+)',  # Exception: specific description
        r'(\w+Error):',  # ValueError: etc.
        r'(\w+Exception):',  # FileNotFoundException: etc.
        r'at\s+.*?(\w+Error)',  # Errors in stack traces
        r'at\s+.*?(\w+Exception)',  # Exceptions in stack traces
        r'raise\s+(\w+)',  # raise statements
        r'catch\s*\((\w+)\s+\w+\)',  # catch statements
        r'except\s+(\w+)',  # except statements
    ]
    
    exceptions = set()
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Clean and standardize exception names
            exception_name = match.strip()
            if exception_name and ('error' in exception_name.lower() or 
                                 'exception' in exception_name.lower()):
                exceptions.add(exception_name)
    
    # Additional processing: Find lines containing Exception or Error
    lines = text.split('\n')
    for line in lines:
        if 'exception' in line.lower() or 'error' in line.lower():
            # Try to extract more specific exception names from the line
            words = re.findall(r'[A-Z][a-zA-Z]*(?:Error|Exception)', line)
            for word in words:
                exceptions.add(word)
    
    return sorted(list(exceptions))

def analyze_exception_frequency(text):
    """
    Analyze the frequency of exception occurrences
    
    Args:
        text (str): Input text content
        
    Returns:
        Counter: Counter containing exception occurrence counts
    """
    # Pattern to match exception names
    pattern = r'[A-Z][a-zA-Z]*(?:Error|Exception)'
    exceptions = re.findall(pattern, text)
    
    return Counter(exceptions)

def main():
    parser = argparse.ArgumentParser(description='Extract different types of exceptions from txt document')
    parser.add_argument('file_path', help='Path to the txt file to analyze')
    parser.add_argument('--frequency', '-f', action='store_true', 
                       help='Show exception frequency')
    
    args = parser.parse_args()
    
    try:
        # Extract exception types
        exceptions = extract_exceptions_from_file(args.file_path)
        
        print(f"Found {len(exceptions)} different exception types in file '{args.file_path}':")
        print("-" * 50)
        
        for i, exception in enumerate(exceptions, 1):
            print(f"{i:2d}. {exception}")
        
        # If frequency display is requested
        if args.frequency:
            print("\n" + "=" * 50)
            print("Exception Frequency Analysis:")
            print("-" * 50)
            
            with open(args.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            freq = analyze_exception_frequency(content)
            for exception, count in freq.most_common():
                print(f"{exception}: {count} times")
                
    except FileNotFoundError:
        print(f"Error: File '{args.file_path}' not found")
    except Exception as e:
        print(f"Error processing file: {e}")

# Example function for direct use
def process_text_directly():
    """
    Example of directly processing text content
    """
    # Sample text
    sample_text = """
    2023-10-01 10:00:00 ERROR: FileNotFoundError: File does not exist
    2023-10-01 10:01:00 INFO: Program execution started
    2023-10-01 10:02:00 ERROR: ValueError: Invalid parameter
    2023-10-01 10:03:00 WARNING: High memory usage
    2023-10-01 10:04:00 ERROR: ConnectionError: Connection failed
    2023-10-01 10:05:00 Exception: Unknown exception occurred
    2023-10-01 10:06:00 at java.lang.NullPointerException
    2023-10-01 10:07:00 raise TypeError
    """
    
    exceptions = extract_exceptions_from_text(sample_text)
    print("Exception types extracted from sample text:")
    for exception in exceptions:
        print(f"  - {exception}")

if __name__ == "__main__":
    # If running the script directly, show usage examples
    print("Exception Extraction Tool")
    print("Usage methods:")
    print("1. Command line: python script.py your_log_file.txt")
    print("2. Call in code: extract_exceptions_from_file('your_file.txt')")
    print("\nExample:")
    process_text_directly()
    
    # Uncomment below if you need to run from command line arguments
    # main()