import pandas as pd

def csv_to_confluence_table(csv_file_path: str) -> str:
    """
    Convert CSV file to Confluence Wiki Markup table format
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        return dataframe_to_confluence_table(df)
    
    except Exception as e:
        return f"Error reading CSV file: {e}"

def dataframe_to_confluence_table(df: pd.DataFrame) -> str:
    """
    Convert pandas DataFrame to Confluence Wiki Markup table format
    """
    # Start with table header
    confluence_table = "|| " + " || ".join(str(col) for col in df.columns) + " ||\n"
    # Add table rows
    for _, row in df.iterrows():
        confluence_table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
    return confluence_table

# Usage example
if __name__ == "__main__":
    csv_file = "your_file.csv"  # Replace with your CSV file path
    confluence_output = csv_to_confluence_table(csv_file)
    print(confluence_output)
