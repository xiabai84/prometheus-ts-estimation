from model.file_util import FileCopier

if __name__ == "__main__":
    # Initialize copier
    copier = FileCopier("/path/to/source", "/path/to/destination")
    
    # Example 1: Copy files starting with "report" and containing "Q1"
    files1 = copier.copy_files_with_filters(
        substring="Q1",
        start_with="report"
    )
    print(f"Copied files (report Q1): {files1}")
    
    # Example 2: Copy PDF files containing "invoice"
    files2 = copier.copy_files_with_filters(
        substring="invoice",
        file_extensions=[".pdf", ".PDF"]
    )
    print(f"Copied files (invoice PDFs): {files2}")
    
    # Example 3: Copy files containing any of these substrings
    files3 = copier.copy_files_multiple_substrings(
        substrings=["urgent", "important", "critical"],
        start_with="doc_"
    )
    print(f"Copied files (urgent/important/critical): {files3}")
    
    # Example 4: Copy files containing all of these substrings
    files4 = copier.copy_files_multiple_substrings(
        substrings=["2024", "financial", "summary"],
        match_all=True
    )
    print(f"Copied files (2024 financial summary): {files4}")
    