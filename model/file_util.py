import shutil
import re
from pathlib import Path
from typing import List, Optional

class FileCopier:
    """
    Advanced file copier with multiple filtering options.
    """
    
    def __init__(self, source_dir: str, destination_dir: str):
        self.source_dir = Path(source_dir)
        self.destination_dir = Path(destination_dir)
        self.destination_dir.mkdir(parents=True, exist_ok=True)
    
    def copy_files_with_filters(
        self,
        substring: str,
        start_with: Optional[str] = None,
        end_with: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        case_sensitive: bool = False,
        use_regex: bool = False
    ) -> List[str]:
        """
        Copy files with multiple filtering options.
        
        Args:
            substring: Substring to search for in filenames
            start_with: Files must start with this string
            end_with: Files must end with this string
            file_extensions: List of allowed file extensions
            case_sensitive: Whether matching should be case sensitive
            use_regex: Treat substring as regex pattern
            
        Returns:
            List of copied filenames
        """
        copied_files = []
        
        for file_path in self.source_dir.iterdir():
            if file_path.is_file():
                filename = file_path.name
                
                if self._matches_filters(
                    filename, substring, start_with, end_with, 
                    file_extensions, case_sensitive, use_regex
                ):
                    destination_file = self.destination_dir / filename
                    shutil.copy2(file_path, destination_file)
                    copied_files.append(filename)
        
        return copied_files
    
    def _matches_filters(
        self,
        filename: str,
        substring: str,
        start_with: Optional[str],
        end_with: Optional[str],
        file_extensions: Optional[List[str]],
        case_sensitive: bool,
        use_regex: bool
    ) -> bool:
        """Check if filename matches all specified filters."""
        # Prepare filename for comparison
        compare_name = filename if case_sensitive else filename.lower()
        compare_substring = substring if case_sensitive else substring.lower()
        
        # Check file extension
        if file_extensions:
            file_ext = Path(filename).suffix.lower()
            if file_ext not in [ext.lower() for ext in file_extensions]:
                return False
        
        # Check start with
        if start_with:
            compare_start = start_with if case_sensitive else start_with.lower()
            if not compare_name.startswith(compare_start):
                return False
        
        # Check end with
        if end_with:
            compare_end = end_with if case_sensitive else end_with.lower()
            if not compare_name.endswith(compare_end):
                return False
        
        # Check substring/regex
        if use_regex:
            pattern = compare_substring if case_sensitive else compare_substring
            if not re.search(pattern, compare_name):
                return False
        else:
            if compare_substring not in compare_name:
                return False
        
        return True
    
    def copy_files_multiple_substrings(
        self,
        substrings: List[str],
        start_with: Optional[str] = None,
        match_all: bool = False
    ) -> List[str]:
        """
        Copy files containing multiple substrings.
        
        Args:
            substrings: List of substrings to search for
            start_with: Files must start with this string
            match_all: If True, all substrings must be present; if False, any substring
        """
        copied_files = []
        
        for file_path in self.source_dir.iterdir():
            if file_path.is_file():
                filename = file_path.name
                
                # Check start condition
                if start_with and not filename.startswith(start_with):
                    continue
                
                # Check substring conditions
                if match_all:
                    # All substrings must be present
                    if all(sub in filename for sub in substrings):
                        destination_file = self.destination_dir / filename
                        shutil.copy2(file_path, destination_file)
                        copied_files.append(filename)
                else:
                    # Any substring must be present
                    if any(sub in filename for sub in substrings):
                        destination_file = self.destination_dir / filename
                        shutil.copy2(file_path, destination_file)
                        copied_files.append(filename)
        
        return copied_files

