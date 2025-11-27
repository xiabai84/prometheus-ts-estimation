import shutil
import os
import re
from pathlib import Path
from typing import List, Optional, Union, Callable, Dict
from datetime import datetime
import hashlib

class FileCopier:
    """
    Advanced file copier with multiple filtering options and file renaming capabilities.
    """
    
    def __init__(self, source_dir: str, destination_dir: str):
        self.source_dir = Path(source_dir)
        self.destination_dir = Path(destination_dir)
        self.destination_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'files_copied': 0,
            'bytes_copied': 0,
            'files_skipped': 0
        }
    
    def copy_files_with_renaming(
        self,
        substring: str,
        start_with: Optional[str] = None,
        end_with: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        case_sensitive: bool = False,
        use_regex: bool = False,
        rename_template: Optional[str] = None,
        rename_function: Optional[Callable[[Path], str]] = None,
        overwrite: bool = False,
        recursive: bool = False
    ) -> List[Dict]:
        """
        Copy files with multiple filtering options and renaming capabilities.
        
        Args:
            substring: Substring to search for in filenames
            start_with: Files must start with this string
            end_with: Files must end with this string
            file_extensions: List of allowed file extensions
            case_sensitive: Whether matching should be case sensitive
            use_regex: Treat substring as regex pattern
            rename_template: Template for new filename (supports variables)
            rename_function: Custom function to generate new filename
            overwrite: Overwrite existing files in destination
            recursive: Search subdirectories recursively
            
        Returns:
            List of dictionaries with copy results
        """
        copied_files = []
        search_pattern = "**/*" if recursive else "*"
        
        for file_path in self.source_dir.glob(search_pattern):
            if file_path.is_file():
                filename = file_path.name
                
                if self._matches_filters(
                    filename, substring, start_with, end_with, 
                    file_extensions, case_sensitive, use_regex
                ):
                    # Generate new filename
                    new_filename = self._generate_new_filename(
                        file_path, filename, rename_template, rename_function
                    )
                    
                    # Copy file with new name
                    result = self._copy_single_file(
                        file_path, new_filename, overwrite
                    )
                    copied_files.append(result)
        
        return copied_files
    
    def _generate_new_filename(
        self,
        file_path: Path,
        original_name: str,
        rename_template: Optional[str],
        rename_function: Optional[Callable[[Path], str]]
    ) -> str:
        """Generate new filename based on template or custom function."""
        if rename_function:
            return rename_function(file_path)
        
        if rename_template:
            return self._apply_rename_template(file_path, original_name, rename_template)
        
        return original_name
    
    def _apply_rename_template(
        self,
        file_path: Path,
        original_name: str,
        template: str
    ) -> str:
        """
        Apply rename template with variable substitution.
        
        Supported variables:
        {name} - Original filename without extension
        {ext} - File extension
        {parent} - Parent directory name
        {counter} - Sequential counter
        {date} - Current date (YYYY-MM-DD)
        {time} - Current time (HH-MM-SS)
        {size} - File size in bytes
        {md5} - First 8 chars of MD5 hash
        """
        stem = file_path.stem
        extension = file_path.suffix
        parent_name = file_path.parent.name
        file_size = file_path.stat().st_size
        
        # Generate variables
        variables = {
            'name': stem,
            'ext': extension.lstrip('.'),
            'full_ext': extension,
            'parent': parent_name,
            'original': original_name,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H-%M-%S'),
            'datetime': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'size': file_size,
            'size_kb': file_size // 1024,
            'size_mb': file_size // (1024 * 1024),
            'md5': self._get_file_hash(file_path)
        }
        
        # Add counter if {counter} is in template
        if '{counter}' in template:
            if not hasattr(self, '_copy_counter'):
                self._copy_counter = 1
            variables['counter'] = self._copy_counter
            self._copy_counter += 1
        
        try:
            new_name = template.format(**variables)
            # Ensure we keep the original extension unless specified in template
            if '{ext}' not in template and '{full_ext}' not in template:
                new_name += extension
            return new_name
        except KeyError as e:
            print(f"Warning: Unknown variable in template: {e}. Using original name.")
            return original_name
    
    def _get_file_hash(self, file_path: Path, hash_length: int = 8) -> str:
        """Calculate MD5 hash of file."""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()[:hash_length]
        except:
            return "unknown"
    
    def _copy_single_file(
        self,
        source_path: Path,
        new_filename: str,
        overwrite: bool
    ) -> Dict:
        """Copy a single file with the new filename."""
        destination_path = self.destination_dir / new_filename
        
        result = {
            'source': str(source_path),
            'destination': str(destination_path),
            'original_name': source_path.name,
            'new_name': new_filename,
            'success': False,
            'error': None,
            'bytes_copied': 0
        }
        
        try:
            # Check if destination exists
            if destination_path.exists() and not overwrite:
                result['error'] = "File already exists and overwrite is False"
                self.stats['files_skipped'] += 1
                return result
            
            # Copy file
            shutil.copy2(source_path, destination_path)
            file_size = source_path.stat().st_size
            
            result['success'] = True
            result['bytes_copied'] = file_size
            
            self.stats['files_copied'] += 1
            self.stats['bytes_copied'] += file_size
            
        except Exception as e:
            result['error'] = str(e)
            self.stats['files_skipped'] += 1
        
        return result
    
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
            file_ext = Path(filename).suffix
            if not any(file_ext.lower() == ext.lower() for ext in file_extensions):
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
        if substring:  # Only check if substring is provided
            if use_regex:
                pattern = compare_substring if case_sensitive else compare_substring
                if not re.search(pattern, compare_name):
                    return False
            else:
                if compare_substring not in compare_name:
                    return False
        
        return True
    
    def copy_files_with_pattern_renaming(
        self,
        search_pattern: str,
        replace_pattern: str,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = False
    ) -> List[Dict]:
        """
        Copy files with pattern-based renaming.
        
        Args:
            search_pattern: Regex pattern to search for in filenames
            replace_pattern: Replacement pattern
            file_extensions: Filter by file extensions
            recursive: Search subdirectories recursively
        """
        copied_files = []
        search_path = "**/*" if recursive else "*"
        
        for file_path in self.source_dir.glob(search_path):
            if file_path.is_file():
                filename = file_path.name
                
                # Check file extension filter
                if file_extensions:
                    file_ext = Path(filename).suffix
                    if not any(file_ext.lower() == ext.lower() for ext in file_extensions):
                        continue
                
                # Apply pattern replacement
                new_filename = re.sub(search_pattern, replace_pattern, filename)
                
                if new_filename != filename:  # Only copy if name changes
                    result = self._copy_single_file(file_path, new_filename, overwrite=False)
                    copied_files.append(result)
        
        return copied_files
    
    def copy_files_with_sequential_renaming(
        self,
        base_name: str,
        start_number: int = 1,
        digits: int = 3,
        substring: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = False
    ) -> List[Dict]:
        """
        Copy files with sequential numbering.
        
        Args:
            base_name: Base name for files
            start_number: Starting number
            digits: Number of digits for padding
            substring: Filter by substring in filename
            file_extensions: Filter by file extensions
            recursive: Search subdirectories recursively
        """
        copied_files = []
        search_path = "**/*" if recursive else "*"
        file_list = []
        
        # First, collect all matching files
        for file_path in self.source_dir.glob(search_path):
            if file_path.is_file():
                filename = file_path.name
                
                # Apply filters
                if (substring and substring not in filename) or \
                   (file_extensions and Path(filename).suffix not in file_extensions):
                    continue
                
                file_list.append(file_path)
        
        # Sort files for consistent numbering
        file_list.sort()
        
        # Copy files with sequential names
        current_number = start_number
        for file_path in file_list:
            extension = file_path.suffix
            new_filename = f"{base_name}_{current_number:0{digits}d}{extension}"
            
            result = self._copy_single_file(file_path, new_filename, overwrite=False)
            copied_files.append(result)
            current_number += 1
        
        return copied_files
    
    def get_statistics(self) -> Dict:
        """Get copy statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset copy statistics."""
        self.stats = {
            'files_copied': 0,
            'bytes_copied': 0,
            'files_skipped': 0
        }
