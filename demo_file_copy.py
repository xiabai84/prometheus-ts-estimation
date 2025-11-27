from model.file_util import FileCopier

if __name__ == "__main__":
    # Initialize copier
    copier = FileCopier("/path/to/source", "/path/to/destination")
    
    # Example 1: Copy files with template renaming
    print("=== Example 1: Template Renaming ===")
    results1 = copier.copy_files_with_renaming(
        substring="report",
        file_extensions=[".pdf", ".docx"],
        rename_template="{name}_backup_{date}{full_ext}",
        recursive=True
    )
    
    for result in results1:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['original_name']} -> {result['new_name']}")
    
    # Example 2: Pattern-based renaming
    print("\n=== Example 2: Pattern Renaming ===")
    results2 = copier.copy_files_with_pattern_renaming(
        search_pattern=r"temp_(\w+)\.txt",
        replace_pattern=r"permanent_\1_backup.txt",
        file_extensions=[".txt"]
    )
    
    # Example 3: Sequential renaming
    print("\n=== Example 3: Sequential Renaming ===")
    results3 = copier.copy_files_with_sequential_renaming(
        base_name="image",
        start_number=1,
        digits=3,
        file_extensions=[".jpg", ".png"],
        substring="vacation"
    )
    
    # Example 4: Custom rename function
    print("\n=== Example 4: Custom Rename Function ===")
    def custom_renamer(file_path: Path) -> str:
        """Custom function to generate filename based on file size."""
        size_kb = file_path.stat().st_size // 1024
        return f"size_{size_kb}kb_{file_path.name}"
    
    results4 = copier.copy_files_with_renaming(
        substring="data",
        rename_function=custom_renamer,
        overwrite=True
    )
    
    # Example 5: Complex template with counter
    print("\n=== Example 5: Template with Counter ===")
    copier.reset_statistics()
    results5 = copier.copy_files_with_renaming(
        file_extensions=[".log"],
        rename_template="log_{date}_{counter:03d}.{ext}",
        recursive=True
    )
    
    # Print statistics
    stats = copier.get_statistics()
    print(f"\n=== Copy Statistics ===")
    print(f"Files copied: {stats['files_copied']}")
    print(f"Bytes copied: {stats['bytes_copied']}")
    print(f"Files skipped: {stats['files_skipped']}")