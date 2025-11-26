import re
from typing import List, Dict, Set, Any, Optional

class LogExtractor:
    """
    A comprehensive library for extracting exceptions and errors from text logs.
    Handles package names, stack traces, keywords, and various exception formats.
    """
    
    def __init__(self, 
                 include_package_names: bool = True, 
                 keywords: Optional[List[str]] = None):
        """
        Initialize the LogExtractor.
        
        Args:
            include_package_names: Whether to include full package paths
            keywords: List of keywords to search for (includes both exception keywords and custom keywords)
        """
        self.include_package_names = include_package_names
        self.keywords = keywords or []
        self._patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> List[tuple]:
        """Compile all regex patterns for exception extraction."""
        patterns = [
            # Exception patterns with package names
            (r'(\b\w*?Error\b(?:\.\w+)*)', 'error_with_package'),
            (r'(\b\w*?Exception\b(?:\.\w+)*)', 'exception_with_package'),
            (r'(\bException\b(?:\.\w+)*)', 'generic_exception_with_package'),
            
            # Stack trace patterns
            (r'at\s+([a-zA-Z0-9_.]+\.[a-zA-Z0-9_]+(?:Error|Exception))', 'stack_trace_full'),
            (r'at\s+([a-zA-Z0-9_.]+(?:Error|Exception))', 'stack_trace_simple'),
            (r'Caused by:\s*([a-zA-Z0-9_.]+\.[a-zA-Z0-9_]+(?:Error|Exception))', 'caused_by'),
            
            # Exception context patterns
            (r'Exception:\s*([^\n]+)', 'exception_description'),
            (r'Error:\s*([^\n]+)', 'error_description'),
            
            # Basic patterns
            (r'raise\s+([a-zA-Z0-9_.]+)', 'raise_stmt'),
            (r'catch\s*\(([a-zA-Z0-9_.]+)\s+\w+\)', 'catch_stmt'),
            (r'except\s+([a-zA-Z0-9_.]+)', 'except_stmt'),
            (r'throws\s+([a-zA-Z0-9_.]+)', 'throws_decl'),
            (r'throw new\s+([a-zA-Z0-9_.]+)', 'throw_new'),
            (r'class\s+([a-zA-Z0-9_.]*Exception)', 'class_def'),
        ]
        
        # Add keyword patterns for all keywords
        for keyword in self.keywords:
            # Escape special regex characters in keywords
            escaped_keyword = re.escape(keyword.lower())
            patterns.append(
                (fr'\b({escaped_keyword})\b', f'keyword_{keyword}')
            )
        
        return patterns
    
    def update_keywords(self, new_keywords: List[str]) -> None:
        """
        Update the keywords and recompile patterns.
        
        Args:
            new_keywords: New list of keywords to search for
        """
        self.keywords = new_keywords
        self._patterns = self._compile_patterns()
    
    def add_keyword(self, keyword: str) -> None:
        """
        Add a single keyword to the keywords list.
        
        Args:
            keyword: Keyword to add
        """
        if keyword not in self.keywords:
            self.keywords.append(keyword)
            self._patterns = self._compile_patterns()
    
    def remove_keyword(self, keyword: str) -> None:
        """
        Remove a keyword from the keywords list.
        
        Args:
            keyword: Keyword to remove
        """
        if keyword in self.keywords:
            self.keywords.remove(keyword)
            self._patterns = self._compile_patterns()
    
    def get_keywords(self) -> List[str]:
        """
        Get the current list of keywords.
        
        Returns:
            List of keywords
        """
        return self.keywords.copy()
    
    def has_keywords(self) -> bool:
        """
        Check if any keywords are configured.
        
        Returns:
            True if keywords are configured, False otherwise
        """
        return len(self.keywords) > 0
    
    def extract_from_text(self, text: str) -> List[str]:
        """
        Extract all exception and error findings from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of unique exception/error names found
        """
        findings = set()
        
        for pattern, pattern_type in self._patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if pattern_type.startswith('keyword_'):
                    # For keyword matches, add the keyword itself
                    findings.add(match.capitalize())
                elif pattern_type in ['exception_description', 'error_description']:
                    extracted = self._extract_from_description(match)
                    findings.update(extracted)
                else:
                    name = match.strip()
                    if self._is_valid_exception_or_error_name(name):
                        findings.add(name)
        
        # Additional processing for context-based extraction
        context_findings = self._extract_from_context(text)
        findings.update(context_findings)
        
        return sorted(list(findings))
    
    def extract_from_file(self, file_path: str, encoding: str = 'utf-8') -> List[str]:
        """
        Extract exceptions and errors from a file.
        
        Args:
            file_path: Path to the file to analyze
            encoding: File encoding (default: utf-8)
            
        Returns:
            List of unique exception/error names found
        """
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
        
        return self.extract_from_text(content)
    
    def extract_with_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract exceptions with additional metadata.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing findings and metadata
        """
        findings = self.extract_from_text(text)
        categorized = self._categorize_findings_detailed(findings)
        
        return {
            'findings': findings,
            'categorized': categorized,
            'summary': {
                'total_findings': len(findings),
                'exceptions_count': len(categorized['exceptions']),
                'errors_count': len(categorized['errors']),
                'packages_count': len(categorized['packages']),
                'keywords_count': len(categorized['keywords'])
            }
        }
    
    def analyze_frequency(self, text: str) -> Dict[str, int]:
        """
        Analyze frequency of each exception/error/keyword in the text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with exception/error names and their counts
        """
        frequency = {}
        
        # Count exceptions and errors
        for pattern, pattern_type in self._patterns:
            if pattern_type.startswith('keyword_'):
                # Handle keywords separately
                continue
            
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if pattern_type in ['exception_description', 'error_description']:
                    # Extract and count from descriptions
                    extracted = self._extract_from_description(match)
                    for name in extracted:
                        count = len(re.findall(re.escape(name), text))
                        frequency[name] = frequency.get(name, 0) + count
                else:
                    name = match.strip()
                    if self._is_valid_exception_or_error_name(name):
                        if '.' in name:
                            # For package names, use simple name for counting
                            simple_name = name.split('.')[-1]
                            count = len(re.findall(re.escape(name), text))
                            frequency[simple_name] = frequency.get(simple_name, 0) + count
                        else:
                            count = len(re.findall(re.escape(name), text))
                            frequency[name] = frequency.get(name, 0) + count
        
        # Count keywords with exact word matching
        for keyword in self.keywords:
            # Use word boundaries to count exact matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            count = len(re.findall(pattern, text, re.IGNORECASE))
            if count > 0:
                frequency[keyword.capitalize()] = count
        
        return dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))
    
    def analyze_frequency_detailed(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Analyze frequency with detailed information including categories.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detailed frequency information
        """
        basic_frequency = self.analyze_frequency(text)
        findings = self.extract_from_text(text)
        categorized = self._categorize_findings_detailed(findings)
        
        detailed_frequency = {}
        total_words = len(text.split())
        
        for item, count in basic_frequency.items():
            category = self._categorize_single_item(item, categorized)
            detailed_frequency[item] = {
                'count': count,
                'category': category
            }
        
        return detailed_frequency
    
    def _categorize_single_item(self, item: str, categorized: Dict[str, List[str]]) -> str:
        """Categorize a single item."""
        if item in categorized['keywords']:
            return 'keyword'
        elif item in categorized['exceptions']:
            return 'exception'
        elif item in categorized['errors']:
            return 'error'
        elif item in categorized['packages']:
            return 'package'
        else:
            return 'unknown'
    
    def _extract_from_description(self, description: str) -> Set[str]:
        """Extract potential exception names from descriptions."""
        findings = set()
        exception_matches = re.findall(r'([A-Z][a-zA-Z]*(?:Error|Exception))', description)
        
        for match in exception_matches:
            if self._is_valid_exception_or_error_name(match):
                findings.add(match)
        
        return findings
    
    def _extract_from_context(self, text: str) -> Set[str]:
        """Extract findings from context including keywords and package names."""
        findings = set()
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Extract keywords from context
            for keyword in self.keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', line, re.IGNORECASE):
                    findings.add(keyword.capitalize())
            
            # Extract full package paths
            if self.include_package_names:
                package_matches = re.findall(
                    r'([a-zA-Z0-9_.]+\.[A-Z][a-zA-Z]*(?:Error|Exception))', line
                )
                for match in package_matches:
                    if self._is_valid_exception_or_error_name(match):
                        findings.add(match)
        
        return findings
    
    def _is_valid_exception_or_error_name(self, name: str) -> bool:
        """Check if the name is a valid exception or error name."""
        if not name or len(name) < 3:
            return False
        
        exclude_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'with', 'from', 'by', 'as', 'is', 'are', 'was', 'were',
            'this', 'that', 'these', 'those', 'what', 'when', 'where'
        }
        
        if name.lower() in exclude_words:
            return False
        
        # Allow configured keywords
        if name.lower() in [kw.lower() for kw in self.keywords]:
            return True
        
        # Allow package names and generic exception/error patterns
        valid_keywords = ['error', 'exception', 'throw', 'catch', 'raise', 'fail']
        if any(keyword in name.lower() for keyword in valid_keywords):
            return True
        
        # Pattern-based validation
        if ('.' in name or
            name.endswith('Error') or 
            name.endswith('Exception') or 
            name == 'Exception' or 
            name == 'Error' or
            name.startswith('Err') or 
            'Fail' in name):
            return True
        
        # CamelCase validation
        if re.match(r'^[A-Z][a-zA-Z]*(Error|Exception|Fail)', name):
            return True
        
        return False
    
    def _categorize_findings_detailed(self, findings: List[str]) -> Dict[str, List[str]]:
        """Categorize findings into detailed groups."""
        exceptions = []
        errors = []
        packages = []
        keywords = []
        
        for finding in findings:
            finding_lower = finding.lower()
            
            # Check if it's a keyword first
            if finding.lower() in [kw.lower() for kw in self.keywords]:
                keywords.append(finding)
            elif '.' in finding:
                packages.append(finding)
            elif 'exception' in finding_lower:
                exceptions.append(finding)
            elif 'error' in finding_lower:
                errors.append(finding)
            else:
                # Default to keyword for unmatched items
                keywords.append(finding)
        
        return {
            'exceptions': sorted(exceptions),
            'errors': sorted(errors),
            'packages': sorted(packages),
            'keywords': sorted(keywords)
        }
    
    def print_report(self, text: str, title: str = "Exception Analysis Report"):
        """Print a formatted report of the analysis."""
        result = self.extract_with_metadata(text)
        frequency = self.analyze_frequency(text)
        detailed_frequency = self.analyze_frequency_detailed(text)
        
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        print(f"\nALL FINDINGS ({result['summary']['total_findings']} total):")
        for i, finding in enumerate(result['findings'], 1):
            print(f"  {i:2d}. {finding}")
        
        print(f"\nDETAILED CATEGORIZATION:")
        print(f"Exceptions ({result['summary']['exceptions_count']}):")
        for exc in result['categorized']['exceptions']:
            print(f"  - {exc}")
        
        print(f"\nErrors ({result['summary']['errors_count']}):")
        for err in result['categorized']['errors']:
            print(f"  - {err}")
        
        if self.include_package_names:
            print(f"\nPackages ({result['summary']['packages_count']}):")
            for pkg in result['categorized']['packages']:
                print(f"  - {pkg}")
        
        if self.has_keywords():
            print(f"\nKeywords ({result['summary']['keywords_count']}):")
            for kw in result['categorized']['keywords']:
                print(f"  - {kw}")
        
        # Enhanced frequency analysis
        if frequency:
            print(f"\nFREQUENCY ANALYSIS (All Items):")
            for finding, count in frequency.items():
                category = detailed_frequency.get(finding, {}).get('category', 'unknown')
                print(f"  - {finding} ({category}): {count} occurrences")
        
        # Top items by category
        if frequency:
            print(f"\nTOP ITEMS BY CATEGORY:")
            categories = {}
            for finding, info in detailed_frequency.items():
                category = info['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append((finding, info['count']))
            
            for category, items in categories.items():
                top_items = sorted(items, key=lambda x: x[1], reverse=True)[:3]
                if top_items:
                    items_str = ', '.join([f'{item[0]}({item[1]})' for item in top_items])
                    print(f"  {category.capitalize()}: {items_str}")
