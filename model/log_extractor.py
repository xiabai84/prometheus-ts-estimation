import re
from typing import List, Dict, Set, Any, Optional

class LogExtractor:
    """
    A comprehensive library for extracting exceptions and errors from text logs.
    Handles package names, stack traces, keywords, and various exception formats.
    """
    
    def __init__(self, 
                 include_package_names: bool = True, 
                 include_keywords: bool = True,
                 custom_keywords: Optional[List[str]] = None):
        """
        Initialize the LogExtractor.
        
        Args:
            include_package_names: Whether to include full package paths
            include_keywords: Whether to include generic 'exception'/'error' keywords
            custom_keywords: Additional custom keywords to search for
        """
        self.include_package_names = include_package_names
        self.include_keywords = include_keywords
        self.custom_keywords = custom_keywords or []
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
        
        # Add keyword patterns if enabled
        if self.include_keywords:
            patterns.extend([
                (r'\b(exception)\b', 'keyword_exception'),
                (r'\b(error)\b', 'keyword_error'),
            ])
        
        # Add custom keyword patterns
        for keyword in self.custom_keywords:
            # Escape special regex characters in custom keywords
            escaped_keyword = re.escape(keyword.lower())
            patterns.append(
                (fr'\b({escaped_keyword})\b', f'custom_keyword_{keyword}')
            )
        
        return patterns
    
    def update_keywords(self, new_keywords: List[str]) -> None:
        """
        Update the custom keywords and recompile patterns.
        
        Args:
            new_keywords: New list of custom keywords to search for
        """
        self.custom_keywords = new_keywords
        self._patterns = self._compile_patterns()
    
    def add_keyword(self, keyword: str) -> None:
        """
        Add a single keyword to the custom keywords list.
        
        Args:
            keyword: Keyword to add
        """
        if keyword not in self.custom_keywords:
            self.custom_keywords.append(keyword)
            self._patterns = self._compile_patterns()
    
    def remove_keyword(self, keyword: str) -> None:
        """
        Remove a keyword from the custom keywords list.
        
        Args:
            keyword: Keyword to remove
        """
        if keyword in self.custom_keywords:
            self.custom_keywords.remove(keyword)
            self._patterns = self._compile_patterns()
    
    def get_keywords(self) -> List[str]:
        """
        Get the current list of custom keywords.
        
        Returns:
            List of custom keywords
        """
        return self.custom_keywords.copy()
    
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
                if pattern_type.startswith('keyword_') or pattern_type.startswith('custom_keyword_'):
                    # For keyword matches, add the keyword itself
                    findings.add(match.capitalize())
                elif pattern_type in ['exception_description', 'error_description']:
                    extracted = self._extract_from_description(match)
                    findings.update(extracted)
                else:
                    name = match.strip()
                    if self._is_valid_exception_or_error_name(name):
                        findings.add(name)
        
        # Additional processing
        keyword_findings = self._extract_from_keyword_context(text)
        findings.update(keyword_findings)
        
        # Add custom keywords that are found in text
        custom_keyword_findings = self._extract_custom_keywords(text)
        findings.update(custom_keyword_findings)
        
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
                'keywords_count': len(categorized['keywords']),
                'custom_keywords_count': len(categorized['custom_keywords'])
            }
        }
    
    def analyze_frequency(self, text: str) -> Dict[str, int]:
        """
        Analyze frequency of each exception/error in the text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with exception/error names and their counts
        """
        findings = self.extract_from_text(text)
        frequency = {}
        
        for finding in findings:
            # Count occurrences of each finding
            if '.' in finding:
                # For package names, use simple name for counting
                simple_name = finding.split('.')[-1]
                count = len(re.findall(re.escape(finding), text))
                frequency[simple_name] = frequency.get(simple_name, 0) + count
            else:
                count = len(re.findall(re.escape(finding), text))
                frequency[finding] = count
        
        return dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))
    
    def _extract_from_description(self, description: str) -> Set[str]:
        """Extract potential exception names from descriptions."""
        findings = set()
        exception_matches = re.findall(r'([A-Z][a-zA-Z]*(?:Error|Exception))', description)
        
        for match in exception_matches:
            if self._is_valid_exception_or_error_name(match):
                findings.add(match)
        
        return findings
    
    def _extract_from_keyword_context(self, text: str) -> Set[str]:
        """Extract findings from lines containing exception/error keywords."""
        findings = set()
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Generic keyword detection
            if self.include_keywords:
                if ('exception' in line_lower and 
                    not re.search(r'[A-Z][a-zA-Z]*(?:Error|Exception)', line)):
                    findings.add('Exception')
                if ('error' in line_lower and 
                    not re.search(r'[A-Z][a-zA-Z]*(?:Error|Exception)', line)):
                    findings.add('Error')
            
            # Extract full package paths
            if self.include_package_names:
                package_matches = re.findall(
                    r'([a-zA-Z0-9_.]+\.[A-Z][a-zA-Z]*(?:Error|Exception))', line
                )
                for match in package_matches:
                    if self._is_valid_exception_or_error_name(match):
                        findings.add(match)
        
        return findings
    
    def _extract_custom_keywords(self, text: str) -> Set[str]:
        """Extract custom keywords from text."""
        findings = set()
        
        for keyword in self.custom_keywords:
            # Case-insensitive search for custom keywords
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                findings.add(keyword.capitalize())
        
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
        
        # Allow custom keywords
        if name.lower() in [kw.lower() for kw in self.custom_keywords]:
            return True
        
        # Allow package names and generic keywords
        if name in ['Exception', 'Error']:
            return self.include_keywords
        
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
        custom_keywords = []
        
        for finding in findings:
            finding_lower = finding.lower()
            
            # Check if it's a custom keyword first
            if finding.lower() in [kw.lower() for kw in self.custom_keywords]:
                custom_keywords.append(finding)
            elif '.' in finding:
                packages.append(finding)
            elif finding in ['Exception', 'Error']:
                keywords.append(finding)
            elif 'exception' in finding_lower:
                exceptions.append(finding)
            elif 'error' in finding_lower:
                errors.append(finding)
            else:
                keywords.append(finding)
        
        return {
            'exceptions': sorted(exceptions),
            'errors': sorted(errors),
            'packages': sorted(packages),
            'keywords': sorted(keywords),
            'custom_keywords': sorted(custom_keywords)
        }
    
    def print_report(self, text: str, title: str = "Exception Analysis Report"):
        """Print a formatted report of the analysis."""
        result = self.extract_with_metadata(text)
        
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
        
        if self.include_keywords:
            print(f"\nKeywords ({result['summary']['keywords_count']}):")
            for kw in result['categorized']['keywords']:
                print(f"  - {kw}")
        
        if self.custom_keywords:
            print(f"\nCustom Keywords ({result['summary']['custom_keywords_count']}):")
            for ckw in result['categorized']['custom_keywords']:
                print(f"  - {ckw}")
        
        # Frequency analysis
        frequency = self.analyze_frequency(text)
        if frequency:
            print(f"\nFREQUENCY ANALYSIS (Top 10):")
            for finding, count in list(frequency.items())[:10]:
                print(f"  - {finding}: {count} occurrences")
