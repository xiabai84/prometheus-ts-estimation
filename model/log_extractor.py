import re
from collections import Counter, defaultdict
from typing import List, Dict, Set, Any, Optional, Tuple

class LogExtractor:
    """
    Optimized library for extracting exceptions and errors from text logs.
    Handles compound words, package names, stack traces, and keywords efficiently.
    """
    
    def __init__(self, 
                 include_package_names: bool = True, 
                 keywords: Optional[List[str]] = None):
        """
        Initialize the LogExtractor.
        
        Args:
            include_package_names: Whether to include full package paths
            keywords: List of keywords to search for
        """
        self.include_package_names = include_package_names
        self.keywords = keywords or []
        self._compiled_patterns = self._compile_optimized_patterns()
        self._cache = {}  # Simple cache for frequently used texts
    
    def _compile_optimized_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Compile optimized regex patterns for maximum performance."""
        patterns = []
        
        # Pre-compile all patterns for better performance
        base_patterns = [
            # Exception patterns - optimized with word boundaries and lookaheads
            (r'\b\w*?Error\b(?:\.\w+)*', 'error_with_package'),
            (r'\b\w*?Exception\b(?:\.\w+)*', 'exception_with_package'),
            (r'\bException\b(?:\.\w+)*', 'generic_exception_with_package'),
            
            # Stack trace patterns - optimized
            (r'at\s+[a-zA-Z0-9_.]+\.[a-zA-Z0-9_]+(?:Error|Exception)', 'stack_trace_full'),
            (r'at\s+[a-zA-Z0-9_.]+(?:Error|Exception)', 'stack_trace_simple'),
            (r'Caused by:\s*[a-zA-Z0-9_.]+\.[a-zA-Z0-9_]+(?:Error|Exception)', 'caused_by'),
            
            # Context patterns
            (r'Exception:\s*[^\n]+', 'exception_description'),
            (r'Error:\s*[^\n]+', 'error_description'),
            
            # Code patterns
            (r'raise\s+[a-zA-Z0-9_.]+', 'raise_stmt'),
            (r'catch\s*\(\s*[a-zA-Z0-9_.]+\s+\w+\)', 'catch_stmt'),
            (r'except\s+[a-zA-Z0-9_.]+', 'except_stmt'),
            (r'throws\s+[a-zA-Z0-9_.]+', 'throws_decl'),
            (r'throw new\s+[a-zA-Z0-9_.]+', 'throw_new'),
            (r'class\s+[a-zA-Z0-9_.]*Exception', 'class_def'),
        ]
        
        # Compile base patterns
        for pattern, pattern_type in base_patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                patterns.append((compiled, pattern_type))
            except re.error:
                continue  # Skip invalid patterns
        
        # Compile keyword patterns
        for keyword in self.keywords:
            try:
                # Use word boundaries for exact matching
                pattern = r'\b' + re.escape(keyword) + r'\b'
                compiled = re.compile(pattern, re.IGNORECASE)
                patterns.append((compiled, f'keyword_{keyword}'))
            except re.error:
                continue
        
        return patterns
    
    def extract_from_text(self, text: str, use_cache: bool = True) -> List[str]:
        """
        Optimized extraction of all exception and error findings from text.
        
        Args:
            text: Input text to analyze
            use_cache: Whether to use caching for repeated texts
            
        Returns:
            List of unique exception/error names found
        """
        if use_cache and text in self._cache:
            return self._cache[text]
        
        findings = set()
        
        # Single pass through all patterns
        for pattern, pattern_type in self._compiled_patterns:
            for match in pattern.finditer(text):
                matched_text = match.group()
                
                if pattern_type.startswith('keyword_'):
                    findings.add(matched_text.capitalize())
                elif pattern_type in ['exception_description', 'error_description']:
                    extracted = self._extract_from_description_optimized(matched_text)
                    findings.update(extracted)
                else:
                    name = self._extract_name_from_match(matched_text, pattern_type)
                    if name and self._is_valid_exception_or_error_name(name):
                        findings.add(name)
        
        # Add context-based findings
        context_findings = self._extract_from_context_optimized(text)
        findings.update(context_findings)
        
        result = sorted(list(findings))
        
        if use_cache:
            # Simple cache management (limit size)
            if len(self._cache) > 100:
                self._cache.clear()
            self._cache[text] = result
        
        return result
    
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
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
        
        return self.extract_from_text(content, use_cache=False)
    
    def analyze_frequency(self, text: str) -> Dict[str, int]:
        """
        Optimized frequency analysis using extraction-based counting.
        Most reliable method that handles compound words correctly.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with accurate frequency counts
        """
        frequency_counter = Counter()
        
        # Single pass through text with all patterns
        for pattern, pattern_type in self._compiled_patterns:
            for match in pattern.finditer(text):
                matched_text = match.group()
                
                if pattern_type.startswith('keyword_'):
                    frequency_counter[matched_text.capitalize()] += 1
                elif pattern_type in ['exception_description', 'error_description']:
                    extracted = self._extract_from_description_optimized(matched_text)
                    for name in extracted:
                        frequency_counter[name] += 1
                else:
                    name = self._extract_name_from_match(matched_text, pattern_type)
                    if name and self._is_valid_exception_or_error_name(name):
                        frequency_counter[name] += 1
        
        # Add context findings with counting
        context_findings = self._count_context_findings(text)
        frequency_counter.update(context_findings)
        
        return dict(frequency_counter.most_common())
    
    def analyze_frequency_advanced(self, text: str, 
                                 min_count: int = 1,
                                 group_similar: bool = False) -> Dict[str, Any]:
        """
        Advanced frequency analysis with additional features.
        
        Args:
            text: Input text to analyze
            min_count: Minimum frequency to include
            group_similar: Group similar exceptions (e.g., *Error, *Exception)
            
        Returns:
            Dictionary with detailed frequency information
        """
        basic_frequency = self.analyze_frequency(text)
        
        # Apply minimum count filter
        if min_count > 1:
            basic_frequency = {k: v for k, v in basic_frequency.items() if v >= min_count}
        
        if not group_similar:
            return basic_frequency
        
        # Group similar exceptions
        grouped = defaultdict(int)
        for name, count in basic_frequency.items():
            if name.endswith('Error'):
                grouped['*Error'] += count
            elif name.endswith('Exception'):
                grouped['*Exception'] += count
            else:
                grouped[name] = count
        
        return dict(sorted(grouped.items(), key=lambda x: x[1], reverse=True))
    
    def analyze_frequency_by_line(self, text: str) -> Dict[str, List[int]]:
        """
        Analyze frequency with line number information.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with findings and their line numbers
        """
        line_occurrences = defaultdict(list)
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            findings = self.extract_from_text(line, use_cache=False)
            for finding in findings:
                line_occurrences[finding].append(line_num)
        
        return dict(line_occurrences)
    
    def extract_with_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract exceptions with comprehensive metadata.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing findings and metadata
        """
        findings = self.extract_from_text(text)
        frequency = self.analyze_frequency(text)
        line_occurrences = self.analyze_frequency_by_line(text)
        categorized = self._categorize_findings_optimized(findings)
        
        return {
            'findings': findings,
            'frequency': frequency,
            'line_occurrences': line_occurrences,
            'categorized': categorized,
            'summary': {
                'total_findings': len(findings),
                'total_occurrences': sum(frequency.values()),
                'exceptions_count': len(categorized['exceptions']),
                'errors_count': len(categorized['errors']),
                'packages_count': len(categorized['packages']),
                'keywords_count': len(categorized['keywords'])
            }
        }
    
    def _extract_name_from_match(self, matched_text: str, pattern_type: str) -> str:
        """Extract clean name from matched text based on pattern type."""
        if pattern_type in ['stack_trace_full', 'caused_by']:
            # Extract just the exception name from full stack trace lines
            parts = matched_text.split('.')
            for part in reversed(parts):
                if any(keyword in part for keyword in ['Error', 'Exception']):
                    return part
            return matched_text.split('.')[-1]
        elif pattern_type in ['raise_stmt', 'catch_stmt', 'except_stmt', 'throws_decl', 'throw_new']:
            # Extract the exception name from code statements
            parts = matched_text.split()
            return parts[-1] if parts else matched_text
        else:
            # For most cases, return the matched text as-is
            return matched_text.strip()
    
    def _extract_from_description_optimized(self, description: str) -> Set[str]:
        """Optimized extraction from exception descriptions."""
        # Look for exception-like patterns in descriptions
        findings = set()
        words = re.findall(r'[A-Z][a-zA-Z]*(?:Error|Exception)', description)
        
        for word in words:
            if self._is_valid_exception_or_error_name(word):
                findings.add(word)
        
        return findings
    
    def _extract_from_context_optimized(self, text: str) -> Set[str]:
        """Optimized context-based extraction."""
        findings = set()
        
        # Look for package names if enabled
        if self.include_package_names:
            package_pattern = re.compile(r'[a-zA-Z0-9_.]+\.[A-Z][a-zA-Z]*(?:Error|Exception)')
            for match in package_pattern.finditer(text):
                findings.add(match.group())
        
        return findings
    
    def _count_context_findings(self, text: str) -> Counter:
        """Count context findings directly without full extraction."""
        counter = Counter()
        
        if self.include_package_names:
            package_pattern = re.compile(r'[a-zA-Z0-9_.]+\.[A-Z][a-zA-Z]*(?:Error|Exception)')
            for match in package_pattern.finditer(text):
                counter[match.group()] += 1
        
        return counter
    
    def _categorize_findings_optimized(self, findings: List[str]) -> Dict[str, List[str]]:
        """Optimized categorization of findings."""
        categories = {
            'exceptions': [],
            'errors': [],
            'packages': [],
            'keywords': []
        }
        
        for finding in findings:
            finding_lower = finding.lower()
            
            if '.' in finding:
                categories['packages'].append(finding)
            elif finding.lower() in [kw.lower() for kw in self.keywords]:
                categories['keywords'].append(finding)
            elif 'exception' in finding_lower:
                categories['exceptions'].append(finding)
            elif 'error' in finding_lower:
                categories['errors'].append(finding)
            else:
                categories['keywords'].append(finding)
        
        # Sort all categories
        for category in categories:
            categories[category].sort()
        
        return categories
    
    def _is_valid_exception_or_error_name(self, name: str) -> bool:
        """Optimized validation of exception/error names."""
        if not name or len(name) < 3:
            return False
        
        # Quick exclude common English words
        exclude_words = {'the', 'and', 'or', 'but', 'for', 'with', 'from'}
        if name.lower() in exclude_words:
            return False
        
        # Allow configured keywords
        if name.lower() in [kw.lower() for kw in self.keywords]:
            return True
        
        # Quick pattern checks
        if (name.endswith(('Error', 'Exception')) or
            name in ('Exception', 'Error') or
            any(keyword in name.lower() for keyword in ['error', 'exception', 'fail'])):
            return True
        
        # CamelCase validation
        if re.match(r'^[A-Z][a-zA-Z]*(Error|Exception|Fail)', name):
            return True
        
        return False
    
    # Utility methods
    def update_keywords(self, new_keywords: List[str]) -> None:
        """Update keywords and recompile patterns."""
        self.keywords = new_keywords
        self._compiled_patterns = self._compile_optimized_patterns()
        self._cache.clear()  # Clear cache when patterns change
    
    def add_keyword(self, keyword: str) -> None:
        """Add a single keyword."""
        if keyword not in self.keywords:
            self.keywords.append(keyword)
            self._compiled_patterns = self._compile_optimized_patterns()
            self._cache.clear()
    
    def remove_keyword(self, keyword: str) -> None:
        """Remove a keyword."""
        if keyword in self.keywords:
            self.keywords.remove(keyword)
            self._compiled_patterns = self._compile_optimized_patterns()
            self._cache.clear()
    
    def get_keywords(self) -> List[str]:
        """Get current keywords."""
        return self.keywords.copy()
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
