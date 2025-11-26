import re
from collections import Counter, defaultdict
from typing import List, Dict, Set, Any, Optional, Tuple

class LogExtractor:
    """
    Fixed library for extracting exceptions and errors from text logs.
    Prevents overcounting by using pattern prioritization and match deduplication.
    """
    
    def __init__(self, 
                 include_package_names: bool = True, 
                 keywords: Optional[List[str]] = None,
                 prefer_simple_names: bool = True):
        """
        Initialize the LogExtractor.
        
        Args:
            include_package_names: Whether to include full package paths
            keywords: List of keywords to search for
            prefer_simple_names: Whether to prefer simple class names over full package names
        """
        self.include_package_names = include_package_names
        self.keywords = keywords or []
        self.prefer_simple_names = prefer_simple_names
        self._compiled_patterns = self._compile_prioritized_patterns()
        self._cache = {}
    
    def _compile_prioritized_patterns(self) -> List[Tuple[re.Pattern, str, int]]:
        """
        Compile patterns with priorities to prevent overmatching.
        Higher priority patterns are processed first.
        """
        patterns = []
        
        # Define patterns with priorities (higher number = higher priority)
        base_patterns = [
            # Highest priority: Full package names in stack traces (most specific)
            (r'at\s+(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))', 'stack_trace_full', 100),
            (r'Caused by:\s*(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))', 'caused_by', 100),
            
            # High priority: Code statements
            (r'(?:raise|throw)\s+([a-zA-Z0-9_.]+(?:Error|Exception))', 'raise_throw', 90),
            (r'(?:throws|catch|except)\s+([a-zA-Z0-9_.]+(?:Error|Exception))', 'declaration', 90),
            
            # Medium priority: Full package names in messages
            (r'\b(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))(?:\s*:|)', 'package_with_colon', 80),
            (r'\b(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))\b', 'package_general', 70),
            
            # Lower priority: Simple exception names
            (r'\b([A-Z][a-zA-Z0-9_]*(?:Error|Exception))(?:\s*:|)', 'simple_with_colon', 60),
            (r'\b([A-Z][a-zA-Z0-9_]*(?:Error|Exception))\b', 'simple_general', 50),
            
            # Context patterns (lowest priority)
            (r'(?:Exception|Error):\s*[^\n]+', 'exception_description', 40),
        ]
        
        # Compile base patterns
        for pattern, pattern_type, priority in base_patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                patterns.append((compiled, pattern_type, priority))
            except re.error as e:
                print(f"Warning: Failed to compile pattern '{pattern}': {e}")
                continue
        
        # Add keyword patterns with medium priority
        for keyword in self.keywords:
            try:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                compiled = re.compile(pattern, re.IGNORECASE)
                patterns.append((compiled, f'keyword_{keyword}', 75))
            except re.error:
                continue
        
        # Sort by priority (highest first)
        patterns.sort(key=lambda x: x[2], reverse=True)
        return patterns
    
    def extract_from_text(self, text: str, use_cache: bool = True) -> List[str]:
        """
        Extract exceptions without duplicates using pattern prioritization.
        
        Args:
            text: Input text to analyze
            use_cache: Whether to use caching
            
        Returns:
            List of unique exception/error names found
        """
        if use_cache and text in self._cache:
            return self._cache[text]
        
        findings = set()
        processed_positions = set()  # Track processed character positions
        
        # Process patterns in priority order
        for pattern, pattern_type, priority in self._compiled_patterns:
            for match in pattern.finditer(text):
                start_pos, end_pos = match.span()
                
                # Skip if this position was already processed by a higher priority pattern
                if self._is_position_processed(start_pos, end_pos, processed_positions):
                    continue
                
                # Mark this position as processed
                processed_positions.add((start_pos, end_pos))
                
                if pattern_type.startswith('keyword_'):
                    findings.add(match.group().capitalize())
                elif pattern_type == 'exception_description':
                    extracted = self._extract_from_description_safe(match.group())
                    findings.update(extracted)
                else:
                    names = self._extract_names_from_match(match, pattern_type)
                    for name in names:
                        if name and self._is_valid_exception_or_error_name(name):
                            findings.add(name)
        
        result = sorted(list(findings))
        
        if use_cache:
            if len(self._cache) > 100:
                self._cache.clear()
            self._cache[text] = result
        
        return result
    
    def _is_position_processed(self, start: int, end: int, processed_positions: Set[tuple]) -> bool:
        """
        Check if a text position range has already been processed.
        """
        for proc_start, proc_end in processed_positions:
            if start >= proc_start and end <= proc_end:
                return True
            # Allow some overlap but not complete containment
            if start < proc_end and end > proc_start:
                return True
        return False
    
    def analyze_frequency(self, text: str) -> Dict[str, int]:
        """
        Accurate frequency analysis without overcounting.
        Uses single-pass with position tracking.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with accurate frequency counts
        """
        frequency_counter = Counter()
        processed_positions = set()
        
        # Single pass through all patterns in priority order
        for pattern, pattern_type, priority in self._compiled_patterns:
            for match in pattern.finditer(text):
                start_pos, end_pos = match.span()
                
                # Skip if this position was already processed
                if self._is_position_processed(start_pos, end_pos, processed_positions):
                    continue
                
                # Mark this position as processed
                processed_positions.add((start_pos, end_pos))
                
                if pattern_type.startswith('keyword_'):
                    frequency_counter[match.group().capitalize()] += 1
                elif pattern_type == 'exception_description':
                    extracted = self._extract_from_description_safe(match.group())
                    for name in extracted:
                        frequency_counter[name] += 1
                else:
                    names = self._extract_names_from_match(match, pattern_type)
                    for name in names:
                        if name and self._is_valid_exception_or_error_name(name):
                            if self.prefer_simple_names and '.' in name:
                                simple_name = name.split('.')[-1]
                                frequency_counter[simple_name] += 1
                                if self.include_package_names:
                                    frequency_counter[name] += 1
                            else:
                                frequency_counter[name] += 1
        
        return dict(frequency_counter.most_common())
    
    def analyze_frequency_exact(self, text: str) -> Dict[str, int]:
        """
        Exact frequency analysis using extraction-based counting.
        Most reliable method - counts based on actual extraction results.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with exact frequency counts
        """
        # Use the extraction results to count frequencies
        findings_with_context = self._extract_findings_with_context(text)
        frequency_counter = Counter()
        
        for finding, context in findings_with_context:
            if self.prefer_simple_names and '.' in finding:
                simple_name = finding.split('.')[-1]
                frequency_counter[simple_name] += 1
                if self.include_package_names:
                    frequency_counter[finding] += 1
            else:
                frequency_counter[finding] += 1
        
        return dict(frequency_counter.most_common())
    
    def _extract_findings_with_context(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract findings with their context for accurate counting.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of tuples (finding, context)
        """
        findings_with_context = []
        processed_positions = set()
        
        for pattern, pattern_type, priority in self._compiled_patterns:
            for match in pattern.finditer(text):
                start_pos, end_pos = match.span()
                
                if self._is_position_processed(start_pos, end_pos, processed_positions):
                    continue
                
                processed_positions.add((start_pos, end_pos))
                
                if pattern_type.startswith('keyword_'):
                    findings_with_context.append((match.group().capitalize(), pattern_type))
                elif pattern_type == 'exception_description':
                    extracted = self._extract_from_description_safe(match.group())
                    for name in extracted:
                        findings_with_context.append((name, pattern_type))
                else:
                    names = self._extract_names_from_match(match, pattern_type)
                    for name in names:
                        if name and self._is_valid_exception_or_error_name(name):
                            findings_with_context.append((name, pattern_type))
        
        return findings_with_context
    
    def _extract_names_from_match(self, match: re.Match, pattern_type: str) -> List[str]:
        """Extract names from match based on pattern type."""
        matched_text = match.group()
        groups = match.groups()
        
        if pattern_type in ['stack_trace_full', 'caused_by']:
            # Full package name from stack traces
            full_name = groups[0] if groups else matched_text
            return self._get_name_variants(full_name)
        
        elif pattern_type in ['raise_throw', 'declaration']:
            # Exception names from code statements
            exception_name = groups[0] if groups else matched_text.split()[-1]
            return self._get_name_variants(exception_name)
        
        elif pattern_type in ['package_with_colon', 'package_general']:
            # Package names from general text
            full_name = groups[0] if groups else matched_text
            return self._get_name_variants(full_name)
        
        elif pattern_type in ['simple_with_colon', 'simple_general']:
            # Simple names from general text
            simple_name = groups[0] if groups else matched_text
            return [simple_name]
        
        return [matched_text]
    
    def _get_name_variants(self, name: str) -> List[str]:
        """Get both full and simple name variants."""
        if '.' in name:
            full_name = name
            simple_name = name.split('.')[-1]
            if self.prefer_simple_names:
                return [simple_name, full_name] if self.include_package_names else [simple_name]
            else:
                return [full_name, simple_name]
        else:
            return [name]
    
    def _extract_from_description_safe(self, description: str) -> Set[str]:
        """Safe extraction from descriptions without overcounting."""
        findings = set()
        
        # Only look for clear exception patterns in descriptions
        exception_pattern = re.compile(r'\b([A-Z][a-zA-Z0-9_]*(?:Error|Exception))\b')
        package_pattern = re.compile(r'\b(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))\b')
        
        for match in exception_pattern.finditer(description):
            name = match.group(1)
            if self._is_valid_exception_or_error_name(name):
                findings.add(name)
        
        for match in package_pattern.finditer(description):
            name = match.group(1)
            if self._is_valid_exception_or_error_name(name):
                if self.prefer_simple_names:
                    findings.add(name.split('.')[-1])
                else:
                    findings.add(name)
        
        return findings
    
    def _is_valid_exception_or_error_name(self, name: str) -> bool:
        """Validation that handles both package and simple names."""
        if not name or len(name) < 3:
            return False
        
        # Extract simple name for validation
        simple_name = name.split('.')[-1] if '.' in name else name
        
        # Quick exclude common English words
        exclude_words = {'the', 'and', 'or', 'but', 'for', 'with', 'from', 'this', 'that'}
        if simple_name.lower() in exclude_words:
            return False
        
        # Allow configured keywords
        if simple_name.lower() in [kw.lower() for kw in self.keywords]:
            return True
        
        # Pattern checks on simple name
        if (simple_name.endswith(('Error', 'Exception')) or
            simple_name in ('Exception', 'Error') or
            any(keyword in simple_name.lower() for keyword in ['error', 'exception', 'fail'])):
            return True
        
        # CamelCase validation on simple name
        if re.match(r'^[A-Z][a-zA-Z0-9]*(Error|Exception|Fail)', simple_name):
            return True
        
        return False
    
    # Utility methods
    def update_keywords(self, new_keywords: List[str]) -> None:
        self.keywords = new_keywords
        self._compiled_patterns = self._compile_prioritized_patterns()
        self._cache.clear()
    
    def add_keyword(self, keyword: str) -> None:
        if keyword not in self.keywords:
            self.keywords.append(keyword)
            self._compiled_patterns = self._compile_prioritized_patterns()
            self._cache.clear()
    
    def remove_keyword(self, keyword: str) -> None:
        if keyword in self.keywords:
            self.keywords.remove(keyword)
            self._compiled_patterns = self._compile_prioritized_patterns()
            self._cache.clear()
    
    def get_keywords(self) -> List[str]:
        return self.keywords.copy()
    
    def clear_cache(self) -> None:
        self._cache.clear()

