import re
from collections import Counter, defaultdict
from typing import List, Dict, Set, Any, Optional, Tuple

class LogExtractor:
    """
    Fixed library for extracting exceptions and errors from text logs.
    Prevents double-counting of package names and simple names.
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
        """Compile patterns with priorities to prevent overmatching."""
        patterns = []
        
        base_patterns = [
            (r'at\s+(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))', 'stack_trace_full', 100),
            (r'Caused by:\s*(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))', 'caused_by', 100),
            (r'(?:raise|throw)\s+([a-zA-Z0-9_.]+(?:Error|Exception))', 'raise_throw', 90),
            (r'(?:throws|catch|except)\s+([a-zA-Z0-9_.]+(?:Error|Exception))', 'declaration', 90),
            (r'\b(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))(?:\s*:|)', 'package_with_colon', 80),
            (r'\b(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))\b', 'package_general', 70),
            (r'\b([A-Z][a-zA-Z0-9_]*(?:Error|Exception))(?:\s*:|)', 'simple_with_colon', 60),
            (r'\b([A-Z][a-zA-Z0-9_]*(?:Error|Exception))\b', 'simple_general', 50),
            (r'(?:Exception|Error):\s*[^\n]+', 'exception_description', 40),
        ]
        
        for pattern, pattern_type, priority in base_patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                patterns.append((compiled, pattern_type, priority))
            except re.error:
                continue
        
        for keyword in self.keywords:
            try:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                compiled = re.compile(pattern, re.IGNORECASE)
                patterns.append((compiled, f'keyword_{keyword}', 75))
            except re.error:
                continue
        
        patterns.sort(key=lambda x: x[2], reverse=True)
        return patterns
    
    def extract_from_text(self, text: str, use_cache: bool = True) -> List[str]:
        """Extract exceptions without duplicates."""
        if use_cache and text in self._cache:
            return self._cache[text]
        
        findings = set()
        processed_positions = set()
        
        for pattern, pattern_type, priority in self._compiled_patterns:
            for match in pattern.finditer(text):
                start_pos, end_pos = match.span()
                
                if self._is_position_processed(start_pos, end_pos, processed_positions):
                    continue
                
                processed_positions.add((start_pos, end_pos))
                
                if pattern_type.startswith('keyword_'):
                    findings.add(match.group().capitalize())
                elif pattern_type == 'exception_description':
                    extracted = self._extract_from_description_safe(match.group())
                    findings.update(extracted)
                else:
                    name = self._extract_primary_name_from_match(match, pattern_type)
                    if name and self._is_valid_exception_or_error_name(name):
                        findings.add(name)
        
        result = sorted(list(findings))
        
        if use_cache:
            if len(self._cache) > 100:
                self._cache.clear()
            self._cache[text] = result
        
        return result
    
    def analyze_frequency(self, text: str) -> Dict[str, int]:
        """
        Fixed frequency analysis without double-counting.
        Counts each occurrence only once, regardless of package/simple name preference.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with accurate frequency counts
        """
        frequency_counter = Counter()
        processed_positions = set()
        
        for pattern, pattern_type, priority in self._compiled_patterns:
            for match in pattern.finditer(text):
                start_pos, end_pos = match.span()
                
                if self._is_position_processed(start_pos, end_pos, processed_positions):
                    continue
                
                processed_positions.add((start_pos, end_pos))
                
                if pattern_type.startswith('keyword_'):
                    frequency_counter[match.group().capitalize()] += 1
                elif pattern_type == 'exception_description':
                    extracted = self._extract_from_description_safe(match.group())
                    for name in extracted:
                        frequency_counter[name] += 1
                else:
                    name = self._extract_primary_name_from_match(match, pattern_type)
                    if name and self._is_valid_exception_or_error_name(name):
                        # CRITICAL FIX: Only count the primary name once per occurrence
                        frequency_counter[name] += 1
        
        return dict(frequency_counter.most_common())
    
    def analyze_frequency_detailed(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Detailed frequency analysis that shows both package and simple name relationships
        without double-counting.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detailed frequency information
        """
        # First, get the actual occurrences with their original forms
        occurrences = self._extract_occurrences_with_original_forms(text)
        frequency_counter = Counter()
        package_mappings = self._extract_package_mappings(text)
        
        # Count each occurrence only once
        for original_name, simple_name in occurrences:
            if self.prefer_simple_names:
                frequency_counter[simple_name] += 1
                # Track package names separately without double-counting
                if self.include_package_names and '.' in original_name:
                    # Don't count package names as separate occurrences
                    # Just track that this package name exists for this simple name
                    pass
            else:
                frequency_counter[original_name] += 1
        
        # Add package names to the frequency counter for reference (with count 0)
        if self.include_package_names:
            for simple_name, packages in package_mappings.items():
                for package_name in packages:
                    if package_name not in frequency_counter:
                        frequency_counter[package_name] = 0
        
        detailed_frequency = {}
        total_occurrences = sum(frequency_counter.values())
        
        for item, count in frequency_counter.items():
            is_package_name = '.' in item
            simple_name = item.split('.')[-1] if is_package_name else item
            
            detailed_frequency[item] = {
                'count': count,
                'is_package_name': is_package_name,
                'simple_name': simple_name,
                'related_names': package_mappings.get(simple_name, []),
                'percentage': round((count / total_occurrences * 100), 2) if total_occurrences > 0 else 0,
            }
        
        return detailed_frequency
    
    def _extract_occurrences_with_original_forms(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract all occurrences with their original forms and corresponding simple names.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of tuples (original_name, simple_name)
        """
        occurrences = []
        processed_positions = set()
        
        for pattern, pattern_type, priority in self._compiled_patterns:
            for match in pattern.finditer(text):
                start_pos, end_pos = match.span()
                
                if self._is_position_processed(start_pos, end_pos, processed_positions):
                    continue
                
                processed_positions.add((start_pos, end_pos))
                
                if pattern_type.startswith('keyword_'):
                    original_name = match.group().capitalize()
                    occurrences.append((original_name, original_name))
                elif pattern_type == 'exception_description':
                    extracted = self._extract_from_description_safe(match.group())
                    for name in extracted:
                        simple_name = name.split('.')[-1] if '.' in name else name
                        occurrences.append((name, simple_name))
                else:
                    original_name = self._extract_original_name_from_match(match, pattern_type)
                    if original_name and self._is_valid_exception_or_error_name(original_name):
                        simple_name = original_name.split('.')[-1] if '.' in original_name else original_name
                        occurrences.append((original_name, simple_name))
        
        return occurrences
    
    def _extract_primary_name_from_match(self, match: re.Match, pattern_type: str) -> str:
        """
        Extract the primary name for counting - avoids double-counting.
        
        Args:
            match: Regex match object
            pattern_type: Type of pattern that matched
            
        Returns:
            Primary name to count (either package name or simple name based on preference)
        """
        original_name = self._extract_original_name_from_match(match, pattern_type)
        
        if not original_name:
            return ""
        
        # Apply naming preference without creating multiple counts
        if self.prefer_simple_names and '.' in original_name:
            return original_name.split('.')[-1]  # Return simple name only
        else:
            return original_name  # Return original name (could be package or simple)
    
    def _extract_original_name_from_match(self, match: re.Match, pattern_type: str) -> str:
        """
        Extract the original name as it appears in the text.
        
        Args:
            match: Regex match object
            pattern_type: Type of pattern that matched
            
        Returns:
            Original exception name as found in text
        """
        matched_text = match.group()
        groups = match.groups()
        
        if pattern_type in ['stack_trace_full', 'caused_by']:
            return groups[0] if groups else matched_text
        elif pattern_type in ['raise_throw', 'declaration']:
            return groups[0] if groups else matched_text.split()[-1]
        elif pattern_type in ['package_with_colon', 'package_general']:
            return groups[0] if groups else matched_text
        elif pattern_type in ['simple_with_colon', 'simple_general']:
            return groups[0] if groups else matched_text
        else:
            return matched_text
    
    def extract_with_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract exceptions with comprehensive metadata.
        Fixed to prevent double-counting in frequency analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing findings and metadata
        """
        findings = self.extract_from_text(text)
        frequency = self.analyze_frequency(text)  # Use the fixed frequency analysis
        line_occurrences = self.analyze_frequency_by_line(text)
        categorized = self._categorize_findings(findings)
        package_analysis = self.get_package_analysis(text)
        detailed_frequency = self.analyze_frequency_detailed(text)
        
        return {
            'findings': findings,
            'frequency': frequency,
            'detailed_frequency': detailed_frequency,
            'line_occurrences': line_occurrences,
            'categorized': categorized,
            'package_analysis': package_analysis,
            'summary': {
                'total_findings': len(findings),
                'total_occurrences': sum(frequency.values()),  # This will now be accurate
                'exceptions_count': len(categorized['exceptions']),
                'errors_count': len(categorized['errors']),
                'packages_count': len(categorized['package_names']),
                'keywords_count': len(categorized['keywords']),
                'simple_names_count': len(categorized['simple_names'])
            }
        }
    
    def analyze_frequency_by_line(self, text: str) -> Dict[str, List[int]]:
        """Analyze frequency with line number information."""
        line_occurrences = defaultdict(list)
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Use the same logic as analyze_frequency but per line
            line_frequency = self.analyze_frequency(line)
            for finding, count in line_frequency.items():
                for _ in range(count):
                    line_occurrences[finding].append(line_num)
        
        return dict(line_occurrences)
    
    def get_package_analysis(self, text: str) -> Dict[str, Any]:
        """Comprehensive package name analysis."""
        findings = self.extract_from_text(text)
        package_mappings = self._extract_package_mappings(text)
        
        simple_names = set()
        package_names = set()
        
        for finding in findings:
            if '.' in finding:
                package_names.add(finding)
                simple_names.add(finding.split('.')[-1])
            else:
                simple_names.add(finding)
        
        multi_package_names = {}
        for simple_name, packages in package_mappings.items():
            if len(packages) > 1:
                multi_package_names[simple_name] = packages
        
        return {
            'simple_names': sorted(list(simple_names)),
            'package_names': sorted(list(package_names)),
            'package_mappings': package_mappings,
            'multi_package_names': multi_package_names,
            'summary': {
                'total_simple_names': len(simple_names),
                'total_package_names': len(package_names),
                'names_with_packages': len(package_mappings),
                'names_with_multiple_packages': len(multi_package_names)
            }
        }
    
    def _is_position_processed(self, start: int, end: int, processed_positions: Set[tuple]) -> bool:
        """Check if a text position has already been processed."""
        for proc_start, proc_end in processed_positions:
            if start >= proc_start and end <= proc_end:
                return True
            if start < proc_end and end > proc_start:
                return True
        return False
    
    def _extract_from_description_safe(self, description: str) -> Set[str]:
        """Safe extraction from descriptions."""
        findings = set()
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
    
    def _extract_package_mappings(self, text: str) -> Dict[str, List[str]]:
        """Extract mappings between simple names and their full package names."""
        package_mappings = defaultdict(list)
        package_pattern = re.compile(r'\b(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))\b')
        
        for match in package_pattern.finditer(text):
            full_name = match.group(1)
            simple_name = full_name.split('.')[-1]
            
            if full_name not in package_mappings[simple_name]:
                package_mappings[simple_name].append(full_name)
        
        return dict(package_mappings)
    
    def _categorize_findings(self, findings: List[str]) -> Dict[str, List[str]]:
        """Categorize findings into different types."""
        categories = {
            'exceptions': [],
            'errors': [],
            'package_names': [],
            'simple_names': [],
            'keywords': []
        }
        
        for finding in findings:
            finding_lower = finding.lower()
            
            if '.' in finding:
                categories['package_names'].append(finding)
                simple_name = finding.split('.')[-1]
                if 'exception' in simple_name.lower():
                    categories['exceptions'].append(simple_name)
                elif 'error' in simple_name.lower():
                    categories['errors'].append(simple_name)
                else:
                    categories['simple_names'].append(simple_name)
            elif finding.lower() in [kw.lower() for kw in self.keywords]:
                categories['keywords'].append(finding)
            elif 'exception' in finding_lower:
                categories['exceptions'].append(finding)
            elif 'error' in finding_lower:
                categories['errors'].append(finding)
            else:
                categories['simple_names'].append(finding)
        
        for category in categories:
            categories[category] = sorted(list(set(categories[category])))
        
        return categories
    
    def _is_valid_exception_or_error_name(self, name: str) -> bool:
        """Validation that handles both package and simple names."""
        if not name or len(name) < 3:
            return False
        
        simple_name = name.split('.')[-1] if '.' in name else name
        
        exclude_words = {'the', 'and', 'or', 'but', 'for', 'with', 'from', 'this', 'that'}
        if simple_name.lower() in exclude_words:
            return False
        
        if simple_name.lower() in [kw.lower() for kw in self.keywords]:
            return True
        
        if (simple_name.endswith(('Error', 'Exception')) or
            simple_name in ('Exception', 'Error') or
            any(keyword in simple_name.lower() for keyword in ['error', 'exception', 'fail'])):
            return True
        
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

