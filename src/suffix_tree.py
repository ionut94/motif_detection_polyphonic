from typing import List, Dict, Tuple, Set, Optional
import numpy as np

class Node:
    """Node class for the suffix tree."""
    def __init__(self):
        self.children = {}
        self.suffix_link = None
        self.start = -1
        self.end = -1
        self.suffix_index = -1

class SuffixTree:
    """Implementation of a suffix tree for degenerate strings."""
    def __init__(self, s: str):
        """Initialize with the input string."""
        self.s = s
        self.root = Node()
        self.build()
    
    def build(self):
        """Build the suffix tree for the string s."""
        # Implementation of Ukkonen's algorithm
        for i in range(len(self.s)):
            self._add_suffix(self.s[i:], i)
    
    def _add_suffix(self, suffix: str, suffix_index: int):
        """Add a suffix to the tree."""
        current = self.root
        for char in suffix:
            if char not in current.children:
                current.children[char] = Node()
            current = current.children[char]
        current.suffix_index = suffix_index
    
    def lce_query(self, s: str, i: int, n: int, mismatch_bound: int) -> Tuple[int, int]:
        """
        Perform an LCE query with allowed mismatches.
        
        Args:
            s: The string to query
            i: Starting position in the string
            n: Length of the string
            mismatch_bound: Maximum allowed mismatches (delta)
            
        Returns:
            Tuple of (length of match, number of mismatches)
        """
        if i >= len(s):
            return 0, 0
        
        current = self.root
        pos = i
        mismatches = 0
        
        while pos < min(len(s), i + n) and mismatches <= mismatch_bound:
            char = s[pos]
            
            if char in current.children:
                current = current.children[char]
                pos += 1
            else:
                # Try to find the best match with minimal difference
                best_diff = float('inf')
                best_char = None
                
                for c in current.children:
                    # Calculate difference using the formula from the paper
                    x_val = ord(char) - ord('A')
                    y_val = ord(c) - ord('A')
                    diff = min((x_val - y_val) % 12, (y_val - x_val) % 12)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_char = c
                
                if best_char is not None and best_diff <= mismatch_bound:
                    mismatches += 1
                    current = current.children[best_char]
                    pos += 1
                else:
                    break
        
        return pos - i, mismatches
    
    @staticmethod
    def create_combined_string(degenerate_string: str, motif_string: str) -> str:
        """
        Create the combined string S = X_{$}#_{1}P#_{2} as described in the paper.
        
        Args:
            degenerate_string: The degenerate string X_{$}
            motif_string: The motif string P
            
        Returns:
            The combined string S
        """
        # Use special characters as separators
        separator1 = '⚑'  # First separator (#_1)
        separator2 = '⚐'  # Second separator (#_2)
        return degenerate_string + separator1 + motif_string + separator2
