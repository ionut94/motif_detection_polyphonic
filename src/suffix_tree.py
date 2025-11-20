from typing import List, Dict, Tuple, Set, Optional, Callable, Union
import numpy as np

# Use try-except for relative imports
try:
    from .constants import TERMINAL_CHAR
    from .exceptions import SuffixTreeError
except ImportError:
    # Fallback for standalone usage
    TERMINAL_CHAR = chr(0)
    
    class SuffixTreeError(Exception):
        pass

class GlobalEnd:
    """
    Helper class for managing the global end pointer in Ukkonen's algorithm.
    
    This class encapsulates the global end pointer used in Ukkonen's suffix tree
    construction algorithm, allowing efficient handling of leaf edge extensions.
    """
    
    def __init__(self):
        """Initialize with value -1 (before any characters processed)."""
        self.value = -1
        
    def increment(self):
        """Increment the global end pointer by 1."""
        self.value += 1
        
    def __int__(self):
        """Return the integer value of the global end pointer."""
        return self.value
        
    def __repr__(self):
        """String representation of the global end pointer."""
        return str(self.value)

class Node:
    """
    Node class for the suffix tree.
    
    Each node represents either an internal node or a leaf in the suffix tree.
    Internal nodes represent branch points where suffixes diverge, while leaves
    represent the end of individual suffixes.
    """
    
    def __init__(self, start: int = -1, end: Optional[Union[int, GlobalEnd]] = None, suffix_index: int = -1):
        """
        Initialize a suffix tree node.
        
        Args:
            start: Starting index of the edge label in the original string
            end: Ending index (can be GlobalEnd for leaves or int for internal nodes)
            suffix_index: Index of suffix ending at this node (for leaves only)
        """
        self.children: Dict[str, 'Node'] = {}  # Maps first char of edge label to child node
        self.suffix_link: Optional['Node'] = None
        # Edge label represented by string slice S[start:end+1]
        self.start = start
        # 'end' can be a pointer to a global 'end' for leaf edges or an int for internal edges
        self.end = end
        self.suffix_index = suffix_index  # Index of suffix ending at this node (if it's a leaf)

    def edge_length(self, current_end: GlobalEnd) -> int:
        """
        Calculates the length of the edge leading to this node.
        
        Args:
            current_end: The current global end pointer
            
        Returns:
            Length of the incoming edge to this node
        """
        if self.start == -1: 
            return 0  # Root has 0 length edge incoming
        # Use current_end for leaves (where end is GlobalEnd), otherwise use the node's fixed int end
        effective_end = int(current_end) if isinstance(self.end, GlobalEnd) else int(self.end)
        return effective_end - self.start + 1

class SuffixTree:
    """Implementation of a suffix tree using Ukkonen's algorithm."""
    def __init__(self, s: str):
        """
        Initialize and build the suffix tree for the input string 's'.
        
        Args:
            s: Input string for which to build the suffix tree
            
        Raises:
            SuffixTreeError: If suffix tree construction fails
        """
        try:
            # Append a unique terminal character not in the alphabet
            self.s = s + TERMINAL_CHAR
            self.N = len(self.s)
            self.root = Node(start=-1, end=-1)  # Root node specifics
            self.root.suffix_link = self.root
            self._build_ukkonen()
        except Exception as e:
            raise SuffixTreeError(f"Failed to build suffix tree: {e}")

    def _build_ukkonen(self):
        """Build the suffix tree using Ukkonen's algorithm."""
        active_node = self.root
        # Active edge is represented by the first character, active_edge_char
        active_edge_char = None
        active_length = 0
        remaining_suffixes = 0
        global_end = GlobalEnd() # Global end pointer for leaf edges, starts at -1

        for i in range(self.N): # Phase i (character self.s[i])
            global_end.increment() # Increment global end for this phase
            remaining_suffixes += 1
            last_new_node = None # Track last created internal node for suffix link

            while remaining_suffixes > 0:
                # Determine the edge to follow or create based on active point
                if active_length == 0:
                    # If length is 0, the edge is determined by the current character s[i]
                    active_edge_char = self.s[i]

                # Check if an edge starting with active_edge_char exists from active_node
                if active_edge_char not in active_node.children:
                    # Rule 2: No edge starts with this char. Create a new leaf edge.
                    new_leaf = Node(start=i, end=global_end, suffix_index=(i - remaining_suffixes + 1))
                    active_node.children[active_edge_char] = new_leaf

                    # If an internal node was created in the previous step of this phase, link it
                    if last_new_node is not None:
                        last_new_node.suffix_link = active_node
                        last_new_node = None # Reset after setting
                else:
                    # Rule 1 or 3: An edge exists. Need to check further.
                    next_node = active_node.children[active_edge_char]
                    edge_len = next_node.edge_length(global_end)

                    # Check if we need to walk down the edge (active point is beyond this edge)
                    if active_length >= edge_len:
                        active_length -= edge_len
                        active_node = next_node
                        # Update active_edge_char based on the character *after* this edge
                        # This corresponds to the character at index (i - active_length) in string S
                        if active_length > 0:
                             active_edge_char = self.s[i - active_length]
                        else:
                             # If active_length becomes 0, next edge starts with current char s[i]
                             active_edge_char = self.s[i]
                        continue # Re-evaluate from the new active_node

                    # Active point is *within* the current edge. Check character match.
                    char_at_active_length_on_edge = self.s[next_node.start + active_length]

                    if char_at_active_length_on_edge == self.s[i]:
                        # Rule 1: Character matches. Observation found. Stop phase.
                        active_length += 1 # Extend the active length along the edge
                        # Set suffix link if an internal node was created in the previous step.
                        if last_new_node is not None:
                             last_new_node.suffix_link = active_node # Link to the current node
                             last_new_node = None
                        break # End this phase, move to next character i+1

                    else:
                        # Rule 3: Character mismatch. Split the edge.
                        # 1. Create the new internal node (split_node)
                        split_node_end = next_node.start + active_length - 1
                        split_node = Node(start=next_node.start, end=split_node_end)
                        active_node.children[active_edge_char] = split_node # Point parent to split_node

                        # 2. Create the new leaf for the current suffix
                        new_leaf = Node(start=i, end=global_end, suffix_index=(i - remaining_suffixes + 1))
                        split_node.children[self.s[i]] = new_leaf # Add new leaf under split_node

                        # 3. Adjust the original child (next_node)
                        next_node.start += active_length # It now starts after the split point
                        # Add original child under split_node, keyed by the mismatching char
                        split_node.children[char_at_active_length_on_edge] = next_node

                        # Set suffix link for previously created internal node (now points to split_node)
                        if last_new_node is not None:
                            last_new_node.suffix_link = split_node
                        # The new split_node needs its suffix link set later
                        last_new_node = split_node

                # One suffix extension processed in this phase
                remaining_suffixes -= 1

                # Update active point for the next extension (next shorter suffix)
                if active_node == self.root and active_length > 0:
                    # Trick for root: decrease length, adjust edge char based on start of next suffix
                    active_length -= 1
                    # The next suffix starts at index i - remaining_suffixes + 1
                    active_edge_char = self.s[i - remaining_suffixes + 1]
                elif active_node != self.root:
                    # Follow suffix link
                    active_node = active_node.suffix_link
                    if active_node is None:
                         # This should not happen in correct Ukkonen's! Indicates a bug in link setting.
                         raise Exception(f"Critical Error: Suffix link led to None node during phase i={i}, remaining_suffixes={remaining_suffixes+1}") # +1 because we just decremented
                # If active_node is root and active_length becomes 0, active point stays (root, None, 0)
                # active_edge_char will be set to s[i] at start of next iteration if needed.


    def lce_k_gamma_query(self, i: int, j: int, delta_bound: int, gamma_bound: int, match_function: Callable, max_length: int = None) -> int:
        """
        Performs the Longest Common Extension query between suffixes starting
        at index i and j in the original string S (excluding terminal char),
        ensuring each individual mismatch is within delta_bound (per-note pitch
        difference) and the cumulative difference stays within gamma_bound.
        Uses direct character comparison.

        Args:
            i: Start index of the first suffix in S (original string, pre-terminal).
            j: Start index of the second suffix in S (original string, pre-terminal).
            delta_bound: Maximum allowed per-note pitch-class difference.
            gamma_bound: Maximum allowed sum of differences for mismatches.
            match_function: The function implementing the matching table M logic.
            max_length: Optional maximum length to compare (for pattern matching).

        Returns:
            The length of the longest common extension found within bounds.
        """
        # Adjust N to exclude the terminal character for comparison length
        original_N = self.N - 1
        if i >= original_N or j >= original_N:
            return 0

        length = 0
        current_gamma = 0

        # Compare characters directly between the two suffixes
        while True:
            current_i = i + length
            current_j = j + length

            # Stop if either suffix goes beyond original string length
            if current_i >= original_N or current_j >= original_N:
                break

            # Stop if we've reached the maximum length constraint
            if max_length is not None and length >= max_length:
                break

            s_char_i = self.s[current_i]
            s_char_j = self.s[current_j]

            # Perform match using the provided function
            is_match, per_note_delta, gamma_increase = match_function(s_char_i, s_char_j)

            if not is_match:
                if per_note_delta > delta_bound:
                    break

                proposed_gamma = current_gamma + gamma_increase
                if proposed_gamma > gamma_bound:
                    break

                current_gamma = proposed_gamma
            else:
                if gamma_increase:
                    proposed_gamma = current_gamma + gamma_increase
                    if proposed_gamma > gamma_bound:
                        break
                    current_gamma = proposed_gamma

            # If match or mismatch within bounds, increment length
            length += 1

        return length

    # Note: Removed the old incorrect lce_query and create_combined_string methods implicitly
    # by replacing the entire file content.
