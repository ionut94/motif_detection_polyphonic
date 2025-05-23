"""
Custom exceptions for the motif finding application.

This module defines application-specific exceptions to provide better error
handling and more informative error messages.
"""

class MotifFinderError(Exception):
    """Base exception class for motif finder errors."""
    pass

class MIDIProcessingError(MotifFinderError):
    """Raised when MIDI file processing fails."""
    pass

class SuffixTreeError(MotifFinderError):
    """Raised when suffix tree operations fail."""
    pass

class InvalidMotifError(MotifFinderError):
    """Raised when motif format or content is invalid."""
    pass

class ParameterError(MotifFinderError):
    """Raised when invalid parameters are provided."""
    pass

class FileNotFoundError(MotifFinderError):
    """Raised when required files are not found."""
    pass

class MemoryError(MotifFinderError):
    """Raised when memory limits are exceeded."""
    pass
