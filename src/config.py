"""
Configuration management for the motif finding application.

This module provides configuration classes and utilities for managing
application parameters, validation, and settings.
"""

from dataclasses import dataclass
from typing import Optional, List

# Use layered imports so standalone scripts can still reference shared modules
try:
    from .constants import Config
except ImportError:
    try:
        from constants import Config  # type: ignore
    except ImportError:
        class Config:
            DEFAULT_DELTA = 0
            DEFAULT_GAMMA = 0
            MIN_MOTIF_LENGTH = 2
            MAX_MOTIF_LENGTH = 100
            DEFAULT_OUTPUT_FORMAT = 'csv'
            MAX_CHANNELS = 16

try:
    from .exceptions import ParameterError
except ImportError:
    try:
        from exceptions import ParameterError  # type: ignore
    except ImportError:
        class ParameterError(Exception):
            pass

@dataclass
class MotifSearchConfig:
    """Configuration for motif search parameters."""
    
    delta: int = Config.DEFAULT_DELTA
    gamma: int = Config.DEFAULT_GAMMA
    output_format: str = Config.DEFAULT_OUTPUT_FORMAT
    max_channels: int = Config.MAX_CHANNELS
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate all configuration parameters."""
        if self.delta < 0:
            raise ParameterError(f"Delta must be non-negative, got {self.delta}")
        
        if self.gamma < 0:
            raise ParameterError(f"Gamma must be non-negative, got {self.gamma}")
        
        if self.max_channels <= 0 or self.max_channels > 16:
            raise ParameterError(f"Max channels must be between 1 and 16, got {self.max_channels}")
        
        if self.output_format not in ['csv', 'json', 'txt']:
            raise ParameterError(f"Invalid output format: {self.output_format}")

@dataclass
class MotifConfig:
    """Configuration for motif definition and validation."""
    
    pitches: List[int]
    min_length: int = Config.MIN_MOTIF_LENGTH
    max_length: int = Config.MAX_MOTIF_LENGTH
    
    def __post_init__(self):
        """Validate motif configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate motif parameters."""
        if not self.pitches:
            raise ParameterError("Motif cannot be empty")
        
        if len(self.pitches) < self.min_length:
            raise ParameterError(f"Motif too short: {len(self.pitches)} < {self.min_length}")
        
        if len(self.pitches) > self.max_length:
            raise ParameterError(f"Motif too long: {len(self.pitches)} > {self.max_length}")
        
        # Validate MIDI pitch ranges (0-127)
        for pitch in self.pitches:
            if not isinstance(pitch, int) or pitch < 0 or pitch > 127:
                raise ParameterError(f"Invalid MIDI pitch: {pitch}")

class ConfigManager:
    """Manages application configuration and provides validation utilities."""
    
    @staticmethod
    def create_search_config(delta: int = None, gamma: int = None, 
                           output_format: str = None) -> MotifSearchConfig:
        """Create a validated search configuration."""
        return MotifSearchConfig(
            delta=delta if delta is not None else Config.DEFAULT_DELTA,
            gamma=gamma if gamma is not None else Config.DEFAULT_GAMMA,
            output_format=output_format if output_format else Config.DEFAULT_OUTPUT_FORMAT
        )
    
    @staticmethod
    def create_motif_config(pitches: List[int]) -> MotifConfig:
        """Create a validated motif configuration."""
        return MotifConfig(pitches=pitches)
    
    @staticmethod
    def parse_motif_string(motif_str: str) -> List[int]:
        """Parse a comma-separated string of MIDI pitches into a list of integers."""
        try:
            pitches = [int(x.strip()) for x in motif_str.split(',')]
            # Validate through MotifConfig
            MotifConfig(pitches=pitches).validate()
            return pitches
        except ValueError as e:
            raise ParameterError(f"Invalid motif format: {e}")
        except ParameterError:
            raise  # Re-raise parameter errors
