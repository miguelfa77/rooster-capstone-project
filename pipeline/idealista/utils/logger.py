# utils/logger.py
"""
Simple logging utility that can be easily extended later.
"""

from datetime import datetime
from ..config import LOG_TIMESTAMPS


class Logger:
    """Simple logger that can be easily extended later"""
    
    @staticmethod
    def _format_message(level: str, message: str) -> str:
        """Format log message with optional timestamp"""
        if LOG_TIMESTAMPS:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"[{timestamp}] [{level}] {message}"
        return f"[{level}] {message}"
    
    @staticmethod
    def info(message: str):
        """Log info message"""
        print(Logger._format_message("INFO", message), flush=True)

    @staticmethod
    def warning(message: str):
        """Log warning message"""
        print(Logger._format_message("WARNING", message), flush=True)

    @staticmethod
    def error(message: str):
        """Log error message"""
        print(Logger._format_message("ERROR", message), flush=True)

    @staticmethod
    def success(message: str):
        """Log success message"""
        print(Logger._format_message("SUCCESS", message), flush=True)

