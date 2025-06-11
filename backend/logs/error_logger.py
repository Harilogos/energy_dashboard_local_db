"""
Dedicated Error Logger for capturing only errors in a separate log file.

This module provides a centralized error logging system that writes
only ERROR level messages to a dedicated error.log file.
"""

import logging
from datetime import datetime
from pathlib import Path

class ErrorLogger:
    """
    Dedicated error logger that captures only ERROR level messages.
    """

    def __init__(self, log_file="logs/error.log"):
        """
        Initialize the error logger.

        Args:
            log_file (str): Path to the error log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create dedicated error logger
        self.logger = logging.getLogger('error_only')
        self.logger.setLevel(logging.ERROR)

        # Remove any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create file handler for error log
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.ERROR)

        # Create formatter for error messages
        formatter = logging.Formatter(
            '%(asctime)s - ERROR - %(name)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)

        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

        print(f"Error logger initialized. Errors will be logged to: {self.log_file}")

    def log_error(self, message, module_name="unknown"):
        """
        Log an error message.

        Args:
            message (str): Error message
            module_name (str): Name of the module where error occurred
        """
        # Format the message with module info
        formatted_message = f"[{module_name}] {message}"

        # Log the error
        self.logger.error(formatted_message)

    def clear_log(self):
        """Clear the error log file."""
        try:
            if self.log_file.exists():
                self.log_file.unlink()
                print(f"Error log cleared: {self.log_file}")
        except Exception as e:
            print(f"Failed to clear error log: {e}")

    def get_recent_errors(self, lines=50):
        """
        Get recent error messages from the log file.

        Args:
            lines (int): Number of recent lines to return

        Returns:
            list: List of recent error messages
        """
        try:
            if not self.log_file.exists():
                return []

            with open(self.log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
        except Exception as e:
            print(f"Failed to read error log: {e}")
            return []

    def get_error_count(self):
        """
        Get the total number of errors in the log file.

        Returns:
            int: Number of error entries
        """
        try:
            if not self.log_file.exists():
                return 0

            with open(self.log_file, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except Exception as e:
            print(f"Failed to count errors: {e}")
            return 0


class ErrorLoggerHandler(logging.Handler):
    """
    Custom logging handler that captures ERROR messages and sends them to error.log
    """

    def __init__(self, error_logger):
        """
        Initialize the handler.

        Args:
            error_logger (ErrorLogger): Instance of ErrorLogger
        """
        super().__init__()
        self.error_logger = error_logger
        self.setLevel(logging.ERROR)

    def emit(self, record):
        """
        Emit an error record to the error logger.

        Args:
            record: LogRecord instance
        """
        try:
            self.error_logger.log_error(
                message=record.getMessage(),
                module_name=record.name
            )
        except Exception:
            self.handleError(record)


# Global error logger instance
error_logger = ErrorLogger()

# Global flag to track if error logging has been set up
_error_logging_setup = False

def setup_error_logging():
    """
    Setup error logging for the entire application.
    This function should be called once at application startup.
    """
    global _error_logging_setup

    # Check if already set up to avoid duplicate setup
    if _error_logging_setup:
        return

    # Get the root logger
    root_logger = logging.getLogger()

    # Create error handler
    error_handler = ErrorLoggerHandler(error_logger)

    # Add error handler to root logger
    root_logger.addHandler(error_handler)

    # Mark as set up
    _error_logging_setup = True

    print("Error logging setup complete. All ERROR messages will be captured in error.log")

def log_error(message, module_name="app"):
    """
    Convenience function to log an error.

    Args:
        message (str): Error message
        module_name (str): Name of the module
    """
    error_logger.log_error(message=message, module_name=module_name)

def get_error_summary():
    """
    Get a summary of recent errors.

    Returns:
        dict: Error summary information
    """
    recent_errors = error_logger.get_recent_errors(10)
    total_errors = error_logger.get_error_count()

    return {
        "total_errors": total_errors,
        "recent_errors": recent_errors,
        "log_file": str(error_logger.log_file),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def clear_error_log():
    """Clear the error log file."""
    error_logger.clear_log()

if __name__ == "__main__":
    # Test the error logger
    print("Testing Error Logger...")

    # Setup error logging
    setup_error_logging()

    # Test logging some errors
    log_error("Test error message 1", "test_module")
    log_error("Test error message 2", "test_module")

    # Get error summary
    summary = get_error_summary()
    print(f"Total errors: {summary['total_errors']}")
    print(f"Log file: {summary['log_file']}")

    print("Error logger test complete!")
