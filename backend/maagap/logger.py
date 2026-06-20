import logging
import sys

def get_logger(name: str = "MAAGAP") -> logging.Logger:
    """Returns a configured logger for the MAAGAP application."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Define formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
    return logger

def banner(logger: logging.Logger, text: str) -> None:
    """Logs a visually distinct banner message."""
    divider = "=" * 70
    logger.info(divider)
    logger.info(f"  {text}")
    logger.info(divider)
