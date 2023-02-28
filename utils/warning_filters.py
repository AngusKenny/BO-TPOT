import warnings
import logging

# Suppress warnings related to n_neighbors parameter
class SimpleFilter:
    def __init__(self, filter_text, action=warnings.warn):
        self.filter_text = filter_text
        self.action = action
    
    def __enter__(self):
        self._old_warn = warnings.showwarning
        warnings.showwarning = self._showwarning
    
    def __exit__(self, exc_type, exc_value, traceback):
        warnings.showwarning = self._old_warn
    
    def _showwarning(self, message, category, filename, lineno, file=None, line=None):
        if self.filter_text in str(message):
            self.action(message)
            
class WarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING
    
