import sys
import time
import threading

# from old project (recreating rust spinner)
class Spinner:
    """
    An ASCII spinner for indicating progress in the terminal with completion/failure animations.
    """
    def __init__(self, message="Processing", spinner_type="braille"):
        self.message = message
        self.spinning = False
        self.spinner_thread = None
        
        # Different spinner styles
        spinner_styles = {
            "braille": ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
            "dots": ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'],
            "simple": ['-', '\\', '|', '/'],
            "arrows": ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
            "bouncing": ['.  ', '.. ', '...', ' ..', '  .', '   ']
        }
        
        # Status indicators
        self.success_symbol = "✓"
        self.failure_symbol = "✗"
        
        self.spinner_chars = spinner_styles.get(spinner_type, spinner_styles["braille"])

    def spin(self):
        """Run the spinner animation."""
        i = 0
        while self.spinning:
            sys.stdout.write(f"\r{self.message} {self.spinner_chars[i % len(self.spinner_chars)]} ")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def start(self):
        """Start the spinner animation in a separate thread."""
        self.spinning = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()

    def stop(self, success=None):
        """
        Stop the spinner animation and display completion status.
        
        Args:
            success: None for neutral stop, True for success animation, False for failure animation
        """
        self.spinning = False
        if self.spinner_thread:
            self.spinner_thread.join()
            
        # Clear the line
        sys.stdout.write("\r" + " " * (len(self.message) + 10))
        
        # Show completion animation if requested
        if success is True:
            sys.stdout.write(f"\r{self.message} {self.success_symbol} Done\n")
        elif success is False:
            sys.stdout.write(f"\r{self.message} {self.failure_symbol} Failed\n")
        else:
            # Just clear the line for neutral stop
            sys.stdout.write("\r")
            
        sys.stdout.flush()
        
    def __enter__(self):
        """Support for using the spinner as a context manager with 'with' statement."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure the spinner stops when exiting the context.
        Show success animation if no exception, failure animation if exception occurred."""
        success = None if exc_type is None else False
        self.stop(success=success)