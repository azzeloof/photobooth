import logging
import subprocess
from common import PhotoboothError

logger = logging.getLogger(__name__)

"""
the user must be part of the lp group for this to work
$ sudo usermod -a -G lp $USER
"""

class DS40Error(PhotoboothError):
    """Custom exception for DS40 printer errors"""
    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(message, recoverable, component="DS40")


class DS40:
    def __init__(self, printer_name="DS40"):
        self.printer_name = printer_name
        self.print_command = ["lp"]
        self.print_command.extend(["-d", printer_name])

    def print(self, file_path):
        """
        Print image to printer
        file_path: path to the file
        """
        try:
            command = self.print_command + [file_path]
            subprocess.run(command)
        except FileNotFoundError as e:
            raise DS40Error(f"Printer not found: {self.printer_name} - {e}")
        except PermissionError as e:
            raise DS40Error(f"Permission denied: {self.printer_name} - {e}")
        except subprocess.CalledProcessError as e:
            raise DS40Error(f"Printing failed: {e}")
        except Exception as e:
            raise DS40Error(f"An unexpected error occurred: {e}")
