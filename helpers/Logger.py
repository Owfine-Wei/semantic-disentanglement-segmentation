"""
Simple file logger utility.

Creates a text log file under `log_root` and provides a `log` method and
callable interface to append text entries.
"""

import os


class Logger:
    """Append-only logger that writes strings to a dated file.

    The logger ensures the output directory exists and writes appended
    content using UTF-8 encoding.
    """

    def __init__(self, date, info, log_root):

        self.date = date
        self.info = info
        self.log_root = log_root

        # create log directory if missing
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        file_name = f"log_{self.date}_{self.info}.txt"
        self.log_path = os.path.join(self.log_root, file_name)

    def log(self, content):
        """
        Append `content` (converted to str) to the log file.

        Content is written as-is (no newline added automatically).
        """

        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(str(content))

    def __call__(self, content):
        # allow instance to be used as a callable alias for `log`
        self.log(content)
