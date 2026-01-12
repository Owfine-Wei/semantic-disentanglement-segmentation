import os

class Logger:
    def __init__(self, date, info, log_root):

        self.date = date
        self.info = info
        self.log_root = log_root

        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        file_name = f"log_{self.date}_{self.info}.txt"
        self.log_path = os.path.join(self.log_root, file_name)
        
        # print(f"Logger initialized. File path: {self.log_path}")

    def log(self, content):

        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(str(content))

    def __call__(self, content):
        self.log(content)
