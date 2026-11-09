# balancer/logging_config.py
import logging
import os

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "teamforge.log")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("team_balancer")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

# Replace handlers so reloading the module never duplicates output
logger.handlers = []
logger.addHandler(console_handler)
logger.addHandler(file_handler)
