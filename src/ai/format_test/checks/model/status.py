from enum import Enum
from .checks import Checks


class CheckStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    SKIPPED = "skipped"

class TestResult:
    def __init__(self, result: CheckStatus, check: Checks, description: str):
        self.result = result
        self.check = check
        self.description = description