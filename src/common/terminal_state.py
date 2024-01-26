from enum import Enum


class TerminalState(Enum):
    NONE: str = None
    DEATH: str = "death"
    TASK_COMPLETED: str = "task completed"
    TIME_IS_UP: str = "time is up"
