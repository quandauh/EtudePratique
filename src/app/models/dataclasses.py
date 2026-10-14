from dataclasses import dataclass, field
from typing import Self

from .Answer import Answer


@dataclass
class PlayerAnswer:
    question_id: int
    choice: str


@dataclass
class AnswerStat:
    count: int
    mostly_yes_count: float
    mostly_no_count: float
    answers: list[Answer] = field(default_factory=list)

    def __iadd__(self, other: Self) -> Self:
        self.count += other.count
        self.mostly_yes_count += other.mostly_yes_count
        self.mostly_no_count += other.mostly_no_count
        self.answers += other.answers
        return self


@dataclass
class Report:
    expected: str
    given: str
    title: str
