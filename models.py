from dataclasses import dataclass

@dataclass
class ChunkResult:
    text: str
    filename: str
    score: float
