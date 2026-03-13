from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReadFileResult:
    path: str
    content: str

    def getContent(self) -> str:
        return self.content


class AgentTools:
    """Tools intended for use by agents."""

    def useTool(self, toolName: str, *args, **kwargs):
        if not hasattr(self, toolName):
            raise ValueError(f"Tool {toolName} not found")

        toolFunc = getattr(self, toolName)
        return toolFunc(*args, **kwargs)

    @staticmethod
    def readFile(path: str, max_bytes: int = 262_144) -> ReadFileResult:
        """Read a file and return its contents.

        Args:
            path: Path to the file to read.
            max_bytes: Maximum number of bytes to read to avoid large loads.

        Returns:
            ReadFileResult with the normalized path and content.
        """
        file_path = Path(path).expanduser().resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise IsADirectoryError(f"Not a file: {file_path}")

        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_bytes)

        return ReadFileResult(path=str(file_path), content=content)
