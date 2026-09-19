"""Persistent LangGraph working-memory checkpointer."""

from pathlib import Path
import sqlite3


class WorkingMemory:
    def __init__(self, database_path: str | Path = "memory/working_memory.sqlite3"):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.persistent = False
        self._connection = None

        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        except ImportError:
            from langgraph.checkpoint.memory import MemorySaver
            self.checkpointer = MemorySaver()
        else:
            self._connection = sqlite3.connect(str(self.database_path), check_same_thread=False)
            # AgentState intentionally contains the existing pandas DataFrame;
            # keep the workflow unchanged and let LangGraph serialize that
            # state through its supported pickle fallback.
            self.checkpointer = SqliteSaver(
                self._connection,
                serde=JsonPlusSerializer(pickle_fallback=True),
            )
            self.checkpointer.setup()
            self.persistent = True

    def get_checkpointer(self):
        return self.checkpointer

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
