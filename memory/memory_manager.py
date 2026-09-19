"""Coordinates memory retrieval and persistence without creating a second workflow."""

from pathlib import Path

from .conversation_memory import ConversationMemory
from .long_term_memory import LongTermMemory
from .working_memory import WorkingMemory


class MemoryManager:
    def __init__(self, directory: str | Path = "memory"):
        directory = Path(directory)
        self.working = WorkingMemory(directory / "working_memory.sqlite3")
        self.conversations = ConversationMemory(directory / "conversations.sqlite3")
        self.long_term = LongTermMemory(directory / "long_term.sqlite3")

    def retrieve_context(self, thread_id: str, query: str) -> str:
        conversations = self.conversations.retrieve(thread_id, query)
        memories = self.long_term.retrieve(thread_id, query)
        sections = []
        if memories:
            sections.append("Relevant long-term memory:\n" + "\n".join(f"- {item}" for item in memories))
        if conversations:
            sections.append(
                "Relevant previous conversation:\n"
                + "\n".join(f"- {item['role']}: {item['content']}" for item in conversations)
            )
        return "\n\n".join(sections)

    def remember(self, thread_id: str, user_query: str, response: str) -> None:
        self.conversations.add(thread_id, "user", user_query)
        if response:
            self.conversations.add(thread_id, "assistant", response)
        self.long_term.extract_and_store(thread_id, user_query)

    def close(self) -> None:
        self.conversations.close()
        self.long_term.close()
        self.working.close()
