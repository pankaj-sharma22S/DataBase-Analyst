"""Coordinates memory retrieval and persistence without creating a second workflow."""

from pathlib import Path
import os

from .conversation_memory import ConversationMemory
from .long_term_memory import LongTermMemory
from .preference_memory import PreferenceMemory
from .postgres_memory import PostgresMemoryStore
from .working_memory import WorkingMemory


class MemoryManager:
    def __init__(self, directory: str | Path = "memory"):
        directory = Path(directory)
        memory_url = os.getenv("MEMORY_URL") or os.getenv("MEMORY__URL")
        self.postgres = PostgresMemoryStore(memory_url) if memory_url and memory_url.startswith("postgres") else None
        self.working = WorkingMemory(directory / "working_memory.sqlite3", database_url=memory_url if self.postgres else None)
        self.conversations = None if self.postgres else ConversationMemory(directory / "conversations.sqlite3")
        self.long_term = None if self.postgres else LongTermMemory(directory / "long_term.sqlite3")
        self.preferences = None if self.postgres else PreferenceMemory(directory / "long_term.sqlite3")

    def retrieve_context(self, thread_id: str, query: str) -> str:
        conversations = self.postgres.retrieve_conversations(thread_id, query) if self.postgres else self.conversations.retrieve(thread_id, query)
        memories = self.postgres.retrieve_long_term(thread_id, query) if self.postgres else self.long_term.retrieve(thread_id, query)
        preferences = self.postgres.retrieve_preferences(thread_id, query) if self.postgres else self.preferences.retrieve(thread_id, query)
        sections = []
        if preferences:
            sections.append("Relevant preferences for this thread:\n" + "\n".join(f"- {item}" for item in preferences))
        if memories:
            sections.append("Relevant long-term memory:\n" + "\n".join(f"- {item}" for item in memories))
        if conversations:
            sections.append(
                "Relevant previous conversation:\n"
                + "\n".join(f"- {item['role']}: {item['content']}" for item in conversations)
            )
        return "\n\n".join(sections)

    def remember(self, thread_id: str, user_query: str, response: str) -> None:
        if self.postgres:
            self.postgres.add_conversation(thread_id, "user", user_query)
        else:
            self.conversations.add(thread_id, "user", user_query)
        if response:
            if self.postgres:
                self.postgres.add_conversation(thread_id, "assistant", response)
            else:
                self.conversations.add(thread_id, "assistant", response)
        if self.postgres:
            self.postgres.extract_long_term(thread_id, user_query)
            self.postgres.extract_preference(thread_id, user_query)
        else:
            self.long_term.extract_and_store(thread_id, user_query)
            self.preferences.extract_and_store(thread_id, user_query)

    def close(self) -> None:
        if self.postgres:
            self.postgres.close()
        else:
            self.conversations.close()
            self.long_term.close()
            self.preferences.close()
        self.working.close()
