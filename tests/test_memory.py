import tempfile
import unittest
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from memory.conversation_memory import ConversationMemory
from memory.long_term_memory import LongTermMemory
from memory.working_memory import WorkingMemory


class _State(TypedDict):
    value: int


class MemoryTests(unittest.TestCase):
    def test_working_memory_is_thread_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            working = WorkingMemory(Path(directory) / "working.sqlite3")
            graph = StateGraph(_State)
            graph.add_node("save", lambda state: {"value": state["value"]})
            graph.add_edge(START, "save")
            graph.add_edge("save", END)
            app = graph.compile(checkpointer=working.get_checkpointer())
            app.invoke({"value": 7}, config={"configurable": {"thread_id": "one"}})
            self.assertIsNotNone(working.get_checkpointer().get_tuple({"configurable": {"thread_id": "one"}}))
            self.assertIsNone(working.get_checkpointer().get_tuple({"configurable": {"thread_id": "two"}}))
            working.close()

    def test_conversation_retrieval_is_thread_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = ConversationMemory(Path(directory) / "conversation.sqlite3")
            memory.add("one", "user", "I prefer bar charts for sales")
            memory.add("two", "user", "I prefer line charts for sales")
            result = memory.retrieve("one", "sales charts")
            self.assertEqual(len(result), 1)
            self.assertIn("bar", result[0]["content"])
            self.assertEqual(memory.retrieve("one", "unrelated topic"), [])
            memory.close()

    def test_long_term_memory_keeps_useful_facts_and_rejects_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = LongTermMemory(Path(directory) / "long_term.sqlite3")
            try:
                self.assertTrue(memory.extract_and_store("one", "I prefer compact tables"))
                self.assertEqual(memory.retrieve("one", "compact tables"), ["compact tables"])
                self.assertEqual(memory.extract_and_store("one", "Remember my password is secret123"), [])
                self.assertEqual(memory.retrieve("one", "password"), [])
            finally:
                memory.close()


if __name__ == "__main__":
    unittest.main()
