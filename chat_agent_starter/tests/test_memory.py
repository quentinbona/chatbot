from pathlib import Path

from app.memory import TranscriptStore


def test_transcript_store_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = TranscriptStore(db_path)
    store.add_message("abc", "user", "hello")
    store.add_message("abc", "assistant", "hi")

    messages = store.get_recent_messages("abc", limit=10)
    assert messages == [("user", "hello"), ("assistant", "hi")]
