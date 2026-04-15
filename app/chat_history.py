import json
import os

HISTORY_FILE = "data/chat_history.json"

def _load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def _save_history(history):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def store_turn(session_id: str, question: str, answer: str, sources: list):
    history = _load_history()
    if session_id not in history:
        history[session_id] = []
    
    history[session_id].append({"role": "user", "content": question})
    history[session_id].append({"role": "assistant", "content": answer, "sources": sources})
    
    _save_history(history)

def get_recent_turns(session_id: str, n: int = 4) -> list[dict]:
    history = _load_history()
    session_history = history.get(session_id, [])
    # 1 turn = 1 user message + 1 assistant message = 2 dicts
    messages_to_fetch = n * 2
    return session_history[-messages_to_fetch:] if messages_to_fetch > 0 else session_history

def clear_memory(session_id: str):
    history = _load_history()
    if session_id in history:
        del history[session_id]
        _save_history(history)
