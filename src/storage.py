import sqlite3
import json
import os
from typing import Dict, List, Any

DB_FILE = "news_curator.db"

def init_db():
    """Initializes the SQLite database with necessary tables."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Table: key-value store for single-value state like user_profile
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS app_state (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """)

    # Table: Topics
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS topics (
        topic_name TEXT PRIMARY KEY,
        weight INTEGER DEFAULT 1
    );
    """)

    # Table: Articles
    # Stores details about every article encountered/viewed
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        url TEXT PRIMARY KEY,
        title TEXT,
        content TEXT,
        source_json TEXT  -- Store full original metadata just in case
    );
    """)

    # Table: Viewed History
    # Tracks which articles have been shown to the user
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS viewed_articles (
        url TEXT PRIMARY KEY,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(url) REFERENCES articles(url)
    );
    """)

    # Table: Feedback
    # Tracks user likes/dislikes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        url TEXT PRIMARY KEY,
        liked BOOLEAN,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(url) REFERENCES articles(url)
    );
    """)

    conn.commit()
    conn.close()

def load_preferences() -> Dict[str, Any]:
    """
    Loads user preferences from SQLite/JSON for config,
    but VectorStore is initialized during app startup separately.
    """
    if not os.path.exists(DB_FILE):
        return None
    
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 1. Load Profile (Legacy text profile might still be useful for Query Gen)
        cursor.execute("SELECT value FROM app_state WHERE key = 'user_profile'")
        row = cursor.fetchone()
        user_profile = row['value'] if row else "User has no profile yet."

        # 2. Load Topics (and Weights)
        # We need to perform a migration check if the column 'weight' doesn't exist, 
        # but since we are handling this by recreating the DB or assuming clean slate, let's just query.
        cursor.execute("SELECT topic_name, weight FROM topics")
        rows = cursor.fetchall()
        
        topics = [r['topic_name'] for r in rows]
        topic_weights = {r['topic_name']: r['weight'] for r in rows}

        # 3. Load Viewed IDs & Feedback
        # Note: We now delegate heavy lifting to Chroma, but for startup compatibility
        # we might minimal data. 
        # Actually, let's load Viewed IDs from Chroma in the Graph logic, 
        # or here if we want to sync.
        # For now, we return empty list and let the App initialize VectorStore to get viewed_ids.
        
        # We KEEP the SQLite load just for Topics and the textual Profile summary 
        # (which is still useful for initial search queries).
        
        conn.close()

        if not topics and user_profile == "User has no profile yet.":
             return None

        return {
            "topics": topics,
            "topic_weights": topic_weights,
            "user_profile": user_profile, 
            "viewed_ids": [], # Will be populated by VectorStore in app.py
            "feedback_history": [], # Handled by VectorStore
            "search_results": []
        }

    except sqlite3.Error as e:
        print(f"Database error during load: {e}")
        return None

def save_preferences(state: Dict[str, Any]):
    """
    Saves configuration state (Topics/Profile Text) to SQLite.
    Article interactions are saved to ChromaDB directly during the run loop,
    so we don't need to save them here anymore.
    """
    try:
        init_db()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # 1. Save Profile Text
        user_profile = state.get("user_profile", "")
        cursor.execute("INSERT OR REPLACE INTO app_state (key, value) VALUES (?, ?)", 
                       ('user_profile', user_profile))

        # 2. Save Topics
        # We save the weighted topics. The List[str] 'topics' in state is now secondary.
        topic_weights = state.get("topic_weights", {})
        
        # If topic_weights is empty but topics (list) is not, Initialize weights from list
        legacy_topics = state.get("topics", [])
        if not topic_weights and legacy_topics:
            topic_weights = {t: 1 for t in legacy_topics}
            
        if topic_weights:
            cursor.execute("DELETE FROM topics")
            cursor.executemany("INSERT INTO topics (topic_name, weight) VALUES (?, ?)", 
                               [(t, w) for t, w in topic_weights.items()])
            
        conn.commit()
        conn.close()

    except sqlite3.Error as e:
        print(f"Database error during save: {e}")
