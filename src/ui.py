import os
from typing import List, Dict

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_welcome():
    clear_screen()
    print("Welcome to your AI News Curator!")
    print("I'll learn what you like over time. Let's start.\n")

def show_resuming_message(topics: List[str]):
    print("Welcome back! Loading your preferences...")
    print(f"Resuming feed for: {', '.join(topics)}...")

def get_initial_topics() -> List[str]:
    topics_input = input("Enter 3 topics you are interested in (comma separated): ")
    topics = [t.strip() for t in topics_input.split(",") if t.strip()]
    return topics if topics else ["Technology", "Science", "History"]

def show_feed_start(topics: List[str]):
    print(f"\nStarting feed for: {', '.join(topics)}...")

def show_article(article: Dict):
    print("\n" + "="*50)
    print(f"TITLE: {article.get('title', 'Unknown Title')}")
    print(f"URL:   {article.get('url', '#')}")
    print("-" * 50)
    print(f"SUMMARY: {article.get('content', 'No content available')[:300]}...")
    print("=" * 50 + "\n\033[0m")

def get_user_action() -> str:
    print("\033[0m") # Reset colors
    print("ACTIONS: [L]ike | [D]islike | [S]kip | [N]ew Topic | [P]rofile | [Q]uit")
    return input("Your Choice: ").upper().strip()

def show_profile(profile: str):
    print(f"\n--- Current Preference Profile ---\n{profile}\n----------------------------------")

def show_goodbye():
    print("Goodbye!")

def get_new_topic() -> str:
    return input("Enter new topic: ").strip()

def show_added_topic(topic: str):
    print(f"Added '{topic}'. Refreshing feed...")

def show_no_more_articles():
    print("\nNo more articles found. Try adding new topics!")

def show_updating():
    print("\nUpdating profile choice...", end="", flush=True)

def show_skipping():
    print("Skipping...")

def show_error(message):
    print(f"An error occurred: {message}")
