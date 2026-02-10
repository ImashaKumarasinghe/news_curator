import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_article(article: dict):
    print("\n" + "="*50)
    print(f"TITLE: {article.get('title', 'Unknown Title')}")
    print(f"URL:   {article.get('url', '#')}")
    print("-" * 50)
    print(f"SUMMARY: {article.get('content', 'No content available')[:300]}...")
    print("=" * 50 + "\n")
