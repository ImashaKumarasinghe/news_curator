import sys
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.graph import app
from src.utils import clear_screen, print_article

# Load env variables
load_dotenv()

def main():
    clear_screen()
    print("Welcome to your AI News Curator!")
    print("I'll learn what you like over time. Let's start.\n")
    
    # 1. Get Initial Topics
    topics_input = input("Enter 3 topics you are interested in (comma separated): ")
    topics = [t.strip() for t in topics_input.split(",") if t.strip()]
    
    if not topics:
        topics = ["Technology", "Science", "History"] # Defaults
        
    print(f"\nStarting feed for: {', '.join(topics)}...")
    
    # Initial State
    state_values = {
        "topics": topics,
        "user_profile": f"User is interested in {', '.join(topics)}",
        "viewed_ids": [],
        "feedback_history": [],
        "search_results": []
    }
    
    # Main Loop
    while True:
        try:
            # Run the graph
            # The graph processes: Refine -> Search -> Curate
            # We pass the current state. LangGraph maintains state if we use a checkpointer, 
            # but here we are running it nicely in a loop passing the *accumulated* state logic 
            # or simply letting the graph return the diffs, which we apply?
            # actually app.invoke(state) returns the FINAL state after the run.
            # So we effectively update our local 'state_values' with the result.
            
            result = app.invoke(state_values)
            
            # Update our local state with the result for the next iteration
            state_values = result
            
            article = result.get("current_recommendation")
            
            if not article:
                print("\nNo more articles found. Try adding new topics!")
                new_topic = input("New Topic: ")
                state_values["topics"].append(new_topic)
                continue
                
            # Display
            print_article(article)
            
            # Interaction
            print("[L]ike  [D]islike  [S]kip  [N]ew Topic  [Q]uit")
            choice = input("Your Choice: ").upper().strip()
            
            if choice == 'Q':
                print("Goodbye!")
                break
            
            elif choice == 'N':
                new_topic = input("Enter new topic: ").strip()
                if new_topic:
                    state_values["topics"].append(new_topic)
                    # Clear buffers to force immediate exploration of the new topic
                    state_values["search_results"] = [] 
                    state_values["search_query"] = ""   
                    state_values["loop_count"] = 0
                    print(f"Added '{new_topic}'. Refreshing feed...")
                continue
                
            elif choice in ['L', 'D']:
                liked = (choice == 'L')
                
                # Create feedback entry
                feedback = {
                    "article": article,
                    "liked": liked
                }
                
                # We need to manually append this to the state for the next run 
                # so the 'Refine' node sees it.
                # Since 'feedback_history' is an Annotated[add] field, passing it in the input 
                # will ADD it to the existing list in the graph state if we were using a persistent checkpointer.
                # But since we are invoking stateless (passing full dict), we physically modify the dict.
                
                # Careful: app.invoke returns the full state list for Annotated fields
                # So state_values["feedback_history"] is already the full list.
                # We append the new one.
                # Wait, if we pass the FULL list back to invoke, and it's Annotated[add], 
                # does it double up?
                # No, app.invoke(input) initializes the state with input. 
                
                # To be safe and simple: We just append to our state_values manually.
                # The 'refine_profile' node reads from 'feedback_history'.
                
                # Actually, there's a nuance: 'refine_profile' reads from the history.
                # We should add the NEW feedback to the history NOW so the NEXT run sees it.
                
                # We need to ensure we don't duplicate logic.
                # Let's just append to the list in our state_values.
                state_values["feedback_history"].append(feedback)
                
                print("\nUpdating profile choice...", end="", flush=True)
                
            elif choice == 'S':
                print("Skipping...")
                
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()
