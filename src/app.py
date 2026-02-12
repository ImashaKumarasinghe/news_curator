from dotenv import load_dotenv
from src.graph import app as graph_app
from src.storage import load_preferences, save_preferences
from src.vector_store import VectorStore
import src.ui as ui

class NewsCuratorApp:
    def __init__(self):
        load_dotenv()
        self.state = {}
        # Initialize Vector Store
        self.vector_store = VectorStore()

    def initialize_session(self):
        ui.clear_screen()
        saved_prefs = load_preferences()
        
        # Hydrate viewed_ids from Vector Store
        viewed_ids = self.vector_store.get_all_viewed_ids()
        
        if saved_prefs:
            self.state = saved_prefs
            self.state["search_results"] = []  # Reset transient data
            self.state["viewed_ids"] = viewed_ids # Sync from DB
            self.state["vector_store"] = self.vector_store # Pass store to Graph!
            
            # Legacy migration: Ensure topic_weights exists
            if "topic_weights" not in self.state:
                self.state["topic_weights"] = {}
                
            ui.show_resuming_message(self.state.get("topics", []))
        else:
            ui.show_welcome()
            topics = ui.get_initial_topics()
            ui.show_feed_start(topics)
            
            # Initialize weights
            weights = {t: 1 for t in topics}
            
            self.state = {
                "topics": topics,
                "topic_weights": weights,
                "user_profile": f"User is interested in {', '.join(topics)}",
                "viewed_ids": viewed_ids,
                "feedback_history": [],
                "search_results": [],
                "vector_store": self.vector_store
            }

    def process_new_topic(self):
        new_topic = ui.get_new_topic()
        if new_topic:
            self.state["topics"].append(new_topic)
            self.state["search_results"] = []
            self.state["search_query"] = "" # Reset specific query if any
            self.state["loop_count"] = 0
            ui.show_added_topic(new_topic)

    def run(self):
        self.initialize_session()

        while True:
            try:
                # Invoke Graph
                result = graph_app.invoke(self.state)
                # Update state completely
                self.state = result
                # IMPORTANT: Ensure vector_store is preserved in state if graph didn't return it
                if "vector_store" not in self.state:
                    self.state["vector_store"] = self.vector_store
                
                # Save config (topics/profile text)
                save_preferences(self.state)

                article = result.get("current_recommendation")
                if not article:
                    ui.show_no_more_articles()
                    self.process_new_topic()
                    continue

                ui.show_article(article)
                
                choice = ui.get_user_action()

                if choice == 'Q':
                    self.vector_store.close()
                    ui.show_goodbye()
                    break
                
                elif choice == 'P':
                    ui.show_profile(self.state.get("user_profile", ""))
                    continue
                
                elif choice == 'N':
                    self.process_new_topic()
                    continue

                elif choice in ['L', 'D']:
                    liked = (choice == 'L')
                    # Save to Vector Store immediately
                    self.vector_store.add_interaction(article, liked)
                    
                    # Update internal state for UI feedback loop
                    self.state["feedback_history"].append({"article": article, "liked": liked})
                    ui.show_updating()
                
                elif choice == 'S':
                    # Log as viewed but maybe not liked/disliked? 
                    # For now just skip.
                    # Technically we should add to 'viewed_ids' in vector store?
                    # The get_all_viewed_ids only gets what we UPSERTED.
                    # If we skip, we should strictly just add ID.
                    # Our VectorStore.add_interaction requires an article dict.
                    # Let's just track it in memory as 'viewed_ids' in the state loop for now.
                    ui.show_skipping()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                ui.show_error(e)
                break
