import operator
from typing import Annotated, List, Dict, TypedDict, Any

class CuratorState(TypedDict):
    """
    Represents the state of our news curator graph.
    """
    topics: List[str] # Maintained for legacy compatibility or simple display
    topic_weights: Dict[str, int] # The weighted set of topics
    user_profile: str
    viewed_ids: Annotated[List[str], operator.add]  # URLs or IDs of articles already seen
    feedback_history: Annotated[List[Dict], operator.add] # Stores liked/disliked articles
    current_recommendation: Dict  # The article currently being presented
    search_results: List[Dict] # Buffer of search results
    search_query: str # The active search query
    loop_count: int # Safety counter for agentic loops
    vector_store: Any # Handle to the VectorStore object passed through state

