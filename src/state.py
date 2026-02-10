import operator
from typing import Annotated, List, Dict, TypedDict

class CuratorState(TypedDict):
    """
    Represents the state of our news curator graph.
    """
    topics: List[str]
    user_profile: str
    viewed_ids: Annotated[List[str], operator.add]  # URLs or IDs of articles already seen
    feedback_history: Annotated[List[Dict], operator.add] # Stores liked/disliked articles
    current_recommendation: Dict  # The article currently being presented
    search_results: List[Dict] # Buffer of search results
