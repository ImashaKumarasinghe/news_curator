import os
import json
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

from src.state import CuratorState
from src.grader import grade_documents
from src.utils import parse_content
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Check which API key is available and configure the LLM accordingly
if os.environ.get("GOOGLE_API_KEY"):
    # Using the generic 'latest' alias which is usually the most stable free-tier model
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    print("Using Google Gemini Model (gemini-flash-latest)")
elif os.environ.get("OPENAI_API_KEY"):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    print("Using OpenAI Model")
else:
    raise ValueError("No API Key found. Please set GOOGLE_API_KEY or OPENAI_API_KEY in .env")

try:
    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
except:
    tavily_client = None

# --- Node: Refine User Profile ---
def refine_profile(state: CuratorState):
    """
    Updates the text-based user profile based on the most recent feedback.
    """
    history = state.get("feedback_history", [])
    if not history:
        return {"user_profile": state.get("user_profile", "")}

    # Only process new feedback if we were using a delta system, 
    # but here we'll just look at the last interaction or the whole history if needed.
    # For efficiency, let's look at the full history to rebuild the profile or just the last item.
    # A simple approach: Take the current profile and the NEW feedback to generate a new profile.
    
    current_profile = state.get("user_profile", "No specific preferences yet.")
    
    # We'll grab the last feedback item to update the model incrementally
    last_interaction = history[-1]
    article_title = last_interaction["article"].get("title", "Unknown")
    liked = last_interaction["liked"]
    sentiment = "LIKED" if liked else "DISLIKED"
    
    instructions = f"""
    You are an expert personalized news curator profile builder.
    
    Current User Profile:
    {current_profile}
    
    New Interaction:
    User {sentiment} the article: "{article_title}"
    Content: {last_interaction["article"].get("content", "")[:200]}...
    
    Update the user profile to reflect this new preference. 
    Keep it concise (3-5 sentences). 
    Focus on what topics, styles, or entities the user enjoys or avoids.
    """
    
    response = llm.invoke([HumanMessage(content=instructions)])
    content = parse_content(response.content)
    return {"user_profile": content}


def manage_topics(state: CuratorState):
    """
    Extracts topics from the last liked article and updates weights.
    Also handles decay or negative weights for dislikes contextually.
    """
    history = state.get("feedback_history", [])
    topic_weights = state.get("topic_weights", {})
    
    # Initialize if empty
    if not topic_weights:
        # Fallback to legacy
        legacy = state.get("topics", [])
        topic_weights = {t: 1 for t in legacy}

    if not history:
        return {"topic_weights": topic_weights}

    last_interaction = history[-1]
    article = last_interaction["article"]
    liked = last_interaction["liked"]
    
    # We only auto-add/weight topics if Liked.
    # If Disliked, we could try to find the topic and decrement, 
    # but extracting topics from a disliked article to downvote them is risky 
    # (might ban a broad topic like "Technology" just because of one bad article).
    
    if liked:
        extract_prompt = f"""
        Extract 2-3 specific topics/tags from this article.
        Title: {article.get('title')}
        Summary: {article.get('content')[:300]}
        
        Return ONLY a JSON list of strings, e.g. ["SpaceX", "Mars", "Rockets"]
        """
        try:
            response = llm.invoke([HumanMessage(content=extract_prompt)])
            import json
            # Simple clean up of code blocks if any
            txt = response.content.strip()
            if txt.startswith("```json"): txt = txt[7:-3]
            if txt.startswith("```"): txt = txt[3:-3]
            
            new_topics = json.loads(txt)
            if isinstance(new_topics, list):
                print(f"DEBUG: Found new topics: {new_topics}")
                for t in new_topics:
                    t = t.title().strip()
                    # Increment or Add
                    topic_weights[t] = topic_weights.get(t, 0) + 1
            
        except Exception as e:
            print(f"DEBUG: Topic extraction failed: {e}")
            
    # If Disliked, we might want to decrement the weight of the CURRENTLY SELECTED search topic?
    # But we don't strictly track which topic *generated* this result in the state easily 
    # unless we parse it from the search query. 
    # For now, let's keep it simple: Likes reinforce topics. Dislikes are handled by the Centroid Reranker pushing away.
    
    return {"topic_weights": topic_weights}



def search_for_content(state: CuratorState):
    """
    Searches for content using a Multi-Armed Bandit strategy on Weighted Topics.
    """
    current_results = state.get("search_results", [])
    if len(current_results) > 2:
        return {"search_results": current_results} 
        
    topic_weights = state.get("topic_weights", {})
    legacy_topics = state.get("topics", [])
    viewed = state.get("viewed_ids", [])
    
    # Check for Refined Query from Agentic Loop
    existing_query = state.get("search_query")
    loop_cnt = state.get("loop_count", 0)
    
    if existing_query and loop_cnt > 0:
         search_query = existing_query
         print(f"DEBUG: Retrying with Refined Query (Loop {loop_cnt}): {search_query}")
    else:
        # --- Weighted Topic Selection ---
        selected_topic = ""
        
        # Ensure we have weights
        if not topic_weights and legacy_topics:
            topic_weights = {t: 1 for t in legacy_topics}
            
        if topic_weights:
            # Filter out negative weights (don't search for disliked topics)
            candidates = {k: v for k, v in topic_weights.items() if v > 0}
            if not candidates:
                 # If all hated, pick random legacy or fallback
                 candidates = {t: 1 for t in legacy_topics} if legacy_topics else {"General News": 1}
            
            topics = list(candidates.keys())
            weights = list(candidates.values())
            
            # Weighted Random Choice
            import random
            selected_topic = random.choices(topics, weights=weights, k=1)[0]
            print(f"DEBUG: MAB Strategy Selected Topic: '{selected_topic}' (Weight: {candidates[selected_topic]})")
        else:
            selected_topic = "General Top News"

        # Construct Query
        # We focus explicitly on this topic to ensure diversity and prevent "Mushy Centroid" queries.
        query_prompt = f"""
        Generate a specific search query for the topic: "{selected_topic}".
        User Profile Context: {state.get("user_profile", "")}
        
        The goal is to find fresh, high-quality news SPECIFICALLY about "{selected_topic}".
        Do not make the query too broad.
        Return ONLY the raw query string.
        """
        response_msg = llm.invoke([HumanMessage(content=query_prompt)])
        search_query = parse_content(response_msg.content)
        print(f"DEBUG: Searching Tavily for: {search_query}")
    
    if tavily_client:
        response = tavily_client.search(query=search_query, search_depth="advanced", max_results=5)
        new_results = response.get("results", [])
    else:
        new_results = []
        
    # Filter out already viewed
    filtered_results = [r for r in new_results if r["url"] not in viewed]
    
    return {
        "search_results": current_results + filtered_results,
        "search_query": search_query # Store for refinement if needed
    }


AUTHORITATIVE_DOMAINS = [
    "reuters.com", "bbc.com", "bbc.co.uk", "apnews.com", "npr.org", 
    "pbs.org", "nytimes.com", "wsj.com", "economist.com", "bloomberg.com", 
    "theguardian.com", "washingtonpost.com", "ft.com", "cnbc.com",
    "aljazeera.com", "euronews.com", "dw.com", "france24.com"
]

# --- Node: Grader (Performance Model) ---
def run_grader(state: CuratorState):
    """
    Evaluates the quality of the search results using Vector Centroid Re-ranking
    if available, plus Heuristic scoring (Authoritative + Content length).
    Otherwise falls back to LLM grading.
    """
    print("DEBUG: Grading documents...")
    
    vector_store = state.get("vector_store")
    results = state.get("search_results", [])
    
    if not results:
         return {"relevance_score": 0, "search_results": []}

    # --- 1. Content Length Check (Heuristic) ---
    # Instead of filtering purely, we'll penalize short content by moving it to the bottom
    # or filtering if extremely short (< 100 chars).
    
    clean_results = []
    short_results = []
    
    for r in results:
        if len(r.get("content", "")) < 100:
            continue # Skip extremely short noise
        elif len(r.get("content", "")) < 400:
            short_results.append(r) # Keep but downrank
        else:
            clean_results.append(r)
            
    # Combine: Normal length first, then short
    # This effectively "scores" short content lower by placement
    valid_results = clean_results + short_results
    
    if not valid_results:
         return {"relevance_score": 0, "search_results": []}

    if vector_store:
        # --- 2. Centroid Re-ranking (Performance Model) ---
        print("DEBUG: Re-ranking with Vector Store Centroid...")
        # Re-rank only the 'valid' results. 
        # Note: vector store reranking is purely semantic. It might float a short article up if it matches the query well.
        # So we should rerank the *entire* list, then apply heuristics?
        # Better: Rerank everything first (to get semantic relevance), THEN apply length/authority penalties/boosts.
        
        reranked = vector_store.rerank_results(valid_results)
        
        # --- 3. Authoritative Domain Boost (Heuristic) ---
        high_auth = []
        regular = []
        short_content_bin = [] # Sink for short content even if high authority? No, authority wins usually.
        
        # Let's use a point system to sort
        scored_docs = []
        for i, doc in enumerate(reranked):
            # Start with reverse rank score (0 to N)
            score = len(reranked) - i
            
            # Authoritative Boost
            url = doc.get("url", "").lower()
            if any(domain in url for domain in AUTHORITATIVE_DOMAINS):
                score += len(reranked) * 0.5 # Boost by 50% of list size
                
            # Content Length Penalty
            content_len = len(doc.get("content", ""))
            if content_len < 400:
                score -= len(reranked) * 0.5 # Penalty
                
            scored_docs.append((score, doc))
            
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        final_list = [item[1] for item in scored_docs]
        
        # Take top 5
        filtered = final_list[:5]
        
        return {
            "search_results": filtered,
            "relevance_score": len(filtered)
        }
    else:
        # Fallback to LLM grading
        # We still respect the length filtering (extremely short < 100 removed)
        temp_state = state.copy()
        temp_state["search_results"] = valid_results
        grade = grade_documents(temp_state, llm)
        return grade

# --- Node: Refine Query (Agentic Correction) ---
def refine_query_node(state: CuratorState):
    """
    Generates a generic or improved query if the grading was poor.
    """
    topics = state.get("topics", [])
    current_query = state.get("search_query", "")
    loop_cnt = state.get("loop_count", 0)
    
    print("DEBUG: Refining search query...")
    
    # Simple strategy: Broaden the search
    refine_prompt = f"""
    The previous search for {", ".join(topics)} yielded poor results.
    Previous Query: {current_query}
    
    Generate a NEW, DIFFERENT, BETTER search query. 
    Make it broader or use different keywords.
    Return ONLY the raw query string.
    """
    
    response = llm.invoke([HumanMessage(content=refine_prompt)])
    new_query = parse_content(response.content)
    
    return {
        "search_query": new_query,
        "loop_count": loop_cnt + 1,
        "search_results": [] # Clear bad results to force new search
    }

def should_continue(state: CuratorState) -> Literal["refine_query", "curate"]:
    """
    Decides whether to retry search or proceed to curation.
    """
    score = state.get("relevance_score", 0)
    loop_cnt = state.get("loop_count", 0)
    results = state.get("search_results", [])
    
    # Thresholds: If we have at least 1 relevant doc, or we looped too much
    if len(results) >= 1 or loop_cnt >= 2:
        return "curate"
    
    # Otherwise retry
    return "refine_query"

# --- Node: Curate ---
def select_article(state: CuratorState):
    """
    Selects the best article from the search results to show next.
    """
    results = state.get("search_results", [])
    profile = state.get("user_profile", "")
    
    if not results:
        return {"current_recommendation": None}
        
    # If we have a profile, we could ask the LLM to pick the best one.
    # For speed and simplicity in this step, let's ask the LLM to pick the index of the best article.
    
    candidates_text = ""
    for i, r in enumerate(results):
        candidates_text += f"[{i}] {r.get('title')} - {r.get('content')[:100]}...\n"
        
    selection_prompt = f"""
    You are a curator. Select the best article for the user based on their profile.
    
    User Profile: {profile}
    
    Candidates:
    {candidates_text}
    
    Return ONLY the index number (0-{len(results)-1}) of the best article.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=selection_prompt)])
        index_str = parse_content(response.content)
        index = int(index_str.strip())
        
        selected = results[index]
        remaining = results[:index] + results[index+1:]
    except:
        # Fallback: just take the first one
        selected = results[0]
        remaining = results[1:]
        
    return {
        "current_recommendation": selected,
        "search_results": remaining,
        "viewed_ids": [selected["url"]] # Add to viewed immediately
    }

# --- Graph Definition ---
workflow = StateGraph(CuratorState)

workflow.add_node("manage_topics", manage_topics) # New Node
workflow.add_node("refine_profile", refine_profile)
workflow.add_node("search", search_for_content)
workflow.add_node("grade", run_grader)
workflow.add_node("refine_query", refine_query_node)
workflow.add_node("curate", select_article)

# Flow
workflow.set_entry_point("manage_topics") # Start here
workflow.add_edge("manage_topics", "refine_profile")
workflow.add_edge("refine_profile", "search")
workflow.add_edge("search", "grade")

# Conditional Edge from Grader
workflow.add_conditional_edges(
    "grade",
    should_continue,
    {
        "refine_query": "refine_query",
        "curate": "curate"
    }
)

workflow.add_edge("refine_query", "search")
workflow.add_edge("curate", END)

app = workflow.compile()
