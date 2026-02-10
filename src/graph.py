import os
import json
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

from src.state import CuratorState
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

def parse_content(content):
    """Refines the content output from the LLM, handling list responses."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parsed = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parsed.append(item["text"])
            elif isinstance(item, str):
                parsed.append(item)
            else:
                parsed.append(str(item))
        return " ".join(parsed)
    if isinstance(content, dict) and "text" in content:
        return content["text"]
    return str(content)

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

# --- Node: Search ---
def search_for_content(state: CuratorState):
    """
    Searches for content based on topics and the user profile.
    Only searches if the buffer is running low (or empty).
    """
    current_results = state.get("search_results", [])
    if len(current_results) > 2:
        return {"search_results": current_results} # We have enough matched content
        
    topics = state.get("topics", [])
    profile = state.get("user_profile", "")
    viewed = state.get("viewed_ids", [])
    
    # Construct a search query
    query_prompt = f"""
    Generate a search query to find diverse news articles.
    Topics: {", ".join(topics)}
    User Profile: {profile}
    
    Return ONLY the raw search query string, nothing else.
    """
    response_msg = llm.invoke([HumanMessage(content=query_prompt)])
    search_query = parse_content(response_msg.content)
    print(f"DEBUG: Searching Tavily for: {search_query}")
    
    if tavily_client:
        response = tavily_client.search(query=search_query, search_depth="advanced", max_results=5)
        new_results = response.get("results", [])
    else:
        # Mock for when API is missing
        new_results = [{"title": "Mock Article", "url": "http://example.com", "content": "Please set TAVILY_API_KEY"}]
        
    # Filter out already viewed
    filtered_results = [r for r in new_results if r["url"] not in viewed]
    
    # Add to existing buffer
    return {"search_results": current_results + filtered_results}

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

workflow.add_node("refine_profile", refine_profile)
workflow.add_node("search", search_for_content)
workflow.add_node("curate", select_article)

# Flow
# Start -> Refine (Update model based on previous input) -> Search (If needed) -> Curate -> End
workflow.set_entry_point("refine_profile")
workflow.add_edge("refine_profile", "search")
workflow.add_edge("search", "curate")
workflow.add_edge("curate", END)

app = workflow.compile()
