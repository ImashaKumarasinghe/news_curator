from langchain_core.messages import SystemMessage, HumanMessage
from src.utils import parse_content

def grade_documents(state, llm):
    """
    Evaluates the relevance of search results to the user profile and topics.
    Returns a score (0-10) and filtered results.
    """
    topics = state.get("topics", [])
    profile = state.get("user_profile", "")
    results = state.get("search_results", [])
    
    if not results:
        return {"relevance_score": 0, "search_results": []}

    # Concise prompt for grading
    # We ask for a simple JSON-like list of indices that are relevant
    
    candidates_text = ""
    for i, r in enumerate(results):
        candidates_text += f"[{i}] {r.get('title')} - {r.get('content')[:150]}...\n"

    grader_prompt = f"""
    You are a strict news editor. Grade the relevance of these articles.
    
    User Interests: {", ".join(topics)}
    User Profile: {profile}
    
    Candidates:
    {candidates_text}
    
    Task: Identify which articles are truly relevant and high quality.
    Return ONLY a list of indices of relevant articles (e.g. [0, 2]) or [] if none.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=grader_prompt)])
        content = parse_content(response.content)
        
        # Simple parsing logic
        # Expecting string like "[0, 1]"
        import json
        try:
            # Try finding the list part
            start = content.find('[')
            end = content.find(']') + 1
            if start != -1 and end != -1:
                relevant_indices = json.loads(content[start:end])
            else:
                relevant_indices = []
        except:
            # Fallback
            relevant_indices = []

        # Filter results
        relevant_docs = [results[i] for i in relevant_indices if i < len(results)]
        
        # Calculate Score (Ratio of relevant vs retrieved, or just pure count)
        score = len(relevant_docs)
        
        # If we have at least 1 good doc, we might proceed, but let's aim for 2
        
        return {
            "search_results": relevant_docs, # Update with ONLY relevant ones? Or keep them all tagged?
            # Let's keep only relevant ones to clean up the queue
            "relevance_score": score
        }
            
    except Exception as e:
        print(f"Grader Error: {e}")
        return {"relevance_score": 0}
