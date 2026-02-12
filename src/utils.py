
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
