import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("No JAVA_HOME or GOOGLE_API_KEY found in .env")
else:
    print(f"Using Key: {api_key[:5]}...")
    genai.configure(api_key=api_key)
    print("Listing available models that support generateContent:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
