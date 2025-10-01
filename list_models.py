import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
models = genai.list_models()
for model in models:
    if 'generateContent' in model.supported_generation_methods:
        print(f"Model: {model.name}, Methods: {model.supported_generation_methods}")