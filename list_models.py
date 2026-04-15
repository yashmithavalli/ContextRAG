import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
res = requests.get(url)
if res.status_code == 200:
    models = res.json().get("models", [])
    for m in models:
        print(m["name"])
else:
    print("Error:", res.status_code, res.text)
