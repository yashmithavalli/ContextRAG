import requests

API_URL = "http://localhost:8000"

def run_test():
    print("Writing sample article locally...")
    text = (
        "Apollo 11 was the American spaceflight that first landed humans on the Moon. "
        "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar "
        "Module Eagle on July 20, 1969. Armstrong became the first person to step onto the lunar "
        "surface six hours and 39 minutes later on July 21. Aldrin joined him 19 minutes later, "
        "and they spent about two and a quarter hours together exploring the site they had named "
        "Tranquility Base upon landing. Michael Collins flew the Command Module Columbia alone in "
        "lunar orbit."
    )
    with open("apollo11.txt", "w", encoding="utf-8") as f:
        f.write(text)
        
    print("\n1. Uploading to ContextRAG...")
    with open("apollo11.txt", "rb") as f:
        res = requests.post(f"{API_URL}/upload", files={"file": ("apollo11.txt", f, "text/plain")})
        print("Upload Response:", res.json())
        
    print("\n2. Querying Gemini LLM...")
    query = {"question": "Who were the primary astronauts on Apollo 11 and what was their spacecraft called?"}
    res = requests.post(f"{API_URL}/query", json=query)
    
    data = res.json()
    print("Query Full Response:", res.status_code, data)
    print("\n================ GEMINI ANSWER ================")
    print(data.get("answer", "NO ANSWER"))
    print("===============================================\n")
    print(f"Sources utilized: {len(data.get('sources', []))}")
    
if __name__ == "__main__":
    run_test()
