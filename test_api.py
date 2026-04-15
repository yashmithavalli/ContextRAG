import requests

API_URL = "http://localhost:8000"

print("1. Uploading...")
with open("apollo11.txt", "rb") as f:
    res = requests.post(f"{API_URL}/upload", files={"file": ("apollo11.txt", f, "text/plain")})
    print("Upload Status:", res.status_code)
    print("Upload Response:", res.json())

print("\n2. Querying...")
query = {"question": "Who were the primary astronauts on Apollo 11 and what was their spacecraft called?"}
res2 = requests.post(f"{API_URL}/query", json=query)
print("Query Status:", res2.status_code)
with open("query_out.txt", "w") as out:
    out.write(str(res2.json()))
