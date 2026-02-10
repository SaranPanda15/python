import requests # pyre-ignore-all-errors

url = "http://localhost:8080/analyze"
file_path = "test_fingerprint.png"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/png")}
    response = requests.post(url, files=files)

print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.json()}")
