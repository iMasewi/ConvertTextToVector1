import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_SERVER_URL", "http://127.0.0.1:11434")

def summarize_context(context: str) -> str:
    payload = {
      "model": "deepseek-r1:8b",
      "messages": [
         {"role": "user", "content": f"Tóm tắt ngắn gọn nội dung sau: {context}"}
      ],
      "stream": False
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
