import requests
import os
import sys
sys.path.append("../")
from dotenv import load_dotenv

def grok_chat_completion(content, model="grok-2-latest", stream=False, temperature=0):
    load_dotenv()
    api_key = os.getenv("GROK_2_API_KEY")
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "messages": [
            # {
            #     "role": "system",
            #     "content": "You are a test assistant."
            # },
            {
                "role": "user",
                "content": f"{content}"
            }
        ],
        "model": f"{model}",
        "stream": stream,
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

if __name__ == "__main__":
    content = "What is the weather like today in DFW?"
    response = grok_chat_completion(content)
    print(response)
    