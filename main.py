import requests
import os
import sys
sys.path.append("../")
from dotenv import load_dotenv

def grok_chat_completion(content, model="grok-2-latest", stream=False, temperature=0, logprobs=True):
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
        "temperature": temperature,
        "logprobs": logprobs
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Extracts log probabilities from response and computes their average.
def extract_logprobs(response):
    choices = response.get("choices", [])

    if not choices:
        print("No choices returned in response.")
        return None
    
    first_choice = choices[0] 
    logprobs = first_choice.get("logprobs", {}) # gets logprobs of first choice

    if not logprobs or "content" not in logprobs:
        print("No logprobs returned in response.")
        return None
    
    token_logprob = [entry["logprob"] for entry in logprobs["content"] if "logprob" in entry] # gets logprob of each token

    if not token_logprob:
        print("No logprob returned in reponse.")
        return None
    
    avg_logprob = sum(token_logprob) / len(token_logprob) 
    return avg_logprob


if __name__ == "__main__":
    content = "What is the weather like today in DFW?"
    response = grok_chat_completion(content)
    print(response)

    avg_logprob = extract_logprobs(response)  # Extract log probabilities
    if avg_logprob is not None:
        print(f"Average log probability: {avg_logprob}")
    else:
        print("Failed to calculate average log probability.")

    