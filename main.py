import requests
import os
import sys
import numpy as np
sys.path.append("../")
from dotenv import load_dotenv
import random
import time
import string

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
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        return response.json()
    except requests.Timeout:
        print("API call timed out. Skipping this request.")
        return None

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

""""
Generates minor reworded version of input text by selecting word spans
and replacing them with AI-generated paraphrases.
    num_variants: number of perturbed texts to generate
    r: percentage of words to replace (perturbation rate)
    span_length: number of words in each replacement span 
"""
def generate_perturbations(text, num_variants=5, r=0.3, span_length =5):
    words = text.split() # tokenize text into words
    total_words = len(words)
    total_words_to_replace = int(total_words * r) # target number of words to be replaced (word span)
    span_to_replace = total_words_to_replace // span_length

    perturbed_texts = []

    for variant in range(num_variants):
        words_copy = words[:] # reference copy of words
        used_indices = set() # track replaced word indices
        spans_replaced = 0  # track how many spans have been replaced
        attempts = 0
        MAX_ATTEMPTS = 20  # Prevent infinite loops

        words_replaced = 0
        while words_replaced < total_words_to_replace and attempts < MAX_ATTEMPTS:
            start_idx = random.randint(0, max(0, total_words - span_length))

            if any(idx in used_indices for idx in range(start_idx, start_idx + span_length)):
                attempts += 1
                continue # avoids overlapping replacements

            span = words_copy[start_idx : start_idx + span_length]
            prompt = f"Paraphrase this sentence: " + " ".join(span)
            paraphrased_response = grok_chat_completion(prompt)

            if paraphrased_response:
                paraphrased_text = paraphrased_response.get("choices", [{}])[0].get("content", "").strip()
                if paraphrased_text:
                    # clean punctuation and split
                    clean_text = paraphrased_text.translate(str.maketrans('', '', string.punctuation))
                    paraphrased_tokens = clean_text.split()

                    if len(paraphrased_tokens) == span_length:
                        words_copy[start_idx: start_idx + span_length] = paraphrased_tokens
                        used_indices.update(range(start_idx, start_idx + span_length))
                        words_replaced += span_length
                        print(f"[INFO] Replaced {words_replaced}/{total_words_to_replace} words.")
                    else:
                        print(f"[WARN] Paraphrased span length mismatch. Expected {span_length}, got {len(paraphrased_tokens)}. Skipping.")
                else:
                    print("[WARN] Empty paraphrased response received.")
            else:
                print("[WARN] No paraphrased response received.")

            attempts += 1
            time.sleep(3)  # Avoid rate limits

        if attempts >= MAX_ATTEMPTS:
            print(f"[ERROR] Max attempts reached for variant {variant+1}. Moving on...")

        perturbed_texts.append(" ".join(words_copy))

    return perturbed_texts
    
# Computes score algoithm by computing curvature of log ratio
def detectgpt_score(text, num_samples=5):
    original_response = grok_chat_completion(text)
    original_log_prob = extract_logprobs(original_response)

    if original_log_prob is None:
        print("Error: Could not retrieve original logprob.")
        return None
    
    perturbed_texts = generate_perturbations(text, num_variants=num_samples)
    perturbed_log_probs = [] 

    # gets logprobs for each perturbated version
    for pt in perturbed_texts:
        perturbed_response = grok_chat_completion(pt)
        log_prob = extract_logprobs(perturbed_response)

        if log_prob is not None:
            perturbed_log_probs.append(log_prob) # list of probs
    
    if not perturbed_log_probs:
        print("Error: no valid logprobs for perturbed texts.")
        return None
    
    # curvature: average difference between og and perturbed log probs
    curvature = np.mean(np.array(perturbed_log_probs) - original_log_prob)
    return curvature


def classify_text(text, threshold=-0.1):
    score = detectgpt_score(text)
    if score is None:
        return "Undetermined"
    return "AI-Generated" if score < threshold else "Human-Written"

if __name__ == "__main__":
    content = "The global economy is expected to recover in the next fiscal year with significant growth in emerging markets."
    print(f"Original Text:\n{content}\n")

    print("Running Detection...\n")
    score = detectgpt_score(content)
    classification = classify_text(content)

    if score is not None:
        print(f"Detection Score (Curvature): {score:.4f}")
        print(f"Classification: {classification}")
    else:
        print("Detection failed due to missing log probabilities.") 
    #content = "What is the weather like today in DFW?"
    #response = grok_chat_completion(content)
    #print(response)



# # Testing extract_logprobs 
#     avg_logprob = extract_logprobs(response)  # Extract log probabilities
#     if avg_logprob is not None:
#         print(f"Average log probability: {avg_logprob}")
#     else:
#         print("Failed to calculate average log probability.")

    