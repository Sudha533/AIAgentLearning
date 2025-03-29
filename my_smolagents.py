# --- Direct HF API Test Script (Corrected) ---

import os
# ONLY import InferenceClient from huggingface_hub for this test
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_API_TOKEN")

if not token:
    print("HF_API_TOKEN not found in environment!")
else:
    print(f"HF Token Loaded: {token[:5]}...") # Optional: confirm token loaded

    try:
        # 1. Create the InferenceClient directly with the token
        client = InferenceClient(token=token)

        # 2. Define the model repository ID you want to test as a STRING
        model = "google/flan-t5-small"  # Or "google/flan-t5-large", etc.

        print(f"Attempting direct API call to model: {model}")

        # 3. Call text_generation, passing the STRING repo ID to the 'model' parameter
        response = client.text_generation(
            "Who is the president of France?",
            model=model
        )

        # 4. If successful, print the response
        print("Direct HF API Test Successful:")
        print(response)

    except Exception as e:
        # If it fails (e.g., 503 error, 401 bad token, etc.), print the error
        print(f"Direct HF API Test Failed: {e}")