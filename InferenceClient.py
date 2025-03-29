import PIL.Image
import io  # Required for handling byte data
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
token = os.getenv("HF_API_TOKEN")

if not token:
    print("HF_API_TOKEN not found in environment!")
else:
    print(f"HF Token Loaded: {token[:5]}...")  # Optional: Confirm token

    try:
        # Initialize the InferenceClient with the token
        client = InferenceClient(token=token)

        # --- TEXT GENERATION ---
        model = "google/flan-t5-small"  # Example model
        print(f"Calling model: {model}")

        response = client.text_generation(
            "Who is the president of France?",
            model=model
        )
        print("\n--- Text Generation Response ---\n", response)

        # --- TEXT TO IMAGE ---
        print("\nGenerating image from text...")
        image = client.text_to_image(
            "A futuristic city skyline at sunset",
            model="stabilityai/stable-diffusion-2-1-base",
            height=512,
            width=512
        )

        print("Image generation successful!")

        # Convert bytes to an image
        if isinstance(image, PIL.Image.Image):
            image.show()  # Display the image
        else:
        # If it's bytes, convert to an image
            image = PIL.Image.open(io.BytesIO(image))
            image.show()

    except Exception as e:
        print(f"Error during inference: {e}")
