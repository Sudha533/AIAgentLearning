import PIL.Image
import io
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from transformers import pipeline

# Load API token
load_dotenv()
token = os.getenv("HF_API_TOKEN")

if not token:
    print("HF_API_TOKEN not found in environment!")
    exit()

# Initialize the InferenceClient
client = InferenceClient(token=token)

# 🔹 AI Agent Function
def ai_agent(task, model=None):
    """
    Determines whether to generate text or an image based on the input task.
    """

    class_prompt = f"""
    classify the following request as either 'IMAGE' or 'TEXT':
    Request = {task}
    Classification:
    """

    intent_classification = client.text_generation(
        class_prompt,
        model="google/flan-t5-small",
        max_new_tokens=10
    ).strip()

#    if any(word in task.lower() for word in ["image", "picture", "photo", "art", "drawing", "illustration"]):
    if "IMAGE" in intent_classification.upper():
#        model = model or "stabilityai/stable-diffusion-2-1-base"
        model="runwayml/stable-diffusion-v1-5"
        print(f"\n🎨 Generating an image for: {task}")
        
        image = client.text_to_image(task, model=model, height=512, width=512)
        
        # Convert bytes to an image and display it
        if isinstance(image, PIL.Image.Image):
            image.show()  
        else:
        # If it's bytes, convert to an image
            image = PIL.Image.open(io.BytesIO(image))
            image.show()
        return image  # Returns the generated image
    
    else:
        model = "google/flan-t5-small"
        print(f"\n📝 Generating text for: {task}")

#        response = client.text_generation(task, model=model)
        generator = pipeline("text-generation", model="distilgpt2")
        response = generator(task, max_length=30)
        print("\n💡 AI Response:\n", response)
        return response  # Returns the generated text

# 🔹 Example Calls
ai_agent("Who is the president of France?")  # Text generation
ai_agent("A futuristic city skyline at sunset")  # Image generation
ai_agent("Create a drawing of a dragon flying over a mountain")  # Image generation
ai_agent("Write a poem about the sea")  # Text generation
