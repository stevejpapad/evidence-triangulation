import pandas as pd
import time, os

import base64
import mimetypes
import json
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig



load_dotenv()
GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY")


def mm_inference_google(
    model,
    user_prompt=None,
    image_urls=None,
    max_tokens=1024,
    temperature=0.2
):
    """
    Multimodal inference wrapper for Google GenAI.
    """

    client = genai.Client(api_key=GOOGLE_AI_STUDIO_API_KEY)

    if not user_prompt:
        raise ValueError("user_prompt must be provided.")


    # Construct user parts
    user_parts = [{"text": user_prompt}]

    if image_urls:
        if isinstance(image_urls, str):
            image_urls = [image_urls]
        for url in image_urls:
            user_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": url
                }
            })

    # Prepare contents
    contents = [{"role": "user", "parts": user_parts}]

    config = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens
    )
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        return response.text
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


def image_to_base64(path, max_size=512):
    """
    Loads a local image, resizes it (keeping aspect ratio), and returns a base64 data URL.
    """
    # Some images are png but mistakenly are saved as jpg in dataset column
    if not os.path.exists(path) and path.endswith(".jpg"):
        path = path.replace(".jpg", ".png")

    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/jpeg"  # fallback

    # Load and resize image
    with Image.open(path) as img:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)


        # Save to in-memory buffer
        buffered = BytesIO()
        format = "JPEG" if mime_type == "image/jpeg" else "PNG"
        img.save(buffered, format=format)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return encoded


