import os
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from huggingface_hub import InferenceClient
import torch
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/thibaud/sdxl_dpo_turbo"

def query():
	apiToken = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

	client = InferenceClient(token=apiToken)

	img = client.text_to_image(
		prompt="An astronaut riding a horse on the moon.",
		model="stabilityai/sdxl-turbo",
	)
	img.save("image.png")
	return "success"
