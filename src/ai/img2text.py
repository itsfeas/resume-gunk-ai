import os
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain, SimpleSequentialChain, create_extraction_chain
from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.schema.messages import HumanMessage, SystemMessage
from vertexai.generative_models._generative_models import ResponseBlockedError

def gemini_img2desc(prompt: str, img_url: str):
	llm = ChatVertexAI(model_name="gemini-pro-vision", max_output_tokens=70, temperature=0.2, safety_settings = [
        # { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
		# { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
		# { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
		# { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
		])
	image_message = {
		"type": "image_url",
		"image_url": {
			"url": img_url,
		},
	}
	text_message = {
		"type": "text",
		"text": prompt,
	}
	message = HumanMessage(content=[text_message, image_message])

	output = None
	# output = llm.invoke([message])
	i = 0
	while not output and i<5:
		try:
			output = llm.invoke([message])
		except ResponseBlockedError:
			print("Gemini call failed. Retrying...")
		i += 1
	print('before strip prefix:', output.content)
	return _strip_prefix(output.content.strip(), "A photo with")

# def dalle_img2img(prompt: str):
# 	llm = DallEAPIWrapper(model="dall-e-3")
# 	image_url = llm.run(prompt)
# 	print(image_url)
# 	return image_url

def _strip_prefix(original_string: str, prefix_to_remove: str):
	if original_string.startswith(prefix_to_remove):
		formatted = original_string[len(prefix_to_remove):].strip()
	else:
		formatted = original_string
	return formatted