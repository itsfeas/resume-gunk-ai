import os
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import requests
import io
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/thibaud/sdxl_dpo_turbo"

def query():
	apiToken = os.environ.get("OPENAI_API_TOKEN")
	org = os.environ.get("OPENAI_ORGANIZATION")
	headers = {"Authorization": f"Bearer {apiToken}"}

	template = """Question: {question}

	Answer: Let's think step by step."""

	prompt = PromptTemplate(template=template, input_variables=["question"])

	llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=apiToken, organization=org)
	llm_chain = LLMChain(prompt=prompt, llm=llm)

	question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

	# image = Image.open(io.BytesIO(content))
	# image.save("image.png")
	return llm_chain.run(question)