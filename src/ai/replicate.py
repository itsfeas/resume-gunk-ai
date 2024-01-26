import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Replicate

def img2img_sdxl(prompt: str, negative_prompt: str, image_url: str):
	apiToken = os.environ.get("REPLICATE_API_TOKEN")

	img2img = Replicate(
		model="stability-ai/sdxl:610dddf033f10431b1b55f24510b6009fcba23017ee551a1b9afbc4eec79e29c",
		image_dimensions= "1024x1024",
		apiToken=apiToken
	)
	ret = img2img(
		image = image_url,
		width = 768,
		height = 768,
		prompt = prompt,
		refine = "expert_ensemble_refiner",
		scheduler = "K_EULER",
		lora_scale = 0.6,
		num_outputs = 1,
		guidance_scale = 7.21,
		apply_watermark = False,
		high_noise_frac = 0.9, #default: 0.8
		negative_prompt = negative_prompt,
		prompt_strength = 0.55,
		disable_safety_checker = True,
		num_inference_steps = 67
	)
	return ret

# TO-DO: Add error handling
def img2img_sdxl_cn(prompt: str, negative_prompt: str, image_url: str) -> str:
	apiToken = os.environ.get("REPLICATE_API_TOKEN")

	img2img = Replicate(
		model="fofr/sdxl-multi-controlnet-lora:89eb212b3d1366a83e949c12a4b45dfe6b6b313b594cb8268e864931ac9ffb16",
		image_dimensions= "1024x1024",
		apiToken=apiToken
	)
	# ret = img2img(
	# 	image = image_url,
	# 	width = 1024,
	# 	height = 1024,
	# 	prompt = prompt,
	# 	refine = "base_image_refiner",
	# 	scheduler = "K_EULER",
	# 	lora_scale = 0.9,
	# 	num_outputs = 1,
	# 	controlnet_1 = "lineart",
	# 	controlnet_2 = "edge_canny",
	# 	controlnet_3 = "none",
	# 	lora_weights = "https://storage.googleapis.com/loap-img-storage/food_blooger_iii.tar",
	# 	guidance_scale = 7.5,
	# 	apply_watermark = False,
	# 	negative_prompt = negative_prompt,
	# 	prompt_strength = 0.7,
	# 	sizing_strategy = "input_image",
	# 	controlnet_1_end = 1,
	# 	controlnet_2_end = 1,
	# 	controlnet_3_end = 1,
	# 	controlnet_1_image = image_url,
	# 	controlnet_1_start = 0,
	# 	controlnet_2_image = image_url,
	# 	controlnet_2_start = 0,
	# 	controlnet_3_start = 0,
	# 	num_inference_steps = 50,
	# 	controlnet_1_conditioning_scale = 0.3,
	# 	controlnet_2_conditioning_scale = 0.3,
	# 	controlnet_3_conditioning_scale = 0.75
	# )
	ret = img2img(
		image = image_url,
		width = 1024,
		height = 1024,
		prompt = prompt,
		refine = "base_image_refiner",
		scheduler = "K_EULER",
		lora_scale = 0.8,
		num_outputs = 1,
		controlnet_1 = "lineart",
		controlnet_2 = "edge_canny",
		controlnet_3 = "depth_midas",
		lora_weights = "https://storage.googleapis.com/loap-img-storage/food_blooger_iii.tar",
		guidance_scale = 10,
		apply_watermark = False,
		negative_prompt = negative_prompt,
		prompt_strength = 0.2,
		sizing_strategy = "input_image",
		controlnet_1_end = 1,
		controlnet_2_end = 1,
		controlnet_3_end = 1,
		controlnet_1_image = image_url,
		controlnet_1_start = 0,
		controlnet_2_image = image_url,
		controlnet_2_start = 0,
		controlnet_3_image = image_url,
		controlnet_3_start = 0,
		num_inference_steps = 60,
		controlnet_1_conditioning_scale = 4,
		controlnet_2_conditioning_scale = 2.5,
		controlnet_3_conditioning_scale = 2.5
	)
	prefix = 'https'
	return [prefix + el for el in ret.split(prefix)]