from repository.gen_queue_repo import get_one_queued_img
from repository.img_download_repo import get_url_from_img_id
from ai.img2text import gemini_img2desc
from ai.replicate import img2img_sdxl_cn
from ai.prompt.v5 import img2desc_prompt, img2img_prompt, img2img_negative


def workflow(img_id: str):
    img_url = get_url_from_img_id(img_id)

    desc = gemini_img2desc(img2desc_prompt, img_url)
    final_prompt = img2img_prompt.format(scene_desc=desc)

    print(desc)
    print("final prompt: ", final_prompt)
    
    gen_img_url = img2img_sdxl_cn(final_prompt, img2img_negative, img_url)
    print(gen_img_url)
    return gen_img_url[-1]