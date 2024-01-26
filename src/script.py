from os.path import join, dirname
from dotenv import load_dotenv
import db.rtdb.client as rtdb
import db.storage.client as storage
from repository.gen_queue_repo import get_one_queued_img
from repository.img_download_repo import get_url_from_img_id
from ai.img2text import gemini_img2desc
from ai.replicate import img2img_sdxl_cn
from ai.prompt.v5 import img2desc_prompt, img2img_prompt, img2img_negative
from langchain.utilities.dalle_image_generator import DallEAPIWrapper

if __name__ == "__main__":
    # Load environment variables from the .env file
    dotenv_path = join(dirname(__file__), 'env/.env')
    load_dotenv(dotenv_path)

    llm = DallEAPIWrapper(model="dall-e-3")
    image_url = llm.run("society if we used lisp (the programming lang)")
    print(image_url)

    # rtdb.init_client()
    # # storage.init_client()

    # # img_id = get_one_queued_img()["image_id"]
    # img_id = "d7addae6-f6bc-4796-95bb-5ca7d969047e" # sandwich
    # # img_id = "3c6220fb-b891-4e38-9ea7-336fe5354675" # ice cream
    # img_url = get_url_from_img_id(img_id)
    # desc = gemini_img2desc(img2desc_prompt, img_url)
    # # desc = "A toasted bread bun with two beef patties, lettuce, and special sauce on a white plate. The background includes a black jacket on a hanger and a magazine on a table."
    # print(desc)
    # gen_img_url = img2img_sdxl_cn(img2img_prompt.format(scene_desc=desc), img2img_negative, img_url)
    # print(gen_img_url[-1])