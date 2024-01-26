from os.path import join, dirname
from dotenv import load_dotenv
from pdfminer.high_level import extract_pages

if __name__ == "__main__":
    # Load environment variables from the .env file
    dotenv_path = join(dirname(__file__), 'env/.env')
    load_dotenv(dotenv_path)

    for page_layout in extract_pages("./src/dataset/resume.pdf"):
        for element in page_layout:
            print(element)

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