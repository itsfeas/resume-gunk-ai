img2img_prompt = """
A photo of TOK, with {scene_desc}
"""

img2img_negative = """
blurry, 3d art, non-realistic, unidentifiable objects, words, drawing, extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured, warped metal, warped cutlery
"""

# image_desc_prompt = """
# Describe the scene in this photo with an emphasis on the food, within 180 characters.
# Use the following thought process:
# a) Count the food items in the photo. Don't include this in the description.
# b) Provide visually distinguishing characteristics about each food item
# c) Describe the background very briefly
# """

img2desc_prompt = """
Describe the scene in this photo with an emphasis on the food, within 180 characters.
Use the following thought process:
a) Provide visually distinguishing characteristics about each food entree.
b) Describe the background very briefly. Don't describe the placement of food items on plates.
Continue from "A photo with ..."
"""

tough_to_reproduce_items = ["crumpled plastic, cutlery, chopsticks"]