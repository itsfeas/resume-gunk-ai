text = """
Convert this photo to a photorealistic, 50 mm, Kodak Portra 800 photograph. 
Description of Food: The sandwich is made of two toasted pieces of bread, has two burger patties, there is a layer of lettuce between the patties, a layer of sauce between the patties, and is on a white plate.
Description of Scene: The background is a black table. There is a black jacket and a yellow magazine on the table.
"""

negative_text = """
blurry, 3d art, non-realistic, unidentifiable objects, words, drawing, extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured, warped metal, warped cutlery
"""

image_desc_prompt = """
Describe the scene in this photo with an emphasis on the food. Do this in the following json format:
{ 
    food_desc:  [format: a list of strings. Name all food items, providing 7-10 distinguishing characteristics about specifically their appearance],
    background_desc: [format: a list of strings. provide 2-5 characteristics about the background],
    spatial_info: [format: a list of strings. 2-5 characteristics about the location of each item in the scene. provide information on which quadrant of the picture each item lies and how much of the picture each item occupies].
}
"""