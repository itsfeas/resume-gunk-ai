text = """
Convert this photo to a photorealistic/hyperrealistic, 50 mm, Kodak Portra 800 photograph. 
The following food items are in the photo:
- A Sandwich
Description of Sandwich:
- The sandwich is made of two toasted pieces of bread.
- There are two beef patties.
- There is a layer of lettuce between the patties.
- There is a layer of sauce between the patties.
- The sandwich is on a white plate.
Description of Scene:
- The background is a black table.
- There is a black jacket on the table.
- There is a yellow magazine on the table.
"""

negative_text = """
blurry, 3d art, non-realistic, unidentifiable objects, words, drawing, extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured, warped metal, warped cutlery
"""

image_desc_prompt = """
    Describe the scene in this photo with an emphasis on the food. Name it and then provide 5-10 distinguishing characteristics about specifically its appearance. Then provide 2-5 characteristics about the background. Then provide information about the location of each item in the scene. Do this in the following json format:
    { 
        food_desc:  [a list of strings],
        background_desc: [a list of strings],
    }
"""