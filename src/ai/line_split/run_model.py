from collections import defaultdict
from functools import cache
from os.path import join, dirname
import sys
from dotenv import load_dotenv
import joblib
from features import iter_lines
from length_check import get_lines
from pdfminer.high_level import extract_pages, LAParams

MODEL_NAME = "resume_parser_split"
ORDINAL_ENCODER_NAME = "ordinal_encoder_split"

if __name__ == "__main__":
    # Load environment variables from the .env file
    dotenv_path = join(dirname(__file__), 'env/.env')
    load_dotenv(dotenv_path)

    pdf_path = "./dataset/"
    pdf_file_name = "resume10.pdf" if not len(sys.argv)>1 else sys.argv[1]
    params = LAParams(char_margin=200, line_margin=1)
    df = iter_lines(extract_pages(pdf_path+pdf_file_name, laparams=params))
    lines = get_lines(extract_pages(pdf_path+pdf_file_name, laparams=params))
    
    print("Loading Model...")
    loaded_model = joblib.load(open(MODEL_NAME, 'rb'))
    loaded_encoder = joblib.load(open(ORDINAL_ENCODER_NAME, 'rb'))
    
    print("Predicting...")
    prediction = loaded_model.predict(df)
    prediction = loaded_encoder.inverse_transform(prediction.reshape(-1, 1))
    
    print(df)
    # print(prediction)
    assert len(prediction) == len(lines)
    i = 0
    for pred, x in zip(prediction, lines):
        print(i, pred[0], x.strip())
        i += 1