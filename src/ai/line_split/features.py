from collections import defaultdict, deque
from os import walk
import re
from typing import Iterable, Iterator
import joblib
import numpy as np
import pdfminer.layout as layout
from pdfminer.high_level import extract_pages, LAParams
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from functools import cache, lru_cache
import xgboost

embeddings_model = Doc2Vec.load("doc2vec_model")

SECTIONS_DICT = {
	"projects": ["personal", "projects", "project"],
	"experience": ["work", "experience", "employment", "history"],
	"education": ["education"],
	"volunteering": ["volunteering", "volunteer"],
	"extracurriculars": ["extracurriculars", "extracurricular", "activities"],
	"summary": ["professional", "summary"],
	"achievements": ["achievements"],
	"awards": ["awards"],
    "training": ["training", "trainings"],
	"certifications": ["certifications", "certification", "certificates", "relevant", "courses"],
	"skills": ["technical", "skills"],
	"additional": ["additional", "other", "information", "info"],
}

KEYWORD_TO_SECTION_DICT = {}
for key, value_list in SECTIONS_DICT.items():
	for item in value_list:
		KEYWORD_TO_SECTION_DICT[item] = key

def list_files(mypath) -> list[str]:
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    return(f)

def derive_features_for_line(l_init: str) -> pd.DataFrame:
    l = l_init.lower()
    special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')
    # vec = embeddings_model.infer_vector([l])
    # df_embeddings = pd.DataFrame(vec.reshape(1, -1)).add_prefix("embeddings_")
    keyword_cnt_dict = defaultdict(int)
    keyword_cnt_vec = [0]*len(SECTIONS_DICT.keys())
    l = re.sub(special_char_pattern, '', l)
    for w in l.split():
        if w in KEYWORD_TO_SECTION_DICT:
            keyword_cnt_dict[KEYWORD_TO_SECTION_DICT[w]]+=1
    for i, section in enumerate(iter(SECTIONS_DICT.keys())):
        keyword_cnt_vec[i] = keyword_cnt_dict[section]/len(SECTIONS_DICT[section])
    df_keyword_cnt_vec = pd.DataFrame(np.array(keyword_cnt_vec).reshape(1, -1)).add_prefix("keyword_cnt_vec_")

    # re_num = re.compile("\d+")
    # numeric = [1 if re_num.match(l) is not None else 0]
    # df_numeric = pd.DataFrame(np.array(numeric).reshape(1, -1)).add_prefix("numeric_")

    # capitalized = [1 if l_init.isupper() else 0]
    # df_capitalized = pd.DataFrame(np.array(capitalized).reshape(1, -1)).add_prefix("capitalized_")

    # short_length = [1 if len(l)<20 else 0, 1 if len(l)<40 else 0]
    short_length = [1 if len(l)>100 else len(l)/100]
    df_short_length = pd.DataFrame(np.array(short_length).reshape(1, -1)).add_prefix("short_length_")

    df = pd.concat([df_keyword_cnt_vec, df_short_length], axis=1)
    return df

@cache
def is_bold_font(font: str) -> bool:
    return font.lower().find("bold")>0

@lru_cache(maxsize=1000)
def get_strength(font: str, size: int, is_upper: bool) -> bool:
    return size*(1.20 if is_bold_font(font) or is_upper else 1)

def iter_lines(pages: Iterator[layout.LTPage]) -> pd.DataFrame:
    q = deque(list(pages))
    q.reverse()
    output_by_line = []
    reg = re.compile(r"^\s+$")
    # bfs to keep things in order
    text_strength_meta = [] # strength is a heuristic that takes into account (maybe) capitalization, font size, weight
    textbox_pos = []
    while q:
        el = q.pop()
        if not isinstance(el, Iterable):
            continue
        if isinstance(el, layout.LTTextBoxHorizontal):
            cnt = 0
        for e in el:
            if isinstance(e, layout.LTTextLineHorizontal) and reg.match(e.get_text()) is None:
                features = derive_features_for_line(e.get_text())
                output_by_line.append(features)
                tot, num = 0, 0
                for char in e:
                    if char.__class__ == layout.LTChar:
                        tot += get_strength(char.fontname, char.size, char.get_text().isupper())
                        num += 1
                text_strength_meta.append(tot/num)
                textbox_pos.append(cnt)
                cnt += 1
            q.appendleft(e)
    text_strength_meta = np.array(text_strength_meta)
    text_strength_meta = pd.Series((text_strength_meta - text_strength_meta.mean()) / text_strength_meta.std(), name="text_size_meta")
    # print(textbox_pos)
    textbox_pos = pd.Series(textbox_pos, name="textbox_pos")
    # df_text_size_meta = pd.concat([text_size_meta], axis=0, ignore_index=True).add_prefix("text_size_meta_")
    df_output_by_line = pd.concat(output_by_line, axis=0, ignore_index=True)
    return pd.concat([df_output_by_line, text_strength_meta, textbox_pos], axis=1)