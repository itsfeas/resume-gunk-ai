from itertools import chain
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile, common_texts
from collections import deque
from os import walk
import re
from typing import Iterable, Iterator
import pdfminer.layout as layout
from pdfminer.high_level import extract_pages, LAParams

def list_files(mypath) -> list[str]:
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    return(f)

def get_lines(pages: Iterator[layout.LTPage]) -> list[str]:
    q = deque(list(pages))
    q.reverse()
    output = []
    reg = re.compile(r"^\s+$")
    # dfs to keep things in order
    while q:
        el = q.pop()
        if not isinstance(el, Iterable):
            continue
        for e in el:
            if isinstance(e, layout.LTTextLineHorizontal) and reg.match(e.get_text()) is None:
                output.append(e.get_text().lower())
            q.appendleft(e)
    return output

if __name__ == "__main__":
    datasets_paths = ["./dataset/", "./dataset/sentences/"]
    tot_lines = []
    for dataset_path in datasets_paths:
        pdf_files = list_files(dataset_path)
        for pdf_file_name in pdf_files:
            # if not pdf_file_name.startswith("train_"):
            #     continue
            params = LAParams(char_margin=200, line_margin=1)
            pages = extract_pages(dataset_path+pdf_file_name, laparams=params)
            tot_lines += get_lines(pages)
    print(f"total_lines: {len(tot_lines)+len(common_texts)}")
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(chain(tot_lines, common_texts))]
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
    model.save("doc2vec_model")