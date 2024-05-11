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

def get_length(pages: Iterator[layout.LTPage]) -> int:
    q = deque(list(pages))
    q.reverse()
    output = 0
    reg = re.compile(r"^\s+$")
    # dfs to keep things in order
    while q:
        el = q.pop()
        if not isinstance(el, Iterable):
            continue
        for e in el:
            if isinstance(e, layout.LTTextLineHorizontal) and reg.match(e.get_text()) is None:
                output += 1
            q.appendleft(e)
    return output

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
                output.append(e.get_text())
            q.appendleft(e)
    return output

if __name__ == "__main__":
    pdf_path = "./dataset/"
    label_path = "./dataset/labelled_lines/"
    pdf_files = list_files(pdf_path)
    for pdf_file_name in pdf_files:
        if not pdf_file_name.startswith("train_"):
            continue
        label_file_name = label_path+pdf_file_name
        label_file_name = label_file_name.replace(".pdf", ".txt")

        params = LAParams(char_margin=200, line_margin=1)
        pages = extract_pages(pdf_path+pdf_file_name, laparams=params)
        length_pdf = get_length(pages)
        with open(label_file_name, "r", encoding="utf8") as file:
            length_label = len([l for l in file.readlines() if l != "\n"])
            if length_label != length_pdf:
                print(f"NOT EQUAL LENGTH {pdf_file_name}: ({length_pdf}, {length_label})")
            else:
                print(f"PASSED {label_file_name}: ({length_pdf}, {length_label})")
        # img_files = list_files(img_path)[:max_files]
        # for file in img_files:
        # 	convert_img(img_path+file)