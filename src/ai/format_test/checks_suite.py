import os
import sys

sys.path.extend([os.path.abspath(os.path.join("../line_split/"))])
sys.path.extend([os.path.abspath(os.path.pardir)])

from pdfminer.high_level import extract_pages, LAParams
import pdfminer.layout as layout
from typing import Iterator
from line_split.features import iter_lines
from line_split.length_check import get_lines
from checks.dates.dates_format_check import dates_format_check


def checks(pages: Iterator[layout.LTPage]):
    
    pass

if __name__ == "__main__":
    pdf_path = "../line_split/dataset/"
    pdf_file_name = "resume10.pdf" if not len(sys.argv)>1 else sys.argv[1]
    params = LAParams(char_margin=200, line_margin=1)
    lines = get_lines(extract_pages(pdf_path+pdf_file_name, laparams=params))
    dates = dates_format_check(lines)
    print(dates)
    # for d, l in zip(dates[1], lines):
    #     print(d, l.strip())