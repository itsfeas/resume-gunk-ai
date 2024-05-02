from collections import deque
from typing import Iterable, Iterator
import pdfminer.layout as layout

SECTIONS_DICT = {
	"projects": ["personal", "projects", "project"],
	"experience": ["work", "experience", "employment", "history"],
	"education": ["education"],
	"volunteering": ["volunteering", "volunteer"],
	"extracurriculars": ["extracurriculars", "extracurricular", "activities"],
	"summary": ["professional", "summary"],
	"achievements": ["achievements"],
	"awards": ["awards"],
	"certifications": ["certifications", "certification", "relevant", "courses"],
	"skills": ["technical", "skills"],
	"additional": ["additional", "other"]
}

# Create a new dictionary with the items in the value lists mapped to the key
KEYWORD_TO_SECTION_DICT = {}
for key, value_list in SECTIONS_DICT.items():
	for item in value_list:
		KEYWORD_TO_SECTION_DICT[item] = key

SECTION_HEADERS_SET = set()
for v in SECTIONS_DICT.values():
	SECTION_HEADERS_SET |= set(v)

def collect_headers(pages: Iterator[layout.LTPage]) -> list[layout.LTTextBoxHorizontal]:
    q = deque(list(pages))
    output = []
    # dfs to keep things in order
    prev = "name"
    while q:
        el = q.pop()
        if not isinstance(el, Iterable):
            continue
        for e in el:
            if isinstance(e, layout.LTTextLineHorizontal):
                line = e.get_text().lower().split()
                # print("LINE |", line)
                if len(line)<8:
                    for w in line:
                        if w in SECTION_HEADERS_SET:
                            output.append(e)
                            break
                # continue
            q.append(e)
    return output