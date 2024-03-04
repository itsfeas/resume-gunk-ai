from collections import defaultdict
from functools import cache
from os import walk
from os.path import join, dirname
import sys
from dotenv import load_dotenv
from pdfminer.high_level import extract_text, LAParams

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

def split_resume_to_sections(text: str, initial_class, section_header_set: set[str], keyword_to_section_dict: dict):
	tagger = [["", l.lower()] for l in text.splitlines() if l]
	prev = initial_class
	for i, tagged_line in enumerate(tagger):
		if len(tagged_line[1])<50:
			for w in tagged_line[1].split():
				if w in section_header_set:
					prev = keyword_to_section_dict[w]
					break
		tagger[i][0] = prev
	return tagger

# def get_work_exp_entries(tagged_lst: list[list[str]]):
# 	for 


if __name__ == "__main__":
	# Load environment variables from the .env file
	section_headers_set = set()
	for v in SECTIONS_DICT.values():
		section_headers_set |= set(v)
	dotenv_path = join(dirname(__file__), '../../.env')
	load_dotenv(dotenv_path)
	file = "resume_fiaz.pdf"
	path = f"./dataset/{file if len(sys.argv)<2 else sys.argv[1]}"
	text = extract_text(path, laparams=LAParams(char_margin=200, line_margin=1))
	tagged = split_resume_to_sections(text, "name", section_headers_set, KEYWORD_TO_SECTION_DICT)
	for r in tagged:
		print(r)
	
	# Random Sampling
	# df = random_sampling_remove_null(df, 0.8)

