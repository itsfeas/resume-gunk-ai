from functools import lru_cache
import re

special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')

@lru_cache(maxsize=1000)
def split_cached(s: str) -> list[str]:
    return s.split()

#replaces them with spaces
@lru_cache(maxsize=1000)
def strip_special_chars(s: str) -> str:
    return re.sub(special_char_pattern, ' ', s)