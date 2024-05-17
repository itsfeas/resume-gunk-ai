from collections import defaultdict, deque
from functools import lru_cache
import pdfminer.layout as layout
from typing import Iterator, Optional, Union
from enum import Enum
import dateparser
from ..helper_func.helpers import split_cached, strip_special_chars
from ..model.status import CheckStatus, TestResult
from ..model.checks import Checks
import re
import datetime

class DateSegmentFormat(Enum):
    SHORTHAND = "shorthand"
    FULLSIZE = "fullsize"
    ABSOLUTE = "absolute"
    # YEAR = "year"

SHORTHAND_SET = { 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec'}
FULLSIZE_SET = { 'january', 'february', 'august', 'september', 'october', 'november', 'december' }
ABSOLUTE_SET = {'march', 'april', 'may', 'june', 'july'}

# formats are shorthand, fullsize, absolute (March, April, May, June and July) and year
def identify_form_distribution(lines: list[str]) -> dict[str, int]:
    d = defaultdict(int)
    for line in lines:
        l = line.lower()
        l = strip_special_chars(l)
        words = split_cached(l)
        for w in words:
            if w in SHORTHAND_SET:
                d[DateSegmentFormat.SHORTHAND]+=1
            elif w in FULLSIZE_SET:
                d[DateSegmentFormat.FULLSIZE]+=1
            elif w in ABSOLUTE_SET:
                d[DateSegmentFormat.ABSOLUTE]+=1
    return d

def check_dates(lines: list[str]) -> tuple[int, list[str]]:
    q = []
    date_regex = r"\bpresent$|\d{4}|(?:jan(?:\.|uary)?|feb(?:\.|ruary)?|mar(?:\.|ch)?|apr(?:\.|il)?|may|jun(?:\.|e)?|jul(?:\.|y)?|aug(?:\.|ust)?|sep(?:\.|tember)?|oct(?:\.|ober)?|nov(?:\.|ember)?|dec(?:\.|ember)?)(?:\s+\d{4})?\b"
    for i, line in enumerate(lines[2:]):
        l = strip_special_chars(line.lower()).strip()
        dates = re.findall(date_regex, l)
        # if date is in format MONTH_A - MONTH_B YEAR
        if len(dates) == 2 and len(dates[1].split(" "))==2 and len(dates[0].split(" "))==1:
            dates[0]+=" "+dates[1].split(" ")[1] # TO-DO: improve performance
        if dates:
            # prev = 0
            # [print() for d in dates]
            # converted_to_num = [date_str_to_num(d) for d in dates]
            # for d in dates:
            #     if not isinstance(d, int):

            q.append((i+2, dates))
    return q

MONTH_MAPPING = {
    'january':      1,
    'jan':          1,
    'february':     2,
    'feb':          2,
    'march':        3,
    'mar':          3,
    'april':        4,
    'apr':          4,
    'may':          5,
    'may':          5,
    'june':         6,
    'jun':          6,
    'july':         7,
    'jul':          7,
    'august':       8,
    'aug':          8,
    'september':    9,
    'sep':          9,
    'october':      10,
    'oct':          10,
    'november':     11,
    'nov':          11,
    'december':     12,
    'dec':          12
}
# assumes month is lowercase
@lru_cache(maxsize=1000)
def date_str_to_num(s: str) -> Union[int, tuple[CheckStatus, str]]:
    lst = s.split(" ")
    sumi = 0
    if len(lst) == 1 and lst[0] in MONTH_MAPPING:
        return (CheckStatus.FAILURE, "Month mentioned in line without associated year")
    for it in lst:
        if it in MONTH_MAPPING:
            sumi += MONTH_MAPPING[it]
        elif it.isnumeric():
            sumi += 12*int(it)
        elif it == "present":
            now = datetime.datetime.now()
            sumi += now.year*12+now.month
    return sumi

# accepts only plaintext
def dates_format_check(lines: list[str]):
    results: list[TestResult] = []
    format_hist = identify_form_distribution(lines)
    total_dates = sum(v for v in format_hist.values())
    parse_dates = check_dates(lines)
    print(total_dates, format_hist)
    
    if 1 <= len(format_hist) <= 2:
        results.append(TestResult(CheckStatus.SUCCESS, Checks.date_format_form_distribution, ""))
    elif len(format_hist) == 3:
        results.append(TestResult(CheckStatus.FAILURE, Checks.date_format_form_distribution,
            "You are mixing the shorthand and unabbreviated versions of months in your resume!"))
    
    if total_dates == 0 or len(parse_dates) == 0:
        results.append(TestResult(CheckStatus.WARNING, Checks.dates_are_present,
            "No dates detected in your resume! Ensure dates have been formatted correctly!"))
    # TO-DO: check dates to ensure they are formatted as intervals
    

    print(parse_dates)