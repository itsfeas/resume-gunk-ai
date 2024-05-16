from collections import defaultdict, deque
import pdfminer.layout as layout
from typing import Iterator, Optional, Union
from enum import Enum
from ..helper_func.helpers import split_cached, strip_special_chars
from ..model.status import CheckStatus, TestResult
from ..model.checks import Checks
import re

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
    date_regex = r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\b"
    for i, line in enumerate(lines):
        l = line.lower()
        dates = re.findall(date_regex, l)
        if dates:
            q.append((i, dates))
    return q


# accepts only plaintext
def dates_format_check(lines: list[str]):
    results: list[TestResult] = []
    format_hist = identify_form_distribution(lines)
    total_dates = sum(v for v in format_hist.values())
    print(total_dates, format_hist)
    if 1 <= len(format_hist) <= 2:
        results.append(TestResult(CheckStatus.SUCCESS, Checks.date_format_form_distribution, ""))
    elif len(format_hist) == 3:
        results.append(TestResult(CheckStatus.FAILURE, Checks.date_format_form_distribution,
            "You are mixing the shorthand and unabbreviated versions of months in your resume!"))
    elif len(format_hist) == 0:
        results.append(TestResult(CheckStatus.WARNING, Checks.date_format_form_distribution,
            "No dates detected in your resume! Ensure dates have been formatted correctly!"))
    parse_dates = check_dates(lines)
    # TO-DO: check dates to ensure they are formatted as intervals
    

    print(parse_dates)