"""Anchor relative time expressions in exact Evidence to the date they were said on.

"I ran a race last Friday", said on 25 May 2023, means the Friday before 25 May 2023. A reader
seeing the span weeks later cannot recover that without the anchor. ``annotate`` keeps the exact
text and appends a bracketed resolution after each expression, phrased the way people answer
("the Friday before 25 May 2023", "May 2023"), never a guessed calendar day for a vague phrase.
Deterministic, zero model calls.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "couple of": 2, "few": 3,
}
_SEASONS = {  # (first month, last month); winter spans the new year and is named by its start
    "spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11), "winter": (12, 2),
}
_UNIT = r"(day|week|month|year)s?"
_PATTERN = re.compile(
    r"\b(?:"
    r"(?P<simple>yesterday|today|tonight|tomorrow|last night|this morning|this afternoon"
    r"|this evening)"
    r"|(?P<rel>last|this past|past|this|next|coming) (?P<what>week|weekend|month|year"
    r"|summer|winter|spring|fall|autumn|" + "|".join(_WEEKDAYS) + r")"
    r"|(?:(?P<vague>a few|a couple of|few|couple of|several)|(?P<count>\d+|a|an|one|two|three"
    r"|four|five|six|seven|eight|nine|ten)) " + _UNIT + r" ago"
    r")\b",
    re.IGNORECASE,
)


def _day(value: datetime) -> str:
    return f"{value.day} {value:%B %Y}"


def _month(value: datetime) -> str:
    return f"{value:%B %Y}"


def _shift_months(value: datetime, months: int) -> datetime:
    index = value.year * 12 + value.month - 1 + months
    return value.replace(year=index // 12, month=index % 12 + 1, day=1)


def resolve(expression: str, anchor: datetime) -> str | None:
    """Return a human answer-style resolution of one relative expression, or None."""

    match = _PATTERN.fullmatch(expression.strip())
    if match is None:
        return None
    anchor_day = _day(anchor)
    simple = (match.group("simple") or "").lower()
    if simple:
        if simple == "yesterday":
            return _day(anchor - timedelta(days=1))
        if simple == "tomorrow":
            return _day(anchor + timedelta(days=1))
        if simple == "last night":
            return f"the night of {_day(anchor - timedelta(days=1))}"
        return anchor_day
    relation = (match.group("rel") or "").lower()
    what = (match.group("what") or "").lower()
    if relation:
        past = relation in {"last", "past", "this past"}
        future = relation in {"next", "coming"}
        if what == "week":
            return f"the week before {anchor_day}" if past else (
                f"the week after {anchor_day}" if future else f"the week of {anchor_day}"
            )
        if what == "weekend":
            return f"the weekend before {anchor_day}" if past else (
                f"the weekend after {anchor_day}" if future else f"the weekend of {anchor_day}"
            )
        if what == "month":
            shift = -1 if past else 1 if future else 0
            return _month(_shift_months(anchor, shift))
        if what == "year":
            return str(anchor.year - 1 if past else anchor.year + 1 if future else anchor.year)
        if what in _WEEKDAYS:
            name = what.capitalize()
            target = _WEEKDAYS.index(what)
            if past:
                back = (anchor.weekday() - target - 1) % 7 + 1
                return f"the {name} before {anchor_day}, {_day(anchor - timedelta(days=back))}"
            if future:
                ahead = (target - anchor.weekday() - 1) % 7 + 1
                return f"the {name} after {anchor_day}, {_day(anchor + timedelta(days=ahead))}"
            return f"the {name} of the week of {anchor_day}"
        start, end = _SEASONS[what]
        if past:  # the most recent season that had ended by the anchor
            year = anchor.year if anchor.month > end and end >= start else anchor.year - 1
        elif future:  # the next season that has not started yet
            year = anchor.year if anchor.month < start else anchor.year + 1
        else:
            year = anchor.year
        return f"{'fall' if what == 'autumn' else what} {year}"
    unit = expression.lower().split()[-2].rstrip("s")
    if match.group("vague"):
        plural = unit + "s"
        return f"a few {plural} before {anchor_day}"
    count = match.group("count").lower()
    number = int(count) if count.isdigit() else _NUMBERS[count]
    if unit == "day":
        return _day(anchor - timedelta(days=number))
    if unit == "week":
        return f"the week of {_day(anchor - timedelta(weeks=number))}"
    if unit == "month":
        return _month(_shift_months(anchor, -number))
    return str(anchor.year - number)


def annotate(text: str, anchor: datetime) -> str:
    """Append ``[= resolution]`` after every relative time expression; text is otherwise exact."""

    def replace(match: re.Match[str]) -> str:
        resolved = resolve(match.group(0), anchor)
        return match.group(0) if resolved is None else f"{match.group(0)} [= {resolved}]"

    return _PATTERN.sub(replace, text)
