import re


HUMAN_TERMS = [
    r"\bmen\b",
    r"\bmen's\b",
    r"\bman\b",
    r"\bwomen\b",
    r"\bwomen's\b",
    r"\bwoman\b",
    r"\bboys\b",
    r"\bboy's\b",
    r"\bgirls\b",
    r"\bgirl's\b",
    r"\bmale\b",
    r"\bfemale\b",
    r"\bslim fit\b",
    r"\bregular fit\b",
]


def remove_human_terms(text: str) -> str:
    result = text

    for pattern in HUMAN_TERMS:
        result = re.sub(
            pattern,
            "",
            result,
            flags=re.IGNORECASE,
        )

    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"\s+,", ",", result)

    return result.strip(" ,")