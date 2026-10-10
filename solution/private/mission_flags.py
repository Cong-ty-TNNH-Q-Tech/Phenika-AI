"""Vietnamese flag inference using general negation/material cues.

Rules are developed on train text, measured on validation, frozen for test.
"""
import re
from nlp_fast import normalize


def urgent(text, fallback):
    t=normalize(text)
    if re.search(r'\b(khong (?:can )?(?:gap|voi|nhanh)|tu tu|mai (?:giao|dua)|'
                 r'chieu nay|thu tha|khong khan cap|khong can ngay)\b',t):
        return False
    if re.search(r'\b(gap|khan cap|hoa toc|nhanh|ngay trong|can ngay|som nhat)\b',t):
        return True
    return fallback


def fragile(text,fallback):
    t=normalize(text)
    if re.search(r'\b(khong (?:de )?vo|khong vo duoc|chac chan|khong can (?:qua )?nhe tay)\b',t):
        return False
    if re.search(r'\b(gom|thuy tinh|do su|de vo|mong manh|tranh va cham|nhe tay)\b',t):
        return True
    return fallback


def apply(text,mission):
    result=dict(mission)
    result['urgent']=urgent(text,mission['urgent'])
    result['fragile']=fragile(text,mission['fragile'])
    return result
