"""Sentence validation functions for filtering quality content."""

import re
from typing import Tuple, Optional

from ..config import MAX_SENT_CHARS
from .heuristics import get_heading_pattern, get_list_pattern, get_table_pattern


def looks_like_sentence_with_reason(s: str) -> Tuple[bool, Optional[str]]:
    """Check if a string looks like a valid sentence and return the reason if not.
    
    Args:
        s: Input string to validate
        
    Returns:
        Tuple of (is_valid, rejection_reason)
        rejection_reason is None if valid, otherwise contains the reason for rejection
    """
    if len(s) < 15:
        return False, "length"
    if len(s) > MAX_SENT_CHARS:
        return False, "length"
    if get_heading_pattern().match(s):
        return False, "heading"
    if get_list_pattern().match(s):
        return False, "list"
    if get_table_pattern().match(s):
        return False, "table"
    if not re.search(r"[A-Za-z]", s):
        return False, "no_letters"
    if len(s.split()) < 3:
        return False, "too_few_words"
    # Require sentence-ending punctuation for short sentences
    if len(s.split()) < 8 and not s.rstrip().endswith(('.', '!', '?')):
        return False, "not_sentence_like"
    return True, None
