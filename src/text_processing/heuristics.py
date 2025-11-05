"""Heuristic text processing functions for cleaning Wikipedia markup."""

import re
from typing import List

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from ..config import MAX_SENT_CHARS

# Compiled regex patterns for performance
_url_re = re.compile(r"https?://\S+|www\.\S+", re.I)
_ref_re = re.compile(r"\[\d+\]")
_cite_re = re.compile(r"\{\{.*?\}\}")
_file_re = re.compile(r"\[\[(File|Image):.*?\]\]", re.I)
_link_re = re.compile(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]")
_heading_re = re.compile(r"^\s*={2,}.*?={2,}\s*$")
_list_re = re.compile(r"^\s*([*#:-]|\d+\.)\s+")
_table_re = re.compile(r"^\s*\{\|.*|\|\}|\|\-|\!|\|\s", re.I)
_cleanup_spaces = re.compile(r"\s+")


def clean_wiki_markup(text: str) -> str:
    """Clean Wikipedia markup from text.
    
    Args:
        text: Raw Wikipedia text with markup
        
    Returns:
        Cleaned text with markup removed
    """
    text = _url_re.sub("", text)
    text = _cite_re.sub(" ", text)
    text = _file_re.sub(" ", text)
    text = _link_re.sub(r"\1", text)
    text = _ref_re.sub("", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = _cleanup_spaces.sub(" ", text).strip()
    return text


def split_into_sentences(text: str) -> List[str]:
    """Use NLTK's robust sentence tokenizer.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentences
    """
    from nltk.tokenize import sent_tokenize
    text = text.strip()
    if not text:
        return []
    return sent_tokenize(text)


def soft_clean(s: str) -> str:
    """Clean spacing but don't truncate - semantic parsers need complete sentences.
    
    Args:
        s: Input sentence
        
    Returns:
        Cleaned sentence with normalized spacing
    """
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s([,;:])", r"\1", s)
    return s


# Export regex patterns for use in validation
def get_heading_pattern():
    """Get the compiled heading regex pattern."""
    return _heading_re


def get_list_pattern():
    """Get the compiled list regex pattern."""
    return _list_re


def get_table_pattern():
    """Get the compiled table regex pattern."""
    return _table_re
