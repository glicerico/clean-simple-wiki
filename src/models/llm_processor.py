"""LLM processing for sentence refinement and quality assessment."""

import os
import json
import time
from typing import List, Dict, Any, Tuple

from ..config import LLM_TIMEOUT

# Optional dependency handling
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

try:
    from langsmith import traceable
    from langsmith.wrappers import wrap_openai
    _LANGSMITH_AVAILABLE = True
except Exception:
    _LANGSMITH_AVAILABLE = False
    # Create dummy decorator if langsmith not available
    def traceable(func):
        return func
    wrap_openai = None


# LLM prompts
LLM_SYSTEM_PROMPT = (
    "You are a language-to-knowledge conversion model.\n\n"
    "Your job is to transform text into clear, factual, standalone sentences that can be directly used for knowledge extraction and database ingestion.\n\n"
    "Detailed Guidelines\n\n"
    "    Input type:\n\n"
    "    The input text comes from Simple English Wikipedia and may include sentences, fragments, or explanatory phrasing.\n\n"
    "    Objective:\n\n"
    "    Convert the text into a list of fully independent factual sentences.\n\n"
    "    Each sentence must:\n\n"
    "        Convey one complete, atomic fact.\n\n"
    "        Be understandable without any previous context.\n\n"
    "        Include explicit subjects and objects (no pronouns or implicit references).\n\n"
    "        Avoid transitional or connective words (e.g., \"however,\" \"therefore,\" \"this,\" \"these,\" \"such,\" \"as mentioned,\" etc.).\n\n"
    "        Preserve the factual accuracy of the original content.\n\n"
    "    Filtering Rule:\n\n"
    "        Remove any line or phrase that does not express a factual statement suitable for knowledge extraction.\n\n"
    "        Examples to remove include: definitions that merely restate words (\"This means that…\"), stylistic commentary, or redundant explanations.\n\n"
    "    Language style:\n\n"
    "        Keep the tone factual and formal.\n\n"
    "        Do not simplify scientific or technical terms further — use precise terminology.\n\n"
    "        Do not add new information or interpretations.\n\n"
    "    Output format:\n\n"
    "        Output one factual, standalone sentence per line.\n\n"
    "        Do not number or bullet the lines.\n\n"
    "        Maintain consistent capitalization and punctuation."
)

LLM_USER_TEMPLATE = """TASK:
Return a JSON array with objects:
{{"sentences": ["sentence1", "sentence2", ...], "confidence": <0..1>}}

EXAMPLE INPUT:

Helium is a chemical element. It usually has two neutrons, but some helium atoms have only one. These atoms are still helium because the number of protons defines the element. However, they are not normal helium either.

EXAMPLE OUTPUT:

{{"sentences": ["Helium is a chemical element.", "A typical helium atom contains two neutrons.", "Some helium atoms contain one neutron.", "An atom that contains two protons is defined as helium."], "confidence": 0.9}}

INPUT TEXT:
{payload}

Transform the above text into clear, factual, standalone sentences following the guidelines.
"""


def format_llm_payload(chunks: List[str]) -> str:
    """Format payload as a block of sentences for context-aware processing.
    
    Args:
        chunks: List of sentence chunks
        
    Returns:
        Formatted payload string
    """
    return "\n\n".join(chunks)


def build_llm_messages(chunks: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (user_payload, messages) for the chat completion call.
    
    Args:
        chunks: List of sentence chunks
        
    Returns:
        Tuple of (user_payload, messages_list)
    """
    content = LLM_USER_TEMPLATE.format(payload=format_llm_payload(chunks))
    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return content, messages


def parse_llm_response_text(text: str) -> Dict[str, Any]:
    """Parse the JSON response returned by the LLM into structured record.
    
    Args:
        text: Raw response text from LLM
        
    Returns:
        Parsed response with sentences and confidence
        
    Raises:
        ValueError: If response cannot be parsed or is invalid
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to decode LLM response as JSON: {exc}\n{text[:2000]}") from exc

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object from LLM response, got {type(obj)}")
    
    if "sentences" not in obj:
        raise ValueError(f"LLM response missing 'sentences' field: {obj}")
    
    sentences = obj.get("sentences", [])
    if not isinstance(sentences, list):
        raise ValueError(f"Expected 'sentences' to be a list, got {type(sentences)}")
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        if isinstance(sentence, str) and sentence.strip():
            cleaned_sentences.append(sentence.strip())
    
    return {
        "sentences": cleaned_sentences,
        "confidence": float(obj.get("confidence", 0.0)),
    }


def create_openai_client(enable_tracing: bool = False):
    """Create an OpenAI client with optional tracing.
    
    Args:
        enable_tracing: Whether to enable LangSmith tracing
        
    Returns:
        Configured OpenAI client
        
    Raises:
        RuntimeError: If OPENAI_API_KEY is not set
    """
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed.")
        
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)
    if enable_tracing and _LANGSMITH_AVAILABLE and wrap_openai:
        client = wrap_openai(client)
        print("[llm] LangSmith tracing enabled")
    return client


class LLMProcessor:
    """Handles LLM processing for sentence refinement."""
    
    def __init__(self, model: str = "gpt-4o-mini", enable_tracing: bool = False):
        """Initialize the LLM processor.
        
        Args:
            model: OpenAI model name
            enable_tracing: Whether to enable LangSmith tracing
        """
        self.model = model
        self.client = create_openai_client(enable_tracing)
    
    @traceable(name="llm_batch_review")
    def process_batch(self, chunks: List[str]) -> Dict[str, Any]:
        """Process a batch of sentences through the LLM.
        
        Args:
            chunks: List of sentences to process
            
        Returns:
            Processed result with sentences and confidence
        """
        content, messages = build_llm_messages(chunks)
        
        # Debug: show payload size
        payload_size = len(content)
        num_lines = content.count('\n')
        print(f"[llm]     Payload: {payload_size} chars, {num_lines} lines, {len(chunks)} input sentences")
        
        # Debug: show first few lines of payload
        first_lines = '\n'.join(content.split('\n')[:3])
        print(f"[llm]     First lines:\n{first_lines}")
        
        # Debug: show input sentences
        if len(chunks) <= 5:
            print(f"[llm]     All input sentences:")
            for i, sentence in enumerate(chunks):
                print(f"[llm]       {i+1}. {sentence[:100]}...")
        else:
            print(f"[llm]     First 3 input sentences:")
            for i, sentence in enumerate(chunks[:3]):
                print(f"[llm]       {i+1}. {sentence[:100]}...")
        
        print(f"[llm]     Sending to {self.model}... ")
        api_start = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=messages,
            response_format={"type":"json_object"},
            timeout=LLM_TIMEOUT
        )
        api_elapsed = time.time() - api_start
        print(f"[llm]     API returned in {api_elapsed:.1f}s")
        
        text = resp.choices[0].message.content
        print(f"[llm]     Response length: {len(text)} chars")
        print(f"[llm]     Parsing JSON...")
        
        parsed = parse_llm_response_text(text)
        print(f"[llm]     Parsed {len(parsed.get('sentences', []))} output sentences from JSON")
        return parsed
