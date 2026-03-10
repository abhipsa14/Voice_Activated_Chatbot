"""
Speech Post-Processor for UIT Chatbot
--------------------------------------
Cleans and corrects speech-to-text output to improve accuracy,
especially for Indian English accents and domain-specific terms.

Features:
  - Domain-specific word correction (UIT, AKTU, HOD, etc.)
  - Common Indian English misheard word fixes
  - Confidence estimation from speech patterns
  - Filler word removal
  - Number/abbreviation normalisation
"""

import re
from difflib import SequenceMatcher


# ── Domain-specific corrections ────────────────────────────────────────
# Maps common mis-transcriptions → correct term
DOMAIN_CORRECTIONS = {
    # Institute-specific
    "you it": "UIT",
    "you i t": "UIT",
    "u i t": "UIT",
    "u it": "UIT",
    "uite": "UIT",
    "unite": "UIT",
    "united": "United",
    "uit": "UIT",
    "pray graj": "Prayagraj",
    "pray graj": "Prayagraj",
    "prayer graj": "Prayagraj",
    "praire graj": "Prayagraj",
    "priya graj": "Prayagraj",
    "prayag raj": "Prayagraj",
    "prayagraj": "Prayagraj",
    "allahabad": "Prayagraj",

    # University names
    "a k t u": "AKTU",
    "ak tu": "AKTU",
    "aktu": "AKTU",
    "act you": "AKTU",
    "aicte": "AICTE",
    "a i c t e": "AICTE",
    "a ict": "AICTE",
    "ai city": "AICTE",

    # Roles
    "h o d": "HOD",
    "hod": "HOD",
    "head of department": "HOD",
    "principal": "principal",
    "dean": "dean",
    "crc": "CRC",
    "c r c": "CRC",

    # Technical terms
    "e r p": "ERP",
    "erp": "ERP",
    "cc tv": "CCTV",
    "cctv": "CCTV",
    "wi fi": "Wi-Fi",
    "wifi": "Wi-Fi",
    "fdp": "FDP",
    "f d p": "FDP",
    "t f idf": "TF-IDF",

    # Common Indian English misheard words
    "hostal": "hostel",
    "collage": "college",
    "libary": "library",
    "librry": "library",
    "addmission": "admission",
    "plcement": "placement",
    "ragin": "ragging",
    "raggin": "ragging",
    "semister": "semester",
    "exams": "examination",
    "counceling": "counseling",
    "counselling": "counseling",
}

# Filler words to strip from transcriptions
FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "hmm", "hm", "er", "ah", "like",
    "basically", "actually", "so", "well", "you know", "i mean",
    "okay so", "right so",
}

# Person names that are commonly misheard
PERSON_CORRECTIONS = {
    "sanjay srivastav": "Sanjay Srivastava",
    "sanjay shri vastav": "Sanjay Srivastava",
    "abhishek malvia": "Abishek Malviya",
    "abhishek malviya": "Abishek Malviya",
    "abishek malvia": "Abishek Malviya",
    "shruti sharma": "Shruti Sharma",
    "shruthi sharma": "Shruti Sharma",
    "amit kumar tiwari": "Amit Kumar Tiwari",
    "amit tiwari": "Amit Kumar Tiwari",
    "amitabh srivastav": "Amitabh Srivastava",
    "amitabh shrivastav": "Amitabh Srivastava",
    "ankit kumar gupta": "Ankit Kumar Gupta",
    "ankit gupta": "Ankit Kumar Gupta",
}


def _similarity(a: str, b: str) -> float:
    """Return 0.0-1.0 similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def correct_domain_terms(text: str) -> str:
    """Apply domain-specific corrections to transcribed text."""
    result = text

    # Apply person name corrections (longer phrases first)
    for wrong, right in sorted(PERSON_CORRECTIONS.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        result = pattern.sub(right, result)

    # Apply domain corrections (longer phrases first to avoid partial matches)
    for wrong, right in sorted(DOMAIN_CORRECTIONS.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE)
        result = pattern.sub(right, result)

    return result


def remove_fillers(text: str) -> str:
    """Remove filler words and speech disfluencies."""
    # Remove multi-word fillers first
    for filler in sorted(FILLER_WORDS, key=len, reverse=True):
        if " " in filler:
            pattern = re.compile(r"\b" + re.escape(filler) + r"\b", re.IGNORECASE)
            text = pattern.sub("", text)

    words = text.split()
    cleaned = []
    for w in words:
        if w.lower().strip(".,!?") not in FILLER_WORDS:
            cleaned.append(w)

    return " ".join(cleaned)


def normalise_text(text: str) -> str:
    """Clean up whitespace, punctuation, and capitalisation."""
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Remove leading/trailing junk
    text = text.strip("., \t\n")
    # Capitalise first letter
    if text:
        text = text[0].upper() + text[1:]
    return text


def estimate_confidence(text: str, raw_text: str = None) -> float:
    """
    Estimate confidence in the transcription quality.

    Returns 0.0-1.0 where:
      - 1.0 = high confidence (clear, meaningful text)
      - 0.0 = low confidence (garbage/filler only)

    Heuristics used:
      - Length of meaningful content
      - Ratio of real words vs fillers
      - Presence of recognisable domain terms
      - Reasonable sentence structure
    """
    if not text or not text.strip():
        return 0.0

    words = text.lower().split()
    total_words = len(words)

    if total_words == 0:
        return 0.0

    # Count filler vs real words
    filler_count = sum(1 for w in words if w.strip(".,!?") in FILLER_WORDS)
    real_words = total_words - filler_count
    filler_ratio = filler_count / total_words

    # Base confidence from word count (very short = low confidence)
    if real_words <= 1:
        length_score = 0.3
    elif real_words <= 3:
        length_score = 0.6
    else:
        length_score = 0.9

    # Domain term presence boosts confidence
    domain_score = 0.0
    all_corrections = set(
        w.lower() for w in DOMAIN_CORRECTIONS.values()
    ) | set(
        w.lower() for w in PERSON_CORRECTIONS.values()
    )
    text_lower = text.lower()
    for term in all_corrections:
        if term.lower() in text_lower:
            domain_score = 0.2
            break

    # Filler penalty
    filler_penalty = filler_ratio * 0.4

    # Very short single-word transcriptions are often noise
    noise_penalty = 0.0
    if total_words == 1 and words[0] in {"the", "a", "an", "i", "is", "it"}:
        noise_penalty = 0.5

    confidence = min(1.0, max(0.0,
        length_score + domain_score - filler_penalty - noise_penalty
    ))

    return round(confidence, 2)


def process_speech(raw_text: str) -> tuple[str, float]:
    """
    Full speech post-processing pipeline.

    Args:
        raw_text: Raw transcription from STT engine.

    Returns:
        (processed_text, confidence_score)
    """
    if not raw_text or not raw_text.strip():
        return "", 0.0

    # Step 1: Remove fillers
    cleaned = remove_fillers(raw_text)

    # Step 2: Apply domain corrections
    corrected = correct_domain_terms(cleaned)

    # Step 3: Normalise
    final = normalise_text(corrected)

    # Step 4: Estimate confidence
    confidence = estimate_confidence(final, raw_text)

    return final, confidence


if __name__ == "__main__":
    # Test examples
    test_cases = [
        "um tell me about you it pray graj",
        "what is the hod of computer science",
        "uh does the collage have wifi",
        "who is the principal of uite",
        "basically like um is there hostal facility",
        "a k t u affiliation",
        "shruti sharma placement",
        "the",
        "",
        "tell me about admission process at united institute",
    ]

    print("=== Speech Post-Processor Test ===\n")
    for raw in test_cases:
        processed, confidence = process_speech(raw)
        print(f"  Raw:        '{raw}'")
        print(f"  Processed:  '{processed}'")
        print(f"  Confidence: {confidence:.2f}")
        print()
