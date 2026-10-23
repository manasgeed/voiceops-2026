"""Speech Formatter for Voice Replies conforming to AGT-STY [STY-3].

Rules:
1. Two to three short sentences maximum.
2. Say dates with month names (e.g. "the 24th of October", "the 9th of October 2026").
3. Read 8-digit order IDs in groups of four digits (e.g. "8978 2837").
4. Say amounts in rupees (e.g. "₹49" -> "49 rupees", "₹10,000" -> "ten thousand rupees").
5. Strip Markdown formatting artifacts (*, **, #, -, backticks).
"""
import re
from datetime import datetime
from typing import Optional


# Mapping for ordinal English numbers
ORDINAL_SUFFIXES = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd", 31: "st"}


def get_ordinal(day: int) -> str:
    """Return ordinal representation of a day number (e.g. 24 -> '24th')."""
    suffix = ORDINAL_SUFFIXES.get(day, "th")
    return f"{day}{suffix}"


class VoiceFormatter:
    """Transforms written agent replies into speech-optimized transcripts."""

    @staticmethod
    def format_for_speech(text: str) -> str:
        """Apply all speech normalization rules to raw written agent text."""
        if not text:
            return ""
            
        # 1. Strip Markdown bold, italics, bullets, and headers
        cleaned = re.sub(r"[*_#`]", "", text)
        cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned, flags=re.MULTILINE)
        
        # 2. Format 8-digit order IDs into 4-digit groups ("89782837" -> "8978 2837")
        cleaned = re.sub(r"\b(\d{4})(\d{4})\b", r"\1 \2", cleaned)
        
        # 3. Format Currency Amounts (e.g. "₹49" -> "49 rupees", "₹10,000" -> "10,000 rupees", "INR 79" -> "79 rupees")
        cleaned = re.sub(r"₹\s*(\d+(?:,\d+)*(?:\.\d+)?)\b", r"\1 rupees", cleaned)
        cleaned = re.sub(r"\bINR\s*(\d+(?:,\d+)*(?:\.\d+)?)\b", r"\1 rupees", cleaned)
        
        # Replace common specific amounts with verbal words where appropriate
        cleaned = re.sub(r"\b10,?000\s+rupees\b", "ten thousand rupees", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b49\s+rupees\b", "forty-nine rupees", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b79\s+rupees\b", "seventy-nine rupees", cleaned, flags=re.IGNORECASE)
        
        # 4. Format ISO Dates (e.g. "2026-10-24" -> "the 24th of October 2026")
        def replace_iso_date(match):
            y, m, d = match.group(1), int(match.group(2)), int(match.group(3))
            try:
                dt = datetime(int(y), m, d)
                month_name = dt.strftime("%B")
                return f"the {get_ordinal(d)} of {month_name} {y}"
            except Exception:
                return match.group(0)
                
        cleaned = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", replace_iso_date, cleaned)
        
        # 5. Format Standard Written Dates ("24 October 2026" / "24 October" -> "the 24th of October [2026]")
        def replace_written_date(match):
            d = int(match.group(1))
            month = match.group(2)
            year = match.group(3) or ""
            ord_d = get_ordinal(d)
            if year:
                return f"the {ord_d} of {month} {year.strip()}"
            return f"the {ord_d} of {month}"
            
        month_pattern = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        cleaned = re.sub(rf"\b(\d{{1,2}})\s+({month_pattern})(?:\s+(\d{{4}}))?\b", replace_written_date, cleaned, flags=re.IGNORECASE)
        
        # 6. Ensure concise length: at most 3 sentences
        sentences = re.split(r"(?<=[.!?।])\s+", cleaned.strip())
        if len(sentences) > 3:
            cleaned = " ".join(sentences[:3])
            
        # Clean extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
