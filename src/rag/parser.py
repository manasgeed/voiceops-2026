"""Markdown Knowledge Base and Product Catalog Ingestion Parser.

Converts 21 kb/*.md documents and data/products.csv into structured,
precedence-aware KnowledgeChunk objects preserving [SECTION-ID] headers,
document status, effective dates, and authority tiers.
"""
import re
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.config import KB_DIR, DATA_DIR

class KnowledgeChunk(BaseModel):
    """Structured representation of a document section or catalog row."""
    chunk_id: str = Field(description="Unique stable identifier: <doc_id>_<section_id>")
    doc_id: str = Field(description="Document ID (e.g. POL-RET, AGT-IDV, products.csv)")
    doc_title: str = Field(description="Document title")
    doc_type: str = Field(description="policy, agent_guide, help, faq, marketing, catalog, meta")
    status: str = Field(default="current", description="current, superseded")
    effective_from: Optional[str] = Field(default=None, description="Effective date YYYY-MM-DD")
    section_id: str = Field(description="Clean section ID (e.g. RET-2, IDV-1, products.csv:ELE-0012)")
    section_title: str = Field(default="", description="Human readable title of the section")
    content: str = Field(description="Clean textual content of the chunk")
    authority_tier: int = Field(
        description="Hierarchy tier: 1=Current Policy, 2=Agent Guide, 3=Help, 4=FAQ, 5=Marketing, 6=Superseded"
    )
    language: str = Field(default="en", description="en, hi, hi-en")
    source_file: str = Field(description="Relative path of source file")

def compute_authority_tier(doc_type: str, status: str) -> int:
    """Compute authority tier per KB-00.1 precedence rules:
    
    1. Current policy / meta
    2. Agent guides
    3. Help articles
    4. FAQs
    5. Marketing pages
    6. Superseded documents
    """
    if status == "superseded":
        return 6
    dt = (doc_type or "").lower().strip()
    if dt in ("policy", "meta"):
        return 1
    elif dt == "agent_guide":
        return 2
    elif dt == "help":
        return 3
    elif dt == "faq":
        return 4
    elif dt == "marketing":
        return 5
    elif dt == "catalog":
        return 2  # Catalog facts are authoritative product specifications
    return 4

def parse_frontmatter(raw_text: str) -> tuple[Dict[str, str], str]:
    """Parse YAML frontmatter enclosed in '---' lines."""
    metadata = {}
    body = raw_text
    if raw_text.startswith("---"):
        parts = raw_text.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2]
            for line in fm_text.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip().lower()] = val.strip()
    return metadata, body.strip()

def parse_kb_markdown_file(file_path: Path) -> List[KnowledgeChunk]:
    """Parse a single kb/*.md file into section-level chunks."""
    content = file_path.read_text(encoding="utf-8")
    metadata, body = parse_frontmatter(content)
    
    doc_id = metadata.get("doc_id", file_path.stem)
    doc_title = metadata.get("title", file_path.stem)
    doc_type = metadata.get("doc_type", "help")
    status = metadata.get("status", "current")
    effective_from = metadata.get("effective_from")
    language = metadata.get("language", "en")
    authority_tier = compute_authority_tier(doc_type, status)
    
    # Split by section headers: ## [SECTION-ID] Section Title
    section_pattern = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*)$", re.MULTILINE)
    matches = list(section_pattern.finditer(body))
    
    chunks: List[KnowledgeChunk] = []
    
    # If there's preamble text before the first section
    if matches:
        first_start = matches[0].start()
        preamble = body[:first_start].strip()
        if preamble and len(preamble) > 30:
            intro_id = f"{doc_id}.0"
            chunks.append(KnowledgeChunk(
                chunk_id=f"{doc_id}_{intro_id}",
                doc_id=doc_id,
                doc_title=doc_title,
                doc_type=doc_type,
                status=status,
                effective_from=effective_from,
                section_id=intro_id,
                section_title=f"{doc_title} - Overview",
                content=f"{doc_title}\n\n{preamble}",
                authority_tier=authority_tier,
                language=language,
                source_file=f"kb/{file_path.name}"
            ))
            
        for i, match in enumerate(matches):
            section_id = match.group(1).strip()
            section_title = match.group(2).strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            section_content = body[start_pos:end_pos].strip()
            
            full_text = f"Document: {doc_title} ({doc_id})\nSection: [{section_id}] {section_title}\n\n{section_content}"
            
            chunks.append(KnowledgeChunk(
                chunk_id=f"{doc_id}_{section_id}",
                doc_id=doc_id,
                doc_title=doc_title,
                doc_type=doc_type,
                status=status,
                effective_from=effective_from,
                section_id=section_id,
                section_title=section_title,
                content=full_text,
                authority_tier=authority_tier,
                language=language,
                source_file=f"kb/{file_path.name}"
            ))
    else:
        # Fallback if no explicit section ID headings exist
        chunks.append(KnowledgeChunk(
            chunk_id=f"{doc_id}_FULL",
            doc_id=doc_id,
            doc_title=doc_title,
            doc_type=doc_type,
            status=status,
            effective_from=effective_from,
            section_id=doc_id,
            section_title=doc_title,
            content=f"{doc_title}\n\n{body}",
            authority_tier=authority_tier,
            language=language,
            source_file=f"kb/{file_path.name}"
        ))
        
    return chunks

def parse_product_catalog_csv(file_path: Optional[Path] = None) -> List[KnowledgeChunk]:
    """Parse data/products.csv into searchable catalog chunks formatted as products.csv:SKU."""
    path = file_path or (DATA_DIR / "products.csv")
    if not path.exists():
        return []
        
    chunks: List[KnowledgeChunk] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sku = row.get("sku", "").strip()
            if not sku:
                continue
            title = row.get("title", "")
            brand = row.get("brand", "")
            category = row.get("return_category", "")
            price = row.get("price_inr", "")
            fulfilled_by = row.get("fulfilled_by", "")
            assured = row.get("assured", "")
            warranty = row.get("warranty_months", "")
            desc = row.get("description", "")
            
            content = (
                f"Product SKU: {sku}\n"
                f"Title: {title}\n"
                f"Brand: {brand}\n"
                f"Return Category: {category}\n"
                f"Price: INR {price}\n"
                f"Fulfilled by: {fulfilled_by}\n"
                f"Assured Badge: {assured}\n"
                f"Warranty: {warranty} months\n"
                f"Description: {desc}"
            )
            
            section_id = f"products.csv:{sku}"
            chunks.append(KnowledgeChunk(
                chunk_id=f"CAT_{sku}",
                doc_id="products.csv",
                doc_title=f"Catalog: {title}",
                doc_type="catalog",
                status="current",
                effective_from="2026-08-01",
                section_id=section_id,
                section_title=f"{sku} - {title}",
                content=content,
                authority_tier=2,
                language="en",
                source_file="data/products.csv"
            ))
            
    return chunks

def load_all_chunks(kb_dir: Optional[Path] = None, data_dir: Optional[Path] = None) -> List[KnowledgeChunk]:
    """Load and index all 21 KB documents and catalog items."""
    k_dir = kb_dir or KB_DIR
    d_dir = data_dir or DATA_DIR
    
    chunks: List[KnowledgeChunk] = []
    kb_files = sorted(list(k_dir.glob("*.md")))
    for kf in kb_files:
        chunks.extend(parse_kb_markdown_file(kf))
        
    # Also load product catalog chunks
    chunks.extend(parse_product_catalog_csv(d_dir / "products.csv"))
    
    return chunks
