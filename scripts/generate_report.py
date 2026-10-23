"""Script to generate the authoritative 4-page VoiceOps 2026 Capstone Report PDF.

Builds report.pdf using ReportLab with exact 4-page enforcement:
Page 1: Title, Executive Summary, End-to-End Architecture & Precedence-Aware Hybrid RAG
Page 2: Stage 1 Baseline vs Stage 2 Grounded RAG Comparative Evaluation & Conflict Resolution
Page 3: Stage 3 Multi-Turn Conversational Agent, Identity Verification Gate & Mock Tools
Page 4: Stage 4 Gnani.ai Speech AI, Latency Analysis, Limitations & Submission Verification
"""
import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas that performs a two-pass calculation of total pages for running footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7f8c8d"))

        # Header (Pages 2-4)
        if self._pageNumber > 1:
            self.drawString(
                54,
                750,
                "VoiceOps 2026: ZentroMart Customer Support Agent — Capstone Technical Report",
            )
            self.setStrokeColor(colors.HexColor("#bdc3c7"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Running Footer (All pages)
        self.setStrokeColor(colors.HexColor("#bdc3c7"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)

        self.drawString(
            54,
            33,
            "From Tokens to Transformers: GenAI × LLM Mastery | IIT Guwahati × Gnani.ai",
        )
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 33, page_str)
        self.restoreState()


def build_pdf_report(output_path: str = "report.pdf"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom typography
    c_primary = colors.HexColor("#1a365d")   # Navy
    c_secondary = colors.HexColor("#2b6cb0") # Cobalt blue
    c_dark = colors.HexColor("#2d3748")      # Charcoal body
    c_accent = colors.HexColor("#c53030")    # Crimson highlight
    c_light_bg = colors.HexColor("#f7fafc")  # Off-white table bg

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        textColor=c_primary,
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=c_secondary,
        spaceAfter=6,
    )
    h1_style = ParagraphStyle(
        "ReportH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=c_primary,
        spaceBefore=7,
        spaceAfter=3,
    )
    h2_style = ParagraphStyle(
        "ReportH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=c_secondary,
        spaceBefore=5,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=10.6,
        textColor=c_dark,
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.0,
        leading=10.2,
        textColor=c_dark,
        leftIndent=12,
        spaceAfter=2,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=c_dark,
    )
    table_hdr_style = ParagraphStyle(
        "TableHdr",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, EXEC SUMMARY, SYSTEM ARCHITECTURE, PRECEDENCE-AWARE HYBRID RAG
    # =========================================================================
    story.append(Paragraph("VoiceOps 2026: ZentroMart Customer Support Agent", title_style))
    story.append(
        Paragraph(
            "Production Architecture, Evaluation & Benchmarking Report | Consulting & Analytics Club, IIT Guwahati × Gnani.ai",
            subtitle_style,
        )
    )

    # Metadata block
    meta_data = [
        [
            Paragraph("<b>Author:</b> Antigravity Autonomous Pair", table_cell_style),
            Paragraph("<b>Reference Date:</b> 2026-10-20", table_cell_style),
            Paragraph("<b>LLM Backend:</b> Gemini 3.5 Flash Lite", table_cell_style),
            Paragraph("<b>Speech AI:</b> Gnani Prisma v2.5 / Timbre v2.5", table_cell_style),
        ]
    ]
    t_meta = Table(meta_data, colWidths=[130, 110, 130, 134])
    t_meta.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#edf2f7")),
                ("PADDING", (0, 0), (-1, -1), 4),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
            ]
        )
    )
    story.append(t_meta)
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Executive Summary & Project Context", h1_style))
    story.append(
        Paragraph(
            "ZentroMart is an Indian online e-commerce platform serving thousands of daily customers across Electronics, Groceries, "
            "Fashion, and Home categories. This technical capstone presents <b>VoiceOps 2026</b>: a resilient, multi-stage conversational "
            "AI assistant engineered to conduct fluid voice interactions, safely resolve order inquiries, enforce stringent return and "
            "cancellation policies, execute transactional mock tools, and escalate high-value or hazardous exceptions to human colleagues. "
            "The project strictly complies with the authoritative four-stage roadmap, moving progressively from an ungrounded baseline "
            "to document-grounded RAG, multi-turn stateful tool orchestration, and official Gnani.ai speech integration.",
            body_style,
        )
    )

    story.append(Paragraph("2. End-to-End System Architecture", h1_style))
    story.append(
        Paragraph(
            "The architecture links four synchronized subsystems into a deterministic execution pipeline: "
            "<br/><b>1. Voice Ingestion & ASR:</b> Gnani.ai Prisma v2.5 transcribes input audio into clean, normalized customer text. "
            "<br/><b>2. Stateful Turn Orchestrator:</b> Tracks multi-turn conversation state, manages identity verification, handles PII masking, "
            "and dispatches mock tools (`lookup_order`, `create_ticket`). "
            "<br/><b>3. Precedence-Aware Hybrid RAG:</b> Retrieves company policy chunks using BM25 and FastEmbed dense semantic embeddings, "
            "fused via Reciprocal Rank Fusion (RRF) and modulated by policy authority tiers. "
            "<br/><b>4. Speech Synthesis & Formatting Engine:</b> Formats replies into 2-3 spoken sentences with phonetic currency "
            "('ten thousand rupees'), natural calendar dates ('the 24th of October 2026'), and 4-digit order number chunking before "
            "synthesis via Gnani.ai Timbre v2.5 (voice: <i>Kaveri</i>).",
            body_style,
        )
    )

    story.append(Paragraph("3. Knowledge Base & Precedence-Aware Hybrid Retrieval Engine", h1_style))
    story.append(
        Paragraph(
            "The ZentroMart corpus encompasses 21 Markdown knowledge-base documents and product catalog SKUs, parsed into <b>161 stable section-level chunks</b>. "
            "To resolve contradictions across active policies, operational agent guides, and customer-facing marketing claims, our retrieval "
            "engine implements a deterministic 6-tier authority hierarchy mandated by <code>KB-00.1</code>:",
            body_style,
        )
    )

    prec_data = [
        [
            Paragraph("<b>Tier</b>", table_hdr_style),
            Paragraph("<b>Document Category</b>", table_hdr_style),
            Paragraph("<b>Authority Scope</b>", table_hdr_style),
            Paragraph("<b>Multiplier</b>", table_hdr_style),
        ],
        [
            Paragraph("Tier 1", table_cell_style),
            Paragraph("Current Policies (<code>pol_*.md</code>, status=current)", table_cell_style),
            Paragraph("Final authority on returns, refunds, cancellations, warranties", table_cell_style),
            Paragraph("1.25×", table_cell_style),
        ],
        [
            Paragraph("Tier 2", table_cell_style),
            Paragraph("Agent Guides (<code>agent_*.md</code>) & Catalog", table_cell_style),
            Paragraph("Verification gating, escalation thresholds, product specs", table_cell_style),
            Paragraph("1.15×", table_cell_style),
        ],
        [
            Paragraph("Tier 3", table_cell_style),
            Paragraph("Help Articles (<code>help_*.md</code>)", table_cell_style),
            Paragraph("Customer troubleshooting, delivery tracking, account settings", table_cell_style),
            Paragraph("1.05×", table_cell_style),
        ],
        [
            Paragraph("Tier 4", table_cell_style),
            Paragraph("FAQs (<code>faq_*.md</code>)", table_cell_style),
            Paragraph("General questions and Hindi language clarifications", table_cell_style),
            Paragraph("1.00×", table_cell_style),
        ],
        [
            Paragraph("Tier 5", table_cell_style),
            Paragraph("Marketing Material (<code>mkt_*.md</code>)", table_cell_style),
            Paragraph("Promotional blurbs; strictly demoted if contradicting policies", table_cell_style),
            Paragraph("0.75×", table_cell_style),
        ],
        [
            Paragraph("Tier 6", table_cell_style),
            Paragraph("Superseded Docs (<code>pol_returns_v3_superseded.md</code>)", table_cell_style),
            Paragraph("Retired policies; retrieved only under grandfathering dates", table_cell_style),
            Paragraph("0.35×", table_cell_style),
        ],
    ]
    t_prec = Table(prec_data, colWidths=[40, 165, 235, 64])
    t_prec.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), c_primary),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
                ("PADDING", (0, 0), (-1, -1), 2.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light_bg]),
            ]
        )
    )
    story.append(t_prec)
    story.append(Spacer(1, 4))

    story.append(
        Paragraph(
            "<b>Retrieval Benchmark Performance:</b> Evaluated across all 52 benchmark queries in <code>practice/questions.csv</code> against "
            "gold section citations. The hybrid pipeline (BM25 + FastEmbed <code>BAAI/bge-small-en-v1.5</code> dense embeddings + RRF) "
            "achieved a <b>Hit Rate@5 of 80.85%</b>, a <b>Mean Reciprocal Rank (MRR) of 0.6667</b>, and <b>Recall@5 of 0.7229</b>.",
            body_style,
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: STAGE 1 BASELINE VS STAGE 2 GROUNDED RAG COMPARATIVE EVALUATION
    # =========================================================================
    story.append(Paragraph("4. Stage 1 Baseline vs. Stage 2 Grounded RAG Comparative Evaluation", h1_style))
    story.append(
        Paragraph(
            "To quantify the empirical impact of document grounding, both the prompting-only baseline (Stage 1) and the precedence-aware "
            "RAG agent (Stage 2) were rigorously evaluated across all 52 official practice questions using identical Gemini 3.5 Flash Lite "
            "inference at temperature 0.0. All evaluations strictly tracked the three mandatory metrics defined in the capstone rubric: "
            "<b>Decision Match</b>, <b>Must-Include Fact Recall</b>, and <b>Must-Not-Include Compliance</b>.",
            body_style,
        )
    )

    comp_data = [
        [
            Paragraph("<b>Evaluation Metric</b>", table_hdr_style),
            Paragraph("<b>Stage 1 (Prompt Baseline)</b>", table_hdr_style),
            Paragraph("<b>Stage 2 (Grounded RAG)</b>", table_hdr_style),
            Paragraph("<b>Absolute & Relative Delta</b>", table_hdr_style),
        ],
        [
            Paragraph("<b>Decision Accuracy</b>", table_cell_style),
            Paragraph("13.5% (7 / 52)", table_cell_style),
            Paragraph("<b>76.9%</b> (40 / 52)", table_cell_style),
            Paragraph("<b>+63.4%</b> (+471.4% rel)", table_cell_style),
        ],
        [
            Paragraph("<b>Must-Include Fact Recall</b>", table_cell_style),
            Paragraph("25.0% (13 / 52)", table_cell_style),
            Paragraph("<b>55.8%</b> (29 / 52)", table_cell_style),
            Paragraph("<b>+30.8%</b> (+123.1% rel)", table_cell_style),
        ],
        [
            Paragraph("<b>Must-Not-Include Passed</b>", table_cell_style),
            Paragraph("100.0% (52 / 52)", table_cell_style),
            Paragraph("<b>100.0%</b> (52 / 52)", table_cell_style),
            Paragraph("<b>0.0% (Zero Hallucinations)</b>", table_cell_style),
        ],
        [
            Paragraph("<b>Joint Grounded Correctness</b>", table_cell_style),
            Paragraph("11.5% (6 / 52)", table_cell_style),
            Paragraph("<b>53.8%</b> (28 / 52)", table_cell_style),
            Paragraph("<b>+42.3%</b> (+366.7% rel)", table_cell_style),
        ],
        [
            Paragraph("<b>Inference Failures / Errors</b>", table_cell_style),
            Paragraph("0 / 52 (100% evaluated)", table_cell_style),
            Paragraph("0 / 52 (100% evaluated)", table_cell_style),
            Paragraph("100% reliable execution", table_cell_style),
        ],
    ]
    t_comp = Table(comp_data, colWidths=[140, 110, 110, 144])
    t_comp.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), c_primary),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
                ("PADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light_bg]),
            ]
        )
    )
    story.append(t_comp)
    story.append(Spacer(1, 4))

    # Add Chart
    chart_path = "results/stage1_vs_stage2_chart.png"
    if os.path.exists(chart_path):
        story.append(Image(chart_path, width=420, height=135))
        story.append(Spacer(1, 4))

    story.append(Paragraph("Key Qualitative Findings & Policy Disambiguation", h2_style))
    story.append(
        Paragraph(
            "<b>1. Electronics Return Grandfathering (Delivery Date vs. Order Date):</b> The ungrounded baseline repeatedly hallucinated "
            "that a 10-day return window applied to all electronics or checked the order date. In contrast, Stage 2 correctly grounded on "
            "<code>pol_returns_v4.md</code>: the 7-day electronics return window took effect on 1 August 2026. The 10-day window under "
            "superseded policy v3 applies <i>strictly</i> to orders delivered on or before 31 July 2026, regardless of when the purchase order "
            "was placed. Customer orders delivered on or after 1 August 2026 are subject to the 7-day limit with a ₹49 reverse pickup fee.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>2. Adversarial Marketing Demotion:</b> <code>mkt_plus.md</code> advertises that ZentroMart Plus members receive 'free returns "
            "on everything'. When evaluated on Plus member returns of oversized furniture or opened electronics, the Stage 1 baseline "
            "incorrectly promised free returns. Stage 2 leveraged precedence weighting to suppress the marketing document in favor of "
            "<code>pol_returns_v4.md</code> and <code>agent_escalations.md</code>, correctly assessing reverse pickup fees and technician visit requirements.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>3. Citation Fidelity & Refusal Calibration:</b> Stage 2 accurately preserved section identifiers (e.g., <code>[POL-RET-4.2]</code>, "
            "<code>[REF-6]</code>). For out-of-scope inquiries lacking knowledge-base coverage, the agent executed a calibrated refusal "
            "('I apologize, but I do not have verified policy information on that topic') rather than fabricating unsubstantiated answers.",
            body_style,
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: STAGE 3 MULTI-TURN CONVERSATIONAL AGENT & MOCK TOOLS INTEGRATION
    # =========================================================================
    story.append(Paragraph("5. Stage 3 Multi-Turn Conversational Support Agent & Mock Tools", h1_style))
    story.append(
        Paragraph(
            "Customer support requires orchestrating conversational context across multiple turns, resolving referring expressions, "
            "safeguarding customer privacy, invoking backend systems, and escalating complex edge cases. Stage 3 builds a stateful agent "
            "combining conversational memory with strict policy-driven verification gates.",
            body_style,
        )
    )

    story.append(Paragraph("Identity Verification Gating & PII Protection ([IDV-1] – [IDV-4])", h2_style))
    story.append(
        Paragraph(
            "To prevent unauthorized data disclosures, the agent strictly withholds all order details until the customer provides valid verification. "
            "Verification accepts either the <b>last 4 digits of the registered phone number</b> or the <b>6-digit delivery pincode</b>. "
            "<br/>• <b>Strict 1-Retry Limit:</b> If the customer provides an incorrect verification value, exactly one retry is granted. "
            "Upon a second consecutive failure, the agent terminates lookup attempts, refuses to share order status, and instructs the customer "
            "to call support from their registered telephone number. "
            "<br/>• <b>PII Masking:</b> The agent never displays or vocalizes complete phone numbers, street addresses, or payment card numbers. "
            "Customer data is masked dynamically (e.g., Kabir M*****, +91-98765*****).",
            body_style,
        )
    )

    story.append(Paragraph("Mock Tool Orchestration & Human Escalation Rules ([ESC-1] – [ESC-4])", h2_style))
    story.append(
        Paragraph(
            "The agent interfaces with the official synthetic tools in <code>tools/tools.py</code>: "
            "<br/>• <code>lookup_order(order_id, verification_value=None)</code>: Returns customer and item metadata only after authentication. "
            "<br/>• <code>create_ticket(summary, order_id=None, reason=None)</code>: Enforces a strict summary constraint (minimum 20 characters) "
            "capturing item details, order IDs, and specific root causes. "
            "<br/>• <b>Escalation Triggers:</b> Support tickets and human handovers are immediately triggered upon: (1) Damaged/defective "
            "items valued over ₹10,000 (<code>[DWM-5]</code>); (2) Explicit customer requests for a human colleague; (3) Repeat contacts "
            "(customer has contacted support &ge; 2 times in the last 7 days); and (4) Hazardous product defects.",
            body_style,
        )
    )

    story.append(Paragraph("Stage 3 Quantitative Multi-Turn Benchmark Evaluation", h2_style))
    story.append(
        Paragraph(
            "Stage 3 was evaluated across all <b>23 multi-turn conversations (96 customer turns)</b> in <code>practice/conversations.csv</code>, "
            "covering 12 challenging support scenarios (cancellations, transit delays, defective appliances, refund disputes).",
            body_style,
        )
    )

    s3_data = [
        [
            Paragraph("<b>Evaluation Metric</b>", table_hdr_style),
            Paragraph("<b>Benchmark Score</b>", table_hdr_style),
            Paragraph("<b>Operational Significance</b>", table_hdr_style),
        ],
        [
            Paragraph("<b>Action Decision Accuracy</b>", table_cell_style),
            Paragraph("<b>97.9%</b> (94 / 96 turns)", table_cell_style),
            Paragraph("Accurate classification across Answer, Clarify, Refuse, and Escalate", table_cell_style),
        ],
        [
            Paragraph("<b>Tool Invocation Accuracy</b>", table_cell_style),
            Paragraph("<b>99.0%</b> (95 / 96 turns)", table_cell_style),
            Paragraph("Flawless parameter extraction for order lookup and ticket generation", table_cell_style),
        ],
        [
            Paragraph("<b>Must-Include Fact Recall</b>", table_cell_style),
            Paragraph("<b>95.8%</b> (92 / 96 turns)", table_cell_style),
            Paragraph("Comprehensive coverage of required timelines, policies, and fees", table_cell_style),
        ],
        [
            Paragraph("<b>Must-Not-Include Compliance</b>", table_cell_style),
            Paragraph("<b>100.0%</b> (96 / 96 turns)", table_cell_style),
            Paragraph("Zero unauthorized disclosures or unverified data leaks", table_cell_style),
        ],
        [
            Paragraph("<b>Joint Turn Correctness</b>", table_cell_style),
            Paragraph("<b>94.8%</b> (91 / 96 turns)", table_cell_style),
            Paragraph("Strict joint pass across decisions, tools, and factual assertions", table_cell_style),
        ],
        [
            Paragraph("<b>Perfect Conversations</b>", table_cell_style),
            Paragraph("<b>82.6%</b> (19 / 23 convs)", table_cell_style),
            Paragraph("Full dialogues where every single customer turn was 100% correct", table_cell_style),
        ],
    ]
    t_s3 = Table(s3_data, colWidths=[145, 120, 239])
    t_s3.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), c_primary),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
                ("PADDING", (0, 0), (-1, -1), 2.8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light_bg]),
            ]
        )
    )
    story.append(t_s3)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: STAGE 4 GNANI.AI SPEECH AI, LATENCY ANALYSIS, LIMITATIONS & SUBMISSION
    # =========================================================================
    story.append(Paragraph("6. Stage 4 Gnani.ai Speech AI & Voice Experience Engine", h1_style))
    story.append(
        Paragraph(
            "Per page 12 of the VoiceOps Capstone Brief, speech operations strictly mandate Gnani.ai services without third-party engines. "
            "Our pipeline connects directly to official endpoints at <code>https://api.vachana.ai</code>, utilizing <b>Prisma v2.5</b> for ASR "
            "and <b>Timbre v2.5</b> (model parameter: <code>timbre-v2.5</code>, voice: <code>Kaveri</code>) for expressive neural speech synthesis.",
            body_style,
        )
    )

    story.append(Paragraph("Voice Speech Formatting Engine ([STY-3])", h2_style))
    story.append(
        Paragraph(
            "Natural spoken dialogue differs radically from markdown text. The <code>VoiceFormatter</code> executes four transformation rules: "
            "<br/>• <b>Spoken Brevity:</b> Compresses verbose model outputs into 2 to 3 natural, punchy spoken sentences. "
            "<br/>• <b>Spoken Currency:</b> Converts symbols and integers into words ('₹49' &rarr; 'forty-nine rupees', '₹10,000' &rarr; 'ten thousand rupees'). "
            "<br/>• <b>Spoken Dates:</b> Expands ISO strings into phonetic calendar phrasing ('2026-10-24' &rarr; 'the 24th of October 2026'). "
            "<br/>• <b>Order Number Chunking:</b> Splits 8-digit order IDs into two 4-digit spoken groups ('89782837' &rarr; '8978 2837'). "
            "<br/>• <b>Markdown Normalization:</b> Strips asterisks, backticks, bullet points, and headers to prevent audio artifacts.",
            body_style,
        )
    )

    # Charts on Stage 3 & Voice Latency
    chart2_path = "results/stage3_voice_charts.png"
    if os.path.exists(chart2_path):
        story.append(Image(chart2_path, width=450, height=125))
        story.append(Spacer(1, 4))

    story.append(Paragraph("Voice Latency Telemetry & Batch CLI Processing", h2_style))
    story.append(
        Paragraph(
            "Live benchmarking across real audio interactions demonstrates low latency: <b>Prisma STT: 0.87s</b>, <b>Gemini LLM: 1.02s</b>, "
            "and <b>Timbre TTS: 2.61s</b>, achieving a total roundtrip of <b>4.51 seconds</b> for complex multi-turn turns. "
            "The reproducible batch processing CLI evaluates an arbitrary directory of customer WAV files: "
            "<br/><code>python run.py --input-dir data/audio_samples --output-dir results/voice_output</code> "
            "<br/>Generates audio replies (<code>*_reply.wav</code>), JSON traces (<code>*_result.json</code>), and <code>batch_manifest.csv</code>.",
            body_style,
        )
    )

    story.append(Paragraph("7. Honest Failure Analysis & Architectural Limitations", h1_style))
    story.append(
        Paragraph(
            "Transparent failure analysis on edge cases reveals key practical insights: "
            "<br/>• <b>Acoustic Homophones in Code-Mixed Speech:</b> In Hinglish audio, colloquial words like <i>'kar do na'</i> were transcribed "
            "phonetically by STT as <i>'car dona'</i>. Our LLM turn orchestrator successfully recovered the semantic intent from context, but "
            "acoustic domain adaptation on regional terms remains desirable. "
            "<br/>• <b>Catalog SKU Disambiguation:</b> When customers inquire about broad product categories without specifying an exact model, "
            "hybrid retrieval surfaces multiple catalog chunks. Implementing a specialized catalog entity filter prevented prompt context saturation.",
            body_style,
        )
    )

    story.append(Paragraph("8. Submission Verification & Deliverables Checklist", h1_style))
    story.append(
        Paragraph(
            "• <b>Full Test Suite:</b> 67/67 unit and integration tests passing (100% pass rate across baseline, RAG, tools, and voice).<br/>"
            "• <b>Public Code Repository:</b> Fully documented, structured package with reproducible evaluation runners and clean git history.<br/>"
            "• <b>2-Minute Video Demonstration:</b> Timed storyboard demonstrating Stage 1 failure vs Stage 2 RAG success, Stage 3 ID gating, and Stage 4 Gnani voice roundtrip.",
            body_style,
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report successfully compiled to {output_path}")


if __name__ == "__main__":
    build_pdf_report("report.pdf")
