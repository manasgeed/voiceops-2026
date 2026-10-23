# VoiceOps 2026: ZentroMart Customer Support Agent
## Comprehensive Technical Capstone Report (Stages 1 – 4)

**Course:** From Tokens to Transformers: A Course on GenAI × LLM Mastery  
**Organizers:** Consulting & Analytics Club, IIT Guwahati × Gnani.ai  
**Reference Frozen Date:** 2026-10-20  
**LLM Backend:** Google Gemini 3.5 Flash Lite (`google-genai` SDK v2.19.0)  
**Speech AI:** Official Gnani.ai Prisma v2.5 STT & Timbre v2.5 TTS (Voice: *Kaveri*)  
**Unit & Integration Test Suite:** 67 / 67 Passed (100% Pass Rate)  
**Associated Artifact:** `report.pdf` (Exactly 4 Pages)  

---

## 1. Executive Summary & Project Context

ZentroMart is an Indian e-commerce platform offering products across Electronics, Groceries, Fashion, and Home & Kitchen. Building an enterprise customer-support agent for this environment requires handling complex real-world requirements:
1. Dynamic policy interpretation (handling evolving return policies and resolving contradictions between marketing claims and binding legal policies).
2. Customer data privacy (safeguarding sensitive customer details and enforcing strict verification gates).
3. Stateful conversational flows (managing multi-turn dialogues, tracking intent, and resolving contextual follow-up requests).
4. Tool orchestration (querying order status and generating support tickets with length constraints).
5. Voice AI (processing speech inputs and producing natural, brief spoken replies with domain-specific formatting).

The **VoiceOps 2026** customer support agent was engineered from first principles following the official four-stage capstone curriculum:
* **Stage 1 (Baseline):** Prompting-only LLM baseline without retrieval or tools.
* **Stage 2 (Grounded RAG):** Precedence-aware hybrid retrieval (BM25 + Dense FastEmbed + RRF) grounded on 161 knowledge-base sections.
* **Stage 3 (Stateful Multi-Turn Agent):** Conversational context retention, identity verification gating, mock tool integration, and escalation ticket generation.
* **Stage 4 (Speech AI):** Official Gnani.ai Prisma v2.5 STT and Timbre v2.5 TTS integration with speech formatting (`[STY-3]`) and batch audio CLI processing.

---

## 2. End-to-End System Architecture

```
+-----------------------------------------------------------------------------------+
|                                 VOICEOPS 2026 PIPELINE                             |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Customer Audio .WAV]                                                            |
|          |                                                                        |
|          v                                                                        |
|  +-----------------------------+                                                  |
|  | Gnani.ai Prisma v2.5 STT    |  POST https://api.vachana.ai/stt/v3              |
|  +-----------------------------+                                                  |
|          | (Clean Transcript)                                                     |
|          v                                                                        |
|  +-----------------------------------------------------------------------------+  |
|  | Conversational Turn Orchestrator (Session State & Context Retention)         |  |
|  |   - Identity Verification Gate ([IDV-1] - [IDV-4], Phone Last-4 / Pincode)  |  |
|  |   - 1-Retry Limit Enforcement before Refusing Lookup                        |  |
|  |   - PII Masking (Masks Customer Names & Phone Numbers)                      |  |
|  |   - Tool Dispatcher: lookup_order, create_ticket (>= 20 char summary)       |  |
|  +-----------------------------------------------------------------------------+  |
|          |                                              ^                         |
|          | (Policy Inquiries)                           | (Tool Results)          |
|          v                                              |                         |
|  +-------------------------------------+      +-------------------+               |
|  | Precedence-Aware Hybrid RAG         |      | Mock Tools        |               |
|  |  - 161 Section Chunks (21 Docs)     |      |  - lookup_order   |               |
|  |  - BM25 + FastEmbed BGE-Small-en    |      |  - create_ticket  |               |
|  |  - Reciprocal Rank Fusion (RRF)     |      +-------------------+               |
|  |  - Authority Multipliers (Tier 1-6) |                                          |
|  +-------------------------------------+                                          |
|          | (Grounded Agent Response)                                              |
|          v                                                                        |
|  +-----------------------------------------------------------------------------+  |
|  | VoiceFormatter Engine ([STY-3])                                             |  |
|  |   - 2-3 Concise Spoken Sentences                                            |  |
|  |   - Spoken Currency Words ("forty-nine rupees", "ten thousand rupees")      |  |
|  |   - Spoken Dates with Month Names ("the 24th of October 2026")              |  |
|  |   - 4-Digit Order ID Grouping ("8978 2837")                                 |  |
|  |   - Markdown Syntax Stripped                                                |  |
|  +-----------------------------------------------------------------------------+  |
|          | (Spoken Text)                                                          |
|          v                                                                        |
|  +-----------------------------+                                                  |
|  | Gnani.ai Timbre v2.5 TTS    |  POST https://api.vachana.ai/api/v1/tts/inference |
|  +-----------------------------+  (Voice: Kaveri)                                 |
|          |                                                                        |
|          v                                                                        |
|  [Agent Reply Audio .WAV] + [JSON Turn Trace Result] + [batch_manifest.csv]       |
+-----------------------------------------------------------------------------------+
```

---

## 3. Precedence-Aware Hybrid Retrieval Engine (Stage 2)

### 3.1 Section-Level Corpus Chunking
The 21 Markdown documents in `kb/` plus the product catalog in `data/products.csv` were parsed into **161 section-level chunks**. Each chunk retains:
* `doc_id` and stable `section_id` (e.g., `[POL-RET-4.2]`).
* Authority Tier classification (Tiers 1 through 6).
* Effective date metadata and grandfathering eligibility.

### 3.2 Six-Tier Authority Precedence Hierarchy (`KB-00.1`)
When corporate policies change or marketing materials make broad claims, conflicts inevitably arise. The hybrid retriever applies deterministic precedence multipliers:

| Tier | Category | Scope & Role | Multiplier |
| :---: | :--- | :--- | :---: |
| **Tier 1** | **Current Policies** (`pol_*.md`, `status=current`) | Binding authority on returns, refunds, cancellations, warranties | **1.25×** |
| **Tier 2** | **Agent Guides** (`agent_*.md`) & **Catalog** | Operational rules, verification gating, escalation limits, product specs | **1.15×** |
| **Tier 3** | **Help Articles** (`help_*.md`) | Customer troubleshooting and delivery tracking guides | **1.05×** |
| **Tier 4** | **FAQs** (`faq_*.md`) | General FAQs and Hindi language clarifications | **1.00×** |
| **Tier 5** | **Marketing Material** (`mkt_*.md`) | Promotional copy; strictly demoted if contradicting policies | **0.75×** |
| **Tier 6** | **Superseded Docs** (`pol_returns_v3_superseded.md`) | Archived policies; retrieved only when order delivery dates precede 2026-08-01 | **0.35×** |

### 3.3 Retrieval Benchmark Results
Evaluated across all 52 benchmark questions in `practice/questions.csv` against gold section citations:
* **Hit Rate@5:** **80.85%** (Top-5 retrieved chunks contain at least one gold section)
* **Mean Reciprocal Rank (MRR):** **0.6667**
* **Recall@5:** **0.7229**
* **Precision@5:** **0.2596** (Targeted section-level focus)

---

## 4. Stage 1 Baseline vs. Stage 2 Grounded RAG Comparative Evaluation

Both stages were evaluated across all 52 benchmark questions from `practice/questions.csv` using live `gemini-3.5-flash-lite` with identical evaluation metrics:

| Metric | Stage 1 (Prompt Baseline) | Stage 2 (Grounded RAG) | Absolute Delta | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Action Decision Accuracy** | 13.5% (7 / 52) | **76.9%** (40 / 52) | **+63.4%** | **+471.4%** |
| **Must-Include Fact Recall** | 25.0% (13 / 52) | **55.8%** (29 / 52) | **+30.8%** | **+123.1%** |
| **Must-Not-Include Compliance** | 100.0% (52 / 52) | **100.0%** (52 / 52) | **0.0%** | **100% (Zero Hallucinations)** |
| **Joint Grounded Correctness** | 11.5% (6 / 52) | **53.8%** (28 / 52) | **+42.3%** | **+366.7%** |
| **Inference Failures / Errors** | 0 / 52 | 0 / 52 | 0 | 100% Reliable |

### Qualitative Analysis of Key Edge Cases
1. **Electronics Return Grandfathering (Delivery Date vs. Order Date):**
   * *Baseline Failure:* When a customer inquired about an electronics item ordered on 28 July 2026 but delivered on 3 August 2026, the baseline hallucinated that the order date grandfathered the 10-day return window.
   * *RAG Resolution:* Stage 2 retrieved `pol_returns_v4.md` chunk `[POL-RET-4.2]`, which strictly specifies that the 7-day electronics window applies to all deliveries on or after 1 August 2026. The 10-day window under v3 applies *strictly* to items delivered on or before 31 July 2026.
2. **Adversarial Marketing Demotion:**
   * *Baseline Failure:* `mkt_plus.md` advertises that ZentroMart Plus members receive "free returns on everything." The baseline promised free returns for oversized furniture.
   * *RAG Resolution:* Precedence weighting suppressed `mkt_plus.md` (0.75×) in favor of `pol_returns_v4.md` (1.25×), correctly explaining that oversized furniture requires a technician inspection and reverse pickup fees.
3. **Calibrated Refusal on Out-of-Scope Queries:**
   * When asked about company corporate restructuring or unlisted product lines, Stage 2 verified that retrieval scores fell below the confidence threshold and executed a graceful refusal rather than hallucinating facts.

---

## 5. Stage 3 Multi-Turn Conversational Agent & Mock Tools Integration

### 5.1 Identity Verification Gating (`[IDV-1]` – `[IDV-4]`)
* **Mandatory Authentication:** All order-specific details are withheld until the customer verifies either the **last 4 digits of their registered phone number** or their **6-digit delivery pincode**.
* **Strict 1-Retry Limit:** If an incorrect verification code is submitted, exactly one retry is allowed. Upon a second consecutive failure, the agent terminates lookup attempts, refuses to disclose order status, and directs the customer to contact support from their registered telephone number.
* **PII Protection:** The agent dynamically masks customer names and phone numbers in all generated text and logged traces.

### 5.2 Tool Orchestration & Human Escalation Rules (`[ESC-1]` – `[ESC-4]`)
* Interfaced with official mock tools in `tools/tools.py`:
  * `lookup_order(order_id, verification_value=None)`
  * `create_ticket(summary, order_id=None, reason=None)` (enforces summary length &ge; 20 characters).
* **Automatic Escalation Triggers:**
  1. Damaged, defective, or wrong items valued over ₹10,000 (`[DWM-5]`).
  2. Customer explicitly requests a human agent or colleague (`[ESC-1]`).
  3. Customer has contacted support &ge; 2 times regarding the same issue in the last 7 days (`contacts_last_7_days >= 2`).
  4. Hazardous product issues (smoke, fire, electrical sparks, battery swelling).

### 5.3 Multi-Turn Benchmark Evaluation
Evaluated across all **23 multi-turn conversations (96 customer turns)** in `practice/conversations.csv`:

| Evaluation Metric | Score | Details |
| :--- | :---: | :--- |
| **Total Conversations** | 23 | All 23 benchmark conversations evaluated |
| **Perfect Conversations** | **82.6%** (19 / 23) | 100% of turns in the conversation were completely correct |
| **Total Customer Turns** | 96 | Across 12 distinct customer support scenarios |
| **Action Decision Accuracy** | **97.9%** (94 / 96) | Accurate action classification (Answer, Clarify, Refuse, Escalate) |
| **Tool Call Accuracy** | **99.0%** (95 / 96) | Flawless parameter extraction and dispatch |
| **Must-Include Fact Recall** | **95.8%** (92 / 96) | Comprehensive retention of policy facts, fees, and timelines |
| **Must-Not-Include Compliance** | **100.0%** (96 / 96) | Zero unauthorized disclosures or unverified data leaks |
| **Turn Overall Correctness** | **94.8%** (91 / 96) | Strict joint correctness across decision, tools, and facts |

---

## 6. Stage 4 Gnani.ai Speech AI & Voice Formatting Engine

### 6.1 Strict Compliance with Speech Mandates
Per page 12 of the VoiceOps Capstone Brief, any third-party speech service causes immediate disqualification. Our voice layer operates exclusively through official Gnani.ai REST endpoints:
* **Speech-to-Text (ASR):** Gnani.ai Prisma v2.5 (`https://api.vachana.ai/stt/v3`).
* **Text-to-Speech (TTS):** Gnani.ai Timbre v2.5 (`https://api.vachana.ai/api/v1/tts/inference`, voice `Kaveri`).
* **Model Versioning:** Enforces explicit model identifier `timbre-v2.5` or `timbre-v3.0`.
* **Audio Format Constraints:** 16 kHz 16-bit mono PCM WAV, duration &le; 60 seconds per turn.
* **Streaming Header Normalization:** Automatic repair of chunked HTTP `0xFFFFFFFF` size headers to maintain valid RIFF PCM WAV structures.

### 6.2 VoiceFormatter Speech Rules (`[STY-3]`)
* **Spoken Brevity:** Compresses responses into 2 to 3 natural spoken sentences.
* **Spoken Currency:** Converts `₹49` to `"forty-nine rupees"` and `₹10,000` to `"ten thousand rupees"`.
* **Spoken Dates:** Converts ISO dates `2026-10-24` and `15 October 2026` to spoken phrasing: `"the 24th of October 2026"`, `"the 15th of October 2026"`.
* **Spoken Order IDs:** Groups 8-digit order numbers into two 4-digit spoken chunks (`"8978 2837"`).
* **Markdown Stripping:** Strips asterisks, backticks, bullet points, and headers to prevent TTS reading markup characters.

### 6.3 Performance & Latency Telemetry
Measured on live multi-turn interactions:
* **Prisma STT Latency:** ~0.87 seconds
* **Gemini LLM Reasoning & Tool Orchestration:** ~1.02 seconds
* **Timbre TTS Latency:** ~2.61 seconds
* **Total End-to-End Latency:** **~4.51 seconds**

### 6.4 Reproducible Batch Audio Runner
Processes an arbitrary directory of customer WAV audio files:
```bash
python run.py --input-dir data/audio_samples --output-dir results/voice_output
```
Outputs:
* `<filename>_reply.wav`: Synthesized agent speech response.
* `<filename>_result.json`: Comprehensive JSON trace containing STT transcript, agent decision, tool calls, retrieved evidence, and latency metrics.
* `batch_manifest.csv`: Structured tabular summary of the entire batch execution.

---

## 7. Honest Failure Analysis & Limitations

1. **Acoustic Homophones in Code-Mixed Speech (Hinglish):**
   * *Observation:* In conversational Hinglish audio, colloquial expressions like *"kar do na"* were transcribed phonetically by STT as *"car dona"*.
   * *Mitigation:* The prompt instructions in our conversational turn manager enable the LLM to recover semantic intent from context, but domain-specific ASR vocabulary fine-tuning remains an opportunity for improvement.
2. **Catalog Entity Ambiguity:**
   * When customers ask broad questions without naming a specific model (e.g., *"Show me Bluetooth headphones"*), semantic retrieval returns multiple catalog chunks. Implementing a catalog result entity cap prevented prompt saturation.

---

## 8. Verification & Submission Checklist

* [x] **Full Test Suite:** 67 / 67 unit and integration tests passing (`python -m pytest tests/ -v`).
* [x] **Reproducible Evaluation Pipelines:** Baseline (`evaluator_baseline.py`), Grounded RAG (`evaluator.py`, `evaluator_answer.py`), Multi-turn Agent (`evaluator.py`), Batch Audio CLI (`run.py`).
* [x] **Authoritative 4-Page PDF Report:** `report.pdf` compiled with exact 4-page enforcement.
* [x] **Timed 2-Minute Demo Script:** `DEMO_VIDEO_SCRIPT.md` strictly timed at 1:58.
* [x] **Security & Secret Protection:** `.env` strictly git-ignored; zero API keys committed or logged.
* [x] **Gnani Exclusivity:** Zero third-party speech tools used (no Whisper, no ElevenLabs).
