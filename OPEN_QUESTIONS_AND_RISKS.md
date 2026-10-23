# Open Questions, Risks, and Evidence Traceability Log

> **Project:** VoiceOps 2026 Capstone (ZentroMart Customer Support Agent)  
> **Course:** From Tokens to Transformers: A Course on GenAI × LLM Mastery (CAC, IIT Guwahati × Gnani.ai)  
> **Log Initialized:** 2026-10-22  
> **Status:** Active / Under Tracking  

---

## 1. Open Questions & Critical Risks

### RISK-01: Capstone Submission Deadline Discrepancy
* **Issue:** Contradictory deadline dates are published across official channels:
  * **Course Website (`https://caciitg.com/llm/`):** States `2026-10-26 Final submissions close` (Sunday night / Monday).
  * **Unstop Course Description:** States `25 Oct 2026 Capstone submission deadline and course conclusion.`
  * **Unstop API Metadata (`competition.end_date`):** Stores `2026-10-25T23:59:00+05:30`.
* **Current Status:** **UNRECONCILED**. It is not confirmed whether the portal closes at 23:59:00 IST on 25 October or extends through 26 October.
* **What Needs to Be Checked:**
  1. Inspect the live Unstop competition dashboard for an active countdown timer on the submission round.
  2. Monitor the official Discord announcements (`#announcements` on `https://discord.gg/ns9XTZhvwS`) for any formal clarification from the course organizers.
* **Mitigation Strategy:** Target all development, benchmarking, report writing, and packaging for completion by **25 October 2026, 18:00 IST** to eliminate the risk of automated lockout.

---

### RISK-02: Gnani.ai Speech API Credentials & Quota Availability
* **Issue:** Page 12 of the VoiceOps Capstone Brief strictly mandates: *"No other speech-to-text or text-to-speech service is allowed, even as a backup. Using any speech service other than Gnani's disqualifies a submission."*
* **Current Status:** **RESOLVED & VERIFIED LIVE**.
  * `GNANI_API_KEY` loaded securely via `.env` (git-ignored, verified without leaking/logging secrets).
  * Direct connectivity verified with official Gnani endpoints:
    - TTS: `https://api.vachana.ai/api/v1/tts/inference` (`timbre-v2.5`, voice `Kaveri`).
    - STT: `https://api.vachana.ai/stt/v3` (`Prisma v2.5`, language `en-IN`, `hi-IN`).
  * End-to-end audio roundtrip validated: TTS speech synthesis &rarr; WAV output &rarr; STT transcription matches spoken keywords.
  * Streaming WAV header size normalization implemented in `src/voice/client.py`.
  * Batch audio runner `run.py` executed successfully against test audio clips with complete speech output and manifest generation.

---

### RISK-03: Customer Audio Files & Batch Runner Validation
* **Issue:** The capstone requires a batch runner (`run.py`) taking a folder of customer audio files and outputting speech responses. However, `Capstone_Project_Dataset_FTTT_2026.zip` contains zero audio recordings; it provides only textual `asr_text` simulation columns.
* **Current Status:** **RESOLVED & VERIFIED LIVE**.
  * Created test customer audio suite in `data/audio_samples/`:
    - `turn1_greeting.wav` (Greeting inquiry, 2.24s)
    - `turn2_order_id.wav` (Order ID provision, 3.20s)
    - `turn3_verification.wav` (Phone digits verification, 3.20s)
    - `turn4_question.wav` (Hinglish high-value damage claim, 6.08s)
  * Batch runner `python run.py --input-dir data/audio_samples --output-dir results/voice_output` processes all 4 turns, generates speech replies (`*_reply.wav`), JSON trace artifacts (`*_result.json`), and `batch_manifest.csv` with 100% success.
  * Verified audio length constraints (&le; 60 seconds per STT REST limit).

---

### RISK-04: LLM API Rate Limits and Reproducibility
* **Issue:** The LLM chosen for Stage 1 (baseline) and Stage 2 (RAG) must support deterministic evaluation across 52 practice questions and 96 multi-turn conversation turns without rate-limiting timeouts.
* **Current Status:** **RESOLVED & VERIFIED**.
  * Model configured and active: `gemini-3.5-flash-lite` via official `google-genai` SDK.
  * Deterministic inference enforced with `temperature=0.0`.
  * Automated exponential backoff retry loop implemented for `429 RESOURCE_EXHAUSTED` and `503` temporary spikes.
  * Stage 1 Baseline (52/52) and Stage 2 Grounded RAG (52/52) evaluations completed with zero errors and persisted to `results/stage1_baseline.csv` and `results/stage2_rag.csv`.

---

## 2. Evidence and Traceability Framework

Every requirement and technical choice in this project is categorized into one of four evidence tiers:

| Tier | Definition | Standard of Verification |
| :--- | :--- | :--- |
| **Tier A: Explicitly Documented Facts** | Stated directly in authoritative official documentation (PDF Brief, Course Website, Unstop Text). | Verifiable by direct quote and document reference. |
| **Tier B: Verified from Actual Dataset** | Inspected, parsed, and confirmed from files inside `Capstone_Project_Dataset_FTTT_2026.zip`. | Validated via code execution (`validate_dataset.py`, pandas row counts, schema joins). |
| **Tier C: Official Documentation Confirmed** | Confirmed from official vendor/API documentation (`docs.gnani.ai`, PyPI metadata). | Validated against SDK specs, endpoint parameters, and network probes. |
| **Tier D: Recommendations & Assumptions** | Architectural designs, chunking strategies, prompt templates, or testing adapters. | Marked as pending empirical evaluation; never treated as pre-verified facts. |

### Evidence Classification Table

| Item / Requirement | Tier | Supporting Source & Evidence | Status |
| :--- | :---: | :--- | :--- |
| **Company Identity: ZentroMart** | **A & B** | PDF Brief p. 2; `CHANGELOG.md`; all 21 KB markdown files. | Verified |
| **Reference Date: 2026-10-20** | **B** | `README.md`; `ZIP_SCHEMA.md`; `kb/00_about_this_knowledge_base.md`. | Verified |
| **CSV Inventory (9 tables)** | **B** | 300 customers, 1,000 orders, 1,783 items, 70 products, 36 pincodes, 4 holidays, 244 logs, 52 questions, 23 conversations. Validated by `validate_dataset.py`. | Verified |
| **Document Precedence Hierarchy** | **B** | `kb/00_about_this_knowledge_base.md`: Policy > Agent Guide > Help > FAQ > Marketing. | Verified |
| **Electronics Return Window (7 vs 10 days)** | **B** | `kb/pol_returns_v4.md` (7 days from 2026-08-01; 10 days grandfathered strictly if delivered before 1 August 2026). Verified distinguishing delivery date from order date. | Verified |
| **Identity Verification Gate (lookup_order)** | **B** | `tools/tools.py`: Returns `verification_required: True` until phone last 4 or pincode matches. | Verified |
| **Ticket Summary Constraint (create_ticket)** | **B** | `tools/tools.py`: Must be &ge; 20 characters; fails with `created: False` otherwise. | Verified |
| **Speech Provider Mandate (Gnani Only)** | **A** | PDF Brief p. 12: Any third-party STT/TTS results in immediate disqualification. | Verified |
| **TTS Model Versioning Requirement** | **A & C** | PDF Brief p. 11 & `docs.gnani.ai`: Must explicitly pass `timbre-v2.5` or `timbre-v3.0`. | Verified |
| **Total Evaluation Rubric (100 Marks)** | **A** | PDF Brief p. 13: 25 Grounded, 15 Retrieval, 20 Conversation, 10 Self-Eval, 10 Voice E2E, 10 Voice Exp, 10 Report & Demo. | Verified |
| **Deliverable Limits (4 pages / 2 mins)** | **A** | PDF Brief p. 14: Report max 4 pages, demo video max 2 minutes. | Verified |
| **Submission Deadline (25 vs 26 Oct)** | **A** | Unstop Description (25 Oct) vs Course Website (26 Oct). | **UNRECONCILED** |
| **Live Gnani STT/TTS Connectivity** | **A & C** | Verified live via official Gnani REST API endpoints (`https://api.vachana.ai`) using `timbre-v2.5` (`Kaveri`) and `Prisma v2.5`. 12/12 voice tests passing. | **Verified Live** |
| **RAG Hybrid Retrieval Pipeline** | **B & C** | BM25 (`rank-bm25`) + FastEmbed dense (`BAAI/bge-small-en-v1.5`, English-only) + TF-IDF fallback + RRF + deterministic sorting. Hit Rate@5: 80.85%, MRR: 0.6667. | Verified |
| **Stage 1 vs Stage 2 Comparative Evaluation** | **B** | Evaluated on all 52 practice questions with `gemini-3.5-flash-lite`: Decision Accuracy 13.5% &rarr; 76.9%, Fact Recall 25.0% &rarr; 55.8%, Grounded Correctness 11.5% &rarr; 53.8%, Must-not-include 100% &rarr; 100%. | Verified |
| **Stage 3 Multi-Turn Conversational Evaluation** | **B** | Evaluated on all 23 conversations (96 turns) with `gemini-3.5-flash-lite`: Decision Accuracy 97.9%, Tool Call Accuracy 99.0%, Must-Include Fact Recall 95.8%, Must-Not-Include Passed 100.0%, Turn Overall Correctness 94.8%, Perfect Conversations 82.6% (19/23). | Verified |
