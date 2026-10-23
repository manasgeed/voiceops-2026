# VoiceOps 2026: 2-Minute Capstone Demo Video Script & Storyboard

> **Project:** VoiceOps 2026 — ZentroMart Customer Support Agent  
> **Course:** From Tokens to Transformers: A Course on GenAI × LLM Mastery (IIT Guwahati × Gnani.ai)  
> **Target Video Duration:** Exactly 1 minute 58 seconds (Strictly &le; 2:00 maximum limit)  
> **Resolution:** 1080p (1920×1080) at 30 fps  
> **Audio Tracks:** Clean spoken narration + Synthesized agent voice (Gnani Timbre `Kaveri`) + System screen capture  

---

## Storyboard Overview & Timeline

```
[0:00 - 0:20] Segment 1: Introduction & Architecture Overview (20s)
[0:20 - 0:50] Segment 2: Stage 1 Baseline Failure vs Stage 2 Grounded RAG Success (30s)
[0:50 - 1:25] Segment 3: Stage 3 Multi-Turn Dialogue, ID Verification Gate & Tools (35s)
[1:25 - 1:50] Segment 4: Stage 4 Gnani.ai Speech AI, Voice Formatting & Batch CLI (25s)
[1:50 - 1:58] Segment 5: Conclusion, 67-Test Verification & Deliverables (8s)
```

---

## Detailed Scene-by-Scene Script

### Segment 1: Introduction & Architecture Overview [0:00 – 0:20] (20 Seconds)

* **Visual on Screen:**
  * Title slide: **VoiceOps 2026: ZentroMart Customer Support Agent**.
  * Subtitle: *IIT Guwahati × Gnani.ai GenAI Capstone*.
  * Architectural Flow Diagram highlighting the 4 synchronized stages:
    1. *Gnani Prisma v2.5 STT*
    2. *Stateful Turn Manager with ID Verification Gate*
    3. *Precedence-Aware Hybrid RAG (BM25 + FastEmbed + RRF)*
    4. *Voice Formatter [STY-3] & Gnani Timbre v2.5 TTS*
* **Presenter Voiceover (0:00 – 0:20):**
  > "Hello! Welcome to VoiceOps 2026, an enterprise conversational voice agent built for ZentroMart, an Indian e-commerce platform. Over the next two minutes, we will demonstrate our full four-stage implementation: comparing an ungrounded LLM baseline with document-grounded RAG, testing multi-turn conversational tools with identity verification gating, and experiencing end-to-end voice AI powered exclusively by Gnani.ai."

---

### Segment 2: Stage 1 Baseline Failure vs. Stage 2 Grounded RAG [0:20 – 0:50] (30 Seconds)

* **Visual on Screen:**
  * Split-screen terminal comparison displaying the same tricky question:
    * *Query:* "I bought a 4K TV on July 25, delivered August 3, 2026. Can I return it under the 10-day policy?"
  * **Left Side (Stage 1 Baseline):**
    * Output: Baseline answers "Yes, 10 days apply because purchase was in July." *(Marked: FAIL — Hallucination).*
  * **Right Side (Stage 2 Grounded RAG):**
    * Output: RAG agent retrieves `pol_returns_v4.md` chunk `[POL-RET-4.2]`.
    * Output: "Under Policy v4, the 7-day electronics return window applies because delivery was on August 3, 2026. The 10-day window strictly applies to deliveries on or before July 31, 2026." *(Marked: PASS — Grounded Citation).*
  * Quick cut to benchmark comparison graphic:
    * Decision Accuracy: **13.5% &rarr; 76.9% (+63.4%)**
    * Fact Recall: **25.0% &rarr; 55.8% (+30.8%)**
    * Must-Not-Include Passed: **100% (Zero Hallucinations)**
* **Presenter Voiceover (0:20 – 0:50):**
  > "First, Stage 1 versus Stage 2. When asked about return windows for electronics, the ungrounded baseline hallucinates and checks the order date. Our Stage 2 Grounded RAG agent retrieves the authoritative policy document, recognizes that the grandfathering cutoff depends strictly on the delivery date, and correctly applies the 7-day limit with exact section citations. Across 52 benchmark questions, RAG boosted decision accuracy by 63.4% while maintaining zero hallucinations."

---

### Segment 3: Stage 3 Multi-Turn Dialogue, ID Verification & Tools [0:50 – 1:25] (35 Seconds)

* **Visual on Screen:**
  * Interactive terminal session running `python -m src.agent.evaluator` or live conversation:
  * **Turn 1:**
    * Customer: *"I want to check my order status."*
    * Agent: Prompts for 8-digit order number.
  * **Turn 2:**
    * Customer: *"Order number is 89782837."*
    * System: Executes `lookup_order(89782837)`.
    * Agent: Withholds details! Requests last 4 digits of phone or delivery pincode per `[IDV-2]`.
  * **Turn 3:**
    * Customer enters incorrect code *"9999"*.
    * Agent: Grants exactly 1 retry per `[IDV-3]`.
    * Customer enters *"4533"*.
    * Agent: Authenticates Kabir Sharma, reveals order status delivered Oct 15.
  * **Turn 4:**
    * Customer: *"The TV screen is cracked. Replace it."*
    * Agent: TV value is ₹24,999 (> ₹10,000 threshold per `[DWM-5]`).
    * System: Dispatches `create_ticket(summary >= 20 chars)` and escalates to human colleague.
* **Presenter Voiceover (0:50 – 1:25):**
  > "In Stage 3, the agent conducts multi-turn conversations with enterprise security. To protect customer privacy, all order details are strictly withheld until the customer verifies their phone suffix or delivery pincode. The agent enforces a strict one-retry limit. Once verified, when the customer reports damage on a television valued above ten thousand rupees, the agent executes our mock ticketing tool and smoothly escalates the case to a human colleague. Across 96 benchmark turns, Stage 3 scored 97.9% decision accuracy and 99% tool accuracy."

---

### Segment 4: Stage 4 Gnani.ai Speech AI, Voice Formatting & Batch CLI [1:25 – 1:50] (25 Seconds)

* **Visual on Screen:**
  * Execution of the batch audio CLI:
    ```bash
    python run.py --input-dir data/audio_samples --output-dir results/voice_output
    ```
  * Shows real-time waveform playback of `turn4_question_reply.wav`.
  * Graphic showing `VoiceFormatter` rule compliance (`[STY-3]`):
    * Markdown stripped.
    * Currency: `₹10,000` &rarr; *"ten thousand rupees"*.
    * Dates: `2026-10-24` &rarr; *"the 24th of October 2026"*.
    * Order grouping: `89782837` &rarr; *"8978 2837"*.
  * Audio plays clear synthesized speech from Gnani Timbre v2.5 (`Kaveri` voice):
    * *Audio Clip:* *"I see you are inquiring about the damaged Zentro Vista 43-inch 4K Smart TV in your order. Since the item is valued above ten thousand rupees, I have escalated this issue and created a support ticket for you."*
* **Presenter Voiceover (1:25 – 1:50):**
  > "Finally, Stage 4 integrates official Gnani.ai speech AI with zero third-party tools. Gnani Prisma v2.5 transcribes spoken audio, our VoiceFormatter converts dates into month names, groups order numbers into four digits, and speaks currency in words. Then, Gnani Timbre v2.5 synthesizes expressive natural speech in under three seconds. Our batch runner processes entire folders of audio clips, outputting audio replies, JSON traces, and manifest summaries."

---

### Segment 5: Conclusion & Deliverables [1:50 – 1:58] (8 Seconds)

* **Visual on Screen:**
  * Final slide showing:
    * `pytest tests/ -v`: **67 passed in 100% pass rate**.
    * Four-page technical report (`report.pdf`).
    * Public GitHub repository link.
* **Presenter Voiceover (1:50 – 1:58):**
  > "With 67 out of 67 tests passing, reproducible batch pipelines, and an authoritative four-page report, VoiceOps 2026 delivers a complete, production-ready system. Thank you!"

---

## Technical Specifications Checklist for Video Recording

| Requirement | Capstone Brief Mandate | Video Conformance | Status |
| :--- | :--- | :--- | :---: |
| **Duration Limit** | Strictly &le; 2 minutes (120 seconds) | Script timed at 1 minute 58 seconds | **Compliant** |
| **Stage 1 vs 2** | Mandatory comparative demonstration | Covered in Segment 2 (0:20 - 0:50) | **Compliant** |
| **Tools & Security** | ID verification gate & mock tools | Covered in Segment 3 (0:50 - 1:25) | **Compliant** |
| **Speech Engine** | Strictly Gnani.ai STT & TTS only | Powered by Prisma v2.5 & Timbre v2.5 in Segment 4 | **Compliant** |
| **Batch CLI** | Demonstrates reproducible CLI runner | Featured in Segment 4 (`python run.py`) | **Compliant** |
