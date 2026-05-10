# Support Triage Agent

Processes support tickets against a 774-document knowledge base spanning HackerRank, Claude, and Visa. Produces five fields per ticket: `status`, `product_area`, `response`, `justification`, `request_type`.

**Challenge:** HackerRank Orchestrate (May 2026)  
**Eval score:** 0.940 aggregate (10-ticket ground truth, five criteria)

---

## Quick Start

```bash
# Install dependencies
uv sync --project code/

# Configure API keys
cp code/.env.example code/.env
# DEEPSEEK_API_KEY is required; ANTHROPIC_API_KEY and GROQ_API_KEY are optional fallbacks

# Build corpus cache (run once, ~6 min, ~780 LLM calls)
LOG_LEVEL=INFO uv run --project code python code/build_corpus.py

# Process tickets
LOG_LEVEL=INFO uv run --project code python code/main.py
```

Output: `support_tickets/output.csv`

---

## Architecture

Two-phase design — offline corpus build and online per-ticket processing share no LLM calls at runtime.

```
PHASE 1: CORPUS BUILD (build_corpus.py, run once)

  774 raw .md files
      │
      ▼
  Step 1: Normalize (Python, no LLM)
      │   Parse frontmatter, strip HTML, extract title + first_paragraph
      ▼
  Step 2: Build Domain Taxonomy (deepseek-reasoner, 1 call per domain)
      │   Flat, mutually exclusive labels from all doc titles/paragraphs
      │   Quality gates: label format, doc count estimates, example IDs
      ▼
  Step 3: Per-Doc Enrichment (deepseek-chat, ~774 calls)
      │   product_area, doc_type, summary, keywords per doc
      │   Validates: confidence ≥ 0.7, ≥ 3 keywords, no invented labels
      ▼
  Step 4: Cross-Domain Reconciliation (deepseek-reasoner, 1 call)
      │   Merge taxonomies, add domain prefix to overlapping labels
      │   "billing" → "hackerrank:billing-subscription" vs "visa:billing-payments"
      ▼
  Step 5: Master Index (Python)
      │   doc_id → metadata lookup table
      ▼
  Step 6: BM25 Indexes (rank-bm25)
      │   Indexed text: title + product_area + keywords + summary + body
      │   Four indexes: hackerrank, claude, visa, all
      ▼
  corpus_cache/  (each step skips if artifacts exist)


PHASE 2: TICKET PROCESSING (main.py, per ticket)

  Ticket {issue, subject, company}
      │
      ▼
  Stage 0: Guardrails (deepseek-chat via LiteLLM)
      │   Three binary validators: prompt injection, malicious intent, support scope
      │   FAIL ──► replied / invalid  (short circuit, no further LLM calls)
      │ PASS
      ▼
  Stage 1: Reformulate (deepseek-chat)
      │   Strip emotion, fix grammar, resolve pronouns
      ▼
  Stage 2: Decompose (deepseek-chat)
      │   shared_context + list of contextualized sub-queries
      ▼
  Stage 2.5: Request Type (deepseek-chat)
      │   product_issue / bug / feature_request / invalid
      │   invalid ──► replied / invalid  (short circuit)
      ▼
  Stage 2.75: Domain Inference (BM25, only if company=None)
      │   Keyword match + BM25 on "all" index → single domain for all sub-queries
      │   No match ──► replied / invalid  (short circuit)
      ▼
  Stage 3: Per-Sub-Query, parallel (up to 4 threads)
      │
      │   3A: BM25 Retrieve  →  top-5 docs, re-ranked by doc_type vs request_type
      │       No docs → skip 3B, escalate immediately
      │   3B: Classify (deepseek-reasoner)
      │       answerable / escalation / invalid — judged against retrieved evidence
      │
      ▼
  Stage 4: Compose (deepseek-chat)
      │   Merges sub-query results; each doc shown with (domain:label) tag
      │   compose LLM reads doc tags to select product_area
      │   Status: LLM decides; prompt enforces escalated when ALL sub-queries non-answerable
      ▼
  Stage 5: Quality Check (deepseek-reasoner + deepseek-chat)
      │   Verifies: grounding, completeness, scope, coherence, contradiction
      │   Fail → regenerate (max 2×); still failing → return best attempt
      ▼
  Stage 6: Schema Validate (Python)
      │   Enum checks, non-empty fields, product_area in taxonomy
      │   INVALID request_type allows empty product_area
      │   Fail → escalate
      ▼
  output.csv row

  All LLM calls traced via Langfuse (model, tokens, latency per stage).
```

---

## Key Design Decisions

**BM25 over embeddings** — deterministic retrieval; same input always produces same output. Vocabulary gap closed by prepending LLM-generated keywords and summaries to the BM25 index at build time.

**LLM Wiki for taxonomy** — one enrichment process handles all three domain formats (HackerRank breadcrumbs, Claude categories, Visa URL paths). No per-domain parsers. Re-run enrichment when the corpus changes.

**Domain-scoped flat labels** — `hackerrank:billing-subscription` vs `visa:billing-payments` avoids cross-domain label collision while keeping product_area a single string.

**Retrieve then classify (Stage 3A → 3B)** — taxonomy-only classification is a prediction; evidence-grounded classification is a judgment. BM25 runs first (<1s); the reasoner then sees actual retrieved docs. If no docs return, Stage 3B is skipped entirely.

**Domain inference once per ticket (Stage 2.75)** — all sub-queries inherit the same inferred domain. Per-sub-query routing produces incoherent results when sub-query wording drifts across domains.

**Escalation = corpus can't resolve, not sensitivity** — prompt encodes this explicitly. A stolen card ticket with procedure docs is `answered`; a novel identity theft edge case without docs is `escalated`.

**Decompose before classifying request type** — compound tickets ("submissions broken and can you add retry?") reveal primary intent through structure: bug beats feature request.

**deepseek-chat for structure, deepseek-reasoner for judgment** — reasoner over-decomposes on structural tasks (4 sub-queries where 2 suffice). Chat under-performs on edge-case classification (sensitive topic → escalation). Each model used only where chain-of-thought changes the answer.

**Quality check returns best attempt on failure** — replacing a grounding-weak response with a generic "escalated for human review" string masks real pipeline output in eval. Stage 6 schema validation handles structural failures separately.

**Multi-provider fallback chain** — DeepSeek is primary; Claude (haiku/sonnet) and Groq (gpt-oss-120b) activate automatically if DeepSeek rate-limits. Each provider gets 3 retries with exponential backoff before the chain advances. Providers excluded at startup if their API key is absent.

**Langfuse observability** — per-stage token cost, latency, and model name recorded per ticket. Optional: pipeline runs identically without it.

---

## Model Usage

### Offline (~780 calls, ~$0.50 total)

| Step | Model | Calls |
|------|-------|-------|
| Step 2 — taxonomy | deepseek-reasoner | 3 (1/domain) |
| Step 3 — enrichment | deepseek-chat | ~774 |
| Step 4 — reconciliation | deepseek-reasoner | 1 |

### Online (~6-7 calls/ticket, ~$0.01/ticket)

| Stage | Model | Fallback |
|-------|-------|---------|
| 1 Reformulate | deepseek-chat | → claude-haiku → groq |
| 2 Decompose | deepseek-chat | → claude-haiku → groq |
| 2.5 Request type | deepseek-chat | → claude-haiku → groq |
| 3B Classify | deepseek-reasoner | → claude-sonnet → groq |
| 4 Compose | deepseek-chat | → claude-haiku → groq |
| 5 Verify | deepseek-reasoner | → claude-sonnet → groq |
| 5 Regenerate | deepseek-chat | → claude-haiku → groq |

temperature=0 everywhere for determinism.

---

## Project Structure

```
code/
  build_corpus.py
  main.py
  schemas.py
  config.py
  exceptions.py        # PipelineError hierarchy (ProviderError, JSONParseError, …)
  stages/
    stage0_guardrails.py
    stage1_reformulate.py
    stage2_decompose.py
    stage25_request_type.py
    stage275_domain_inference.py
    stage3_retrieve.py
    stage3_classify.py
    stage4_compose.py
    stage5_quality.py
    stage6_validate.py
  corpus/
    normalize.py
    taxonomy.py
    enrich.py
    reconcile.py
    index.py
  prompts/             # versioned YAML prompt templates, one subdirectory per stage
    stage0_guardrails/v1.yaml
    stage1_reformulate/v1.yaml
    stage2_decompose/v1.yaml
    stage25_request_type/v1.yaml
    stage3_classify/v1.yaml
    stage4_compose/v1.yaml
    stage5_quality/v1.yaml
    corpus_enrich/v1.yaml
    corpus_taxonomy/v1.yaml
    corpus_reconcile/v1.yaml
  utils/
    llm.py
    provider.py        # DeepSeek / Claude / Groq fallback chain
    stage_handler.py   # @stage_handler decorator — maps raw exceptions to PipelineError subtypes
    prompt_registry.py # loads versioned YAML templates; logs version per ticket
    bm25.py
eval/
  run_eval.py          # LLM-as-judge against ground_truth.csv; outputs aggregate score
  collect_feedback.py  # collects human feedback on pipeline outputs
  promote_to_ground_truth.py
  judge_prompts/
  results/
```

---

## Environment

| Variable | Notes |
|----------|-------|
| `DEEPSEEK_API_KEY` | Primary LLM provider |
| `ANTHROPIC_API_KEY` | Enables Claude fallback |
| `GROQ_API_KEY` | Enables Groq fallback |
| `LANGFUSE_SECRET_KEY` | Enables tracing |
| `LANGFUSE_PUBLIC_KEY` | Required with secret key |
| `LANGFUSE_BASE_URL` | Default: `https://cloud.langfuse.com` |

