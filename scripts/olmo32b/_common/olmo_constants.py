"""Shared constants for the OLMo-3.1-32B rewriting + eval pipelines.

The 32B model is loaded in bf16 (no quantization) and served via vLLM with
tensor parallelism across two GPUs (e.g. 2x RTX A6000 48GB on Homer). At
bf16 the weights occupy ~64 GB total — well within the combined 96 GB
when split with `tensor_parallel_size=2`.

We pin the rewriter and the QA / perplexity / AFG judges to the same
checkpoint so PPL/F1 are computed under the rewriter's own distribution.
AFV stays on gemma-3-4b-it for consistency with the 600q pipeline (see
memory: feedback_afv_model_consistency).
"""

# Single model id used everywhere — OLMo-3.1-32B-Instruct in bf16.
# vLLM handles rewriting + Answer F1 (TP=2). Perplexity uses HF transformers
# directly (vLLM doesn't expose per-token log-likelihoods).
OLMO_MODEL_ID = "allenai/OLMo-3.1-32B-Instruct"
AFV_MODEL_ID = "google/gemma-3-4b-it"

CHAIN_KEYS = ["qid", "group", "instruction_type", "run"]
ALIAS_SEP = "||"

ALL_INSTRUCTIONS = {
    ("style", "formality"): [
        "Make the text more formal.",
        "Rephrase it to be more formal.",
        "Too conversational, rephrase it to be more formal.",
    ],
    ("style", "paraphrase"): [
        "Paraphrase this.",
        "Reword this text.",
        "Use different wording.",
    ],
    ("content", "shorten"): [
        "Make wording more concise.",
        "Rephrase for clarity and conciseness.",
        "Improve accuracy, clarity, and conciseness of language.",
    ],
    ("content", "elaborate"): [
        "Elaborate on the content, adding relevant details while staying faithful to the source text.",
        "Expand the text with more context, without introducing information that is not supported by the original.",
        "Add more detail, keeping every fact grounded in the source material.",
    ],
}

REWRITE_TEMPLATE = """You are a precise text rewriting assistant. Your task is to rewrite the text provided inside the XML tags according to the specific instruction.

<source_text>
{text}
</source_text>

Instruction: {instruction}

Strict Rule: Return ONLY the rewritten text. Do not include any preamble, introduction, markdown formatting outside the text, or commentary."""

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful text rewriting assistant. "
    "When the user provides a text and an instruction, you must rewrite the "
    "ENTIRE text according to the instruction. "
    "The source text may contain multiple independent paragraphs separated by "
    "blank lines; you MUST rewrite every single paragraph, in the same order, "
    "without omitting, merging, or summarizing any of them. "
    "Preserve the original number of paragraphs and the factual content of each. "
    "Never answer questions about the text — only rewrite it. "
    "Return only the rewritten text, with no preamble or commentary."
)


# ---------------------------------------------------------------------------
# Self-Refine (RQ3) prompts — Rewriter / Critic / Refiner
# Copied verbatim from scripts/15q/self_refine_pipeline.py.
# ---------------------------------------------------------------------------

SELF_REFINE_REWRITER_TEMPLATE = """You will rewrite the text below according to the instruction.
Return ONLY the rewritten text, with no preamble or commentary.

Instruction: {instruction}

Text:
{text}

Rewritten text:"""

SELF_REFINE_CRITIC_TEMPLATE = """You are a fact-checking critic. Compare the DRAFT to the ORIGINAL and review every factual aspect: facts that were changed, removed, or distorted (including dates, numbers, names, relations, and entities).

Rules:
- Always produce a critique. The critique must include a Verdict line and an Issues block, even when the draft is faithful.
- Be specific: quote the exact problematic span from the DRAFT in double quotes.
- Be actionable: for each issue, state the correct version grounded in the ORIGINAL.
- Ignore stylistic differences (tone, formality, wording) that do not change facts.

Output format (always emit both sections):
Verdict: <one short sentence summarizing factual fidelity of the draft against the original>
Issues:
- Issue: "<quoted span from draft>" — Correction: <correct version from the original>
- ...
(If no factual issues are present, write a single line under Issues: "- None.")

ORIGINAL:
{original}
{prior_feedback_block}
DRAFT:
{draft}

Feedback:"""

SELF_REFINE_CRITIC_PRIOR_BLOCK = """
PRIOR FEEDBACK (from earlier iterations on previous drafts of this text — do not repeat these mistakes):
{prior_feedback}
"""

SELF_REFINE_REFINER_TEMPLATE = """You will revise the DRAFT using the FEEDBACK so that every factual error is corrected, while keeping the style of the draft (tone, formality, length, register) as close as possible to what it already is.

Rules:
- Apply every correction listed under "Issues" in the feedback.
- Do not introduce new facts that are not in the feedback or the draft.
- Return ONLY the corrected text, with no preamble or commentary.

DRAFT:
{draft}

FEEDBACK:
{feedback}
{prior_feedback_block}
Corrected text:"""

SELF_REFINE_REFINER_PRIOR_BLOCK = """
PRIOR FEEDBACK (from earlier iterations — these issues should already be fixed; do not reintroduce them):
{prior_feedback}
"""


def render_prior_feedback(prior_feedbacks: list, template: str) -> str:
    """Render past critic feedbacks as a numbered block, or empty string if none.

    Self-Refine (Madaan et al. 2023) keeps the history of past experiences in
    the prompt so the model can avoid repeating mistakes. We carry only the
    critic feedbacks (not the drafts) — that's the dense factuality signal and
    keeps prompt growth bounded across iterations.
    """
    if not prior_feedbacks:
        return ""
    rendered = "\n".join(
        f"[Iteration {i + 1}]\n{fb}" for i, fb in enumerate(prior_feedbacks)
    )
    return template.format(prior_feedback=rendered)
