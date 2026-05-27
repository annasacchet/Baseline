"""Shared constants for the Llama-3.1-70B rewriting + eval pipelines.

The 70B model is loaded in 4-bit NF4 on Homer (2x RTX A6000 48GB).
Memory footprint at load: ~40 GB (model) + activations.

We pin the rewriter and the QA / perplexity / AFG judges to the same checkpoint
so PPL/F1 are computed under the rewriter's own distribution. AFV stays on
gemma-3-4b-it for consistency with the 600q pipeline (see memory:
feedback_afv_model_consistency).
"""

LLAMA_MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"
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
