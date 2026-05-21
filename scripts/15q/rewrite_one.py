"""
Riscrive UNA singola domanda MuSiQue, prendendo E0 dal CSV esistente
(step==0). Default: OLMo-3.1-32B-Instruct in 4-bit.

Template di rewriting allineato a scripts/15q/rewriting_pipeline.py
(usa delimitatori XML <source_text>...</source_text>).

Esempio:
  python scripts/15q/rewrite_one.py \\
      --csv results/300q/rewriting_chains_300q.csv \\
      --qid 2hop__635544_110949 \\
      --output rewrite_one_out.csv
"""
import argparse
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

Strict Rule: Return ONLY the rewritten text. Do not include any preamble, introduction, markdown formatting outside the text, or commentary.

Rewritten text:"""


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


@torch.no_grad()
def generate(tok, model, prompt, temperature, max_new_tokens, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = (system_prompt + "\n\n" if system_prompt else "") + prompt
    inputs = tok(text, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        kw.update(do_sample=False)
    out = model.generate(**inputs, **kw)
    new = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV con righe E0 (step==0)")
    ap.add_argument("--qid", required=True)
    ap.add_argument("--model", default="allenai/OLMo-3.1-32B-Instruct")
    ap.add_argument("--output", default="rewrite_one_out.csv")
    ap.add_argument("--n-iterations", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disabilita la quantizzazione 4-bit (default: 4-bit on)",
    )
    ap.add_argument(
        "--single-run",
        action="store_true",
        help="Solo paraphrase/run0 (smoke test veloce)",
    )
    ap.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt da anteporre al messaggio utente. "
             "Passa una stringa vuota ('') per disabilitarlo.",
    )
    args = ap.parse_args()
    use_4bit = not args.no_4bit
    system_prompt = args.system_prompt if args.system_prompt else None

    df = pd.read_csv(args.csv)
    e0_rows = df[(df.qid == args.qid) & (df.step == 0)]
    if e0_rows.empty:
        raise SystemExit(f"qid {args.qid} non trovato nel CSV con step==0")
    E0 = e0_rows.iloc[0]["text"]
    question = e0_rows.iloc[0]["question"]
    print(f"[+] qid={args.qid}")
    print(f"[+] question: {question}")
    print(f"[+] E0 length: {len(E0)} chars")
    print(f"[+] system prompt: {'ON' if system_prompt else 'OFF'}")

    print(f"[+] Loading {args.model} (4-bit={use_4bit}) ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()

    rows = []
    combos = list(ALL_INSTRUCTIONS.items())
    if args.single_run:
        combos = [
            (
                ("style", "paraphrase"),
                [ALL_INSTRUCTIONS[("style", "paraphrase")][0]],
            )
        ]

    for (group, itype), pool in combos:
        for run, instruction in enumerate(pool):
            print(f"\n=== {group}/{itype}/run{run} :: {instruction!r} ===")
            current = E0
            rows.append(
                dict(
                    qid=args.qid,
                    question=question,
                    group=group,
                    instruction_type=itype,
                    run=run,
                    instruction_used="",
                    step=0,
                    text=current,
                    n_tokens=len(tok.encode(current, add_special_tokens=False)),
                )
            )
            for step in range(1, args.n_iterations + 1):
                t0 = time.time()
                prompt = REWRITE_TEMPLATE.format(
                    instruction=instruction, text=current
                )
                current = generate(
                    tok,
                    model,
                    prompt,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    system_prompt=system_prompt,
                )
                elapsed = time.time() - t0
                ntok = len(tok.encode(current, add_special_tokens=False))
                print(f"  step {step}: {ntok} tokens ({elapsed:.1f}s)")
                print(f"    preview: {current[:200]}...")
                rows.append(
                    dict(
                        qid=args.qid,
                        question=question,
                        group=group,
                        instruction_type=itype,
                        run=run,
                        instruction_used=instruction,
                        step=step,
                        text=current,
                        n_tokens=ntok,
                    )
                )

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"\n[+] Salvato {len(rows)} righe in {args.output}")


if __name__ == "__main__":
    main()
