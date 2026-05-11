---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 19px;
    padding: 36px 54px;
    color: #1a1a2e;
  }
  h1 { font-size: 38px; color: #1a1a2e; margin-bottom: 10px; }
  h2 { font-size: 26px; color: #1a1a2e; border-bottom: 3px solid #4e79a7; padding-bottom: 5px; margin-bottom: 14px; }
  h3 { font-size: 20px; color: #4e79a7; margin-bottom: 6px; }
  table { font-size: 16px; width: 100%; border-collapse: collapse; }
  th { background: #4e79a7; color: white; padding: 7px 12px; text-align: left; }
  td { padding: 6px 12px; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) td { background: #f5f8fc; }
  .note { font-size: 14px; color: #777; margin-top: 8px; font-style: italic; }
  .callout {
    border-left: 4px solid #4e79a7;
    padding: 9px 18px;
    background: #f0f4f8;
    border-radius: 0 6px 6px 0;
    color: #1a1a2e;
    margin: 12px 0;
    font-size: 18px;
  }
  .warn-box {
    border-left: 4px solid #e15759;
    padding: 9px 18px;
    background: #fdf3f3;
    border-radius: 0 6px 6px 0;
    color: #1a1a2e;
    margin: 12px 0;
    font-size: 18px;
  }
  .ok-box {
    border-left: 4px solid #59a14f;
    padding: 9px 18px;
    background: #f4faf3;
    border-radius: 0 6px 6px 0;
    color: #1a1a2e;
    margin: 12px 0;
    font-size: 18px;
  }
  .red { color: #e15759; font-weight: bold; }
  .blue { color: #4e79a7; font-weight: bold; }
  .green { color: #59a14f; font-weight: bold; }
  .orange { color: #f28e2b; font-weight: bold; }
  ul { margin-top: 4px; }
  li { margin-bottom: 4px; }
  p { margin: 6px 0; }
---

# Exploratory Analyses & Next Steps

### `elaborate` on 15q · 15q ↔ 300q overlap · Two new datasets

<br>

**Anna Sacchet** &nbsp;·&nbsp; 2026-05-09

---

## Roadmap

1. **`elaborate` on 15q** — what does the model add when it "elaborates"?
2. **15q ↔ 300q overlap** — same questions, different rewriting behaviour
3. **Two new datasets** — pipelines ready, awaiting GPU
4. **Next steps** — priorities and open questions

<div class="callout">
This deck is <b>exploratory</b>. The numbers are intended as a high-level
picture of what is going on, not as final benchmarks. Limits and caveats
are stated explicitly throughout.
</div>

---

# Part 1
# `elaborate` on 15q

### From a puzzle in the metrics
### to a Maynez-style hallucination check

---

## The starting puzzle

Iteration step 1 → step 3, 15q pilot, all four instructions:

| Instruction | ΔF1 | ΔOFS |
|-------------|-----|------|
| paraphrase  | −0.042 | ~0.000 |
| formality   | −0.057 | ~0.000 |
| **elaborate** | <span class="red">−0.104</span> | <span class="red">−0.065</span> |
| shorten     | <span class="red">−0.143</span> | −0.008 |

- **F1**: does the QA model still answer correctly from the rewrite?
- **OFS** (OpenFactScore): how much of the rewrite is grounded in E₀?

<div class="callout">
<code>shorten</code> drops F1 the most but its OFS is flat — it just <b>removes</b> facts.<br>
<code>elaborate</code> is the only instruction where <b>both</b> F1 and OFS drop.<br>
Something is being <b>added</b> to the text, and it is hurting the QA model.
</div>

**The question for Part 1:** *what exactly is `elaborate` adding?*

---

## First attempt · qualitative LLM-as-a-judge

We asked GPT-4o-mini to read each (E₀, rewrite) pair and write
free-text commentary, plus two lists: `lost` and `added` claims.

Output: 180 reports across 45 chains × 4 step-comparisons.

What it told us — qualitatively:

- The rewrite shifts register: encyclopedic → narrative, often promotional
- Most additions are stylistic ("important naval vessel", "leaving an indelible mark")
- Specificity (dates, numbers, exact names) gets smoothed into looser prose
- Genuine factual fabrications exist but seem rare

<div class="warn-box">
This was a <b>reading tool</b>, not a measurement.<br>
Counts mix evaluative adjectives, vague filler and concrete fabricated
facts in the same <code>n_added</code> number — too coarse to draw
conclusions from.
</div>

We needed a more rigorous framing.

---

## The taxonomy we adopted · Maynez et al. 2020

Reference paper: **Maynez, Narayan, Bohnet, McDonald (ACL 2020)** — *On Faithfulness and Factuality in Abstractive Summarization*.

| Type | Definition | Reference of truth |
|------|------------|--------------------|
| **Intrinsic** | Output **contradicts** the input | Input (here: E₀) |
| **Extrinsic** | Output **cannot be verified** from the input | Input |

In plain words:

- **Intrinsic** — rewrite says something the source directly contradicts.
  *E₀: "Clara Morris born 1849"; rewrite: "born 1949".*
- **Extrinsic** — rewrite says something the source doesn't mention.
  Could be true or false in the world; the source is silent.

<span class="note">This taxonomy is the standard reference in summarization.
Cited as foundational by the Huang et al. 2024 LLM hallucination survey
(<i>A Survey on Hallucination in Large Language Models</i>).</span>

---

## What we can measure · what we cannot

| Check | What it asks | Reference |
|-------|--------------|-----------|
| **Intrinsic** | Does the claim contradict E₀? | E₀ alone |
| **Extrinsic** | Is the claim absent from E₀ but verifiable elsewhere? | external knowledge |

<div class="ok-box">
<b>Intrinsic</b> is directly measurable — we have E₀ and the claim, no
external knowledge needed.
</div>

<div class="warn-box">
<b>Extrinsic truth</b> requires a knowledge base (Wikipedia, web search).<br>
Out of scope for this exploratory pass — would require SAFE-style retrieval
(Wei et al. 2024).
</div>

**Choice:** focus on the intrinsic check. The result is a **strict lower
bound** — claims marked "contradicts" are unambiguously hallucinations;
"neutral" claims may still be hallucinated, we just cannot tell from E₀.

---

## Pipeline · two ingredients

**Ingredient 1 · OpenFactScore** (already in the repo).
For each rewrite:
1. break text into atomic claims (single facts)
2. label each claim `SUPPORTED` or `NOT_SUPPORTED` against E₀

→ The `SUPPORTED` ones are fine. We focus on `NOT_SUPPORTED`.

**Ingredient 2 · NLI-style judge** (new).
For each `NOT_SUPPORTED` claim, GPT-4o-mini receives:

```
SOURCE: <full text of E₀>
CLAIM:  <one atomic claim from the rewrite>

Is the source contradicting this claim, or just silent on it?
→ contradicts | neutral
```

<div class="callout">
Same pattern as <b>SummaC</b> (Laban et al. 2022): sentence-level NLI for
factual consistency in summarization. Reduced here to a binary
contradiction check.
</div>

---

## What "contradicts" looks like

Real example from our run:

<div class="warn-box">
<b>CLAIM (rewrite):</b> "New South Wales's formation occurred in 1818."<br><br>
<b>FROM E₀:</b> "records of celebrations on 26 January date back to <b>1808</b>,
with the first official celebration of the formation of New South Wales..."<br><br>
Rewrite says 1818. Source says 1808. Numerically incompatible →
<code>contradicts</code> ✓
</div>

---

## What "neutral" looks like

Real example from our run:

<div class="ok-box">
<b>CLAIM (rewrite):</b> "The entertainment systems include DVD players and high-definition monitors."<br><br>
<b>FROM E₀:</b> "The Hyundai Universe is a heavy-duty luxury coach built by Hyundai Motor Company."<br><br>
E₀ doesn't mention DVD players. It doesn't confirm them, but doesn't deny
them either. The rewrite is just adding plausible details. →
<code>neutral</code> ✓
</div>

The rewrite *might* be making it up. The source-only check cannot tell.
Resolving "neutral → true vs false in the world" needs an external knowledge
base.

---

## Run status · partial but representative

API quota was hit mid-run. **2725 / 3981 NOT_SUPPORTED claims classified.**

| Step | NOT_SUPPORTED total | classified | coverage |
|------|---------------------|------------|----------|
| 1 | 988 | 646 | 65% |
| 2 | 1283 | 922 | 72% |
| 3 | 1710 | 1157 | 68% |

<div class="ok-box">
Coverage is <b>balanced across steps</b> (65–72%); the script
checkpoints every 25 claims, so the partial sample is not biased toward
early or late steps. <b>Per-step trends are reliable.</b>
</div>

<span class="note">Resume is supported. Completing the run takes ~15 min and
~$0.30 (one command: <code>python scripts/15q/hallucination_taxonomy_eval.py</code>).</span>

---

## Results · taxonomy distribution

Of the 2725 NOT_SUPPORTED claims classified:

| Step | <span class="red">contradicts</span> (intrinsic) | neutral (just absent) |
|------|-------|---------|
| 1 | <span class="red"><b>10.5%</b></span> | 89.5% |
| 2 | 9.1% | 90.9% |
| 3 | <span class="red"><b>6.0%</b></span> | 94.0% |

Translated to **% of all atomic facts** in the rewrite (extrapolated):

| Step | Intrinsic hallucinations |
|------|--------------------------|
| 1 | ~1.4% |
| 2 | ~1.5% |
| 3 | ~1.2% |

<div class="callout">
Direct contradictions of E₀ are <b>rare in absolute terms</b>: ~1% of all facts.<br>
The vast majority of non-grounded claims (~90%) are <b>additions</b>, not contradictions.
</div>

---

## Two surprising patterns

**1. Most non-grounded claims are NOT contradictions.**
The model is mostly *adding* content the source doesn't mention,
not *fighting* what the source says.

**2. Contradictions DECREASE with iterations** (10.5% → 6.0%).
Counter-intuitive: more rewriting iterations produce *fewer* direct
contradictions of E₀, not more.

<div class="callout">
<b>Plausible reading.</b> By step 3 the rewrite is so distant from E₀
that it stops engaging with E₀'s specific assertions — and therefore stops
contradicting them.<br>
The text is <b>drifting away from the source</b>, into the model's own
parametric narrative.
</div>

---

## How does this relate to OFS?

OFS gives a top-line groundedness number. The taxonomy adds **resolution
inside** OFS's NOT_SUPPORTED bucket.

| Step | OFS | NOT_SUPPORTED rate (1−OFS) | of which contradicts | of which just absent |
|------|------|----------------------------|----------------------|----------------------|
| 1 | 0.872 | <span class="red">13.5%</span> | 10.5% (=1.4% of all facts) | 89.5% (=12.1%) |
| 2 | 0.846 | <span class="red">16.7%</span> | 9.1% (=1.5%) | 90.9% (=15.2%) |
| 3 | 0.807 | <span class="red">20.8%</span> | 6.0% (=1.2%) | 94.0% (=19.5%) |

<div class="callout">
<b>OFS asks:</b> <i>"How much of the rewrite is not grounded in E₀?"</i>
&nbsp;→&nbsp; goes 13% → 21%.<br>
<b>Taxonomy asks:</b> <i>"Of that ungrounded portion, how much directly
contradicts E₀?"</i>
&nbsp;→&nbsp; only ~1% of all facts. The rest is addition.
</div>

The taxonomy doesn't replace OFS — it **explains** why OFS degrades. The
growth is mostly non-contradicting additions, not factual conflicts.

---

## Caveat · what hides inside the 90% "neutral"?

<div class="warn-box">
We did <b>not</b> measure factuality of neutral claims against the real world.<br>
"Absent from E₀" is not the same as "true in the world".
</div>

Claims labelled `neutral` can be three different things, in **Huang 2024** terms:

| Type of claim | Huang 2024 label | Example |
|----------------|------------------|---------|
| True in world, just not in E₀ | <i>not</i> a hallucination — extrinsic addition | "Shakespeare's mother was Mary Arden" |
| <span class="red">False in the world</span> | <span class="red">factuality hallucination · factual fabrication</span> | <span class="red">"Jefferson Forest spans 13,000 acres" (real: ~6,500)</span> |
| About an obscure entity, unverifiable | factuality hallucination · unverifiability | "X was born on August 15, 1959" |

<span class="note">Note: under <b>Maynez 2020</b>, all three already count as
extrinsic hallucination (none is grounded in the source). The distinction
matters only when applying the Huang factuality framework.</span>

---

## Numerical sensitivity of the caveat

Step 3 has ~1090 neutral claims (extrapolated). Even if **only 10%** were
false in the world, that means **~100 factuality hallucinations per step**
we did not count.

<div class="callout">
Our 1.2–1.5% intrinsic rate is a <b>strict lower bound</b>:<br><br>
<b>Maynez framework</b> (faithfulness vs source):<br>
&nbsp;&nbsp;lower bound = ~13–21% of facts per step (= 1 − OFS, all NOT_SUPPORTED counted)<br><br>
<b>Huang factuality framework</b> (vs world):<br>
&nbsp;&nbsp;lower bound = 1.2–1.5% (just our contradicts);<br>
&nbsp;&nbsp;true rate depends on how many neutral claims are actually false.<br><br>
World-truth verification (e.g. <b>SAFE</b> + Google Search, Wei et al. 2024)
would close the gap.
</div>

---

## Two coexisting frameworks for "hallucination"

| Framework | Reference | Hallucination = |
|-----------|-----------|-----------------|
| **Maynez 2020** | source (E₀) | intrinsic = contradicts source<br>extrinsic = not verifiable from source |
| **Huang 2024** | source AND/OR world | factuality = mismatch with real-world facts<br>faithfulness = mismatch with source / instruction / logic |

Where our intrinsic check fits in:

- Under **Maynez** → our `contradicts` = intrinsic; our `neutral` = extrinsic
- Under **Huang** → our `contradicts` ≈ *faithfulness · context inconsistency*;
  the `neutral` bucket is silent on world-truth (could be factuality
  hallucination or not — we cannot tell)

<div class="callout">
We measure faithfulness vs E₀ reliably. We do <b>not</b> measure factuality
vs the world.<br>
Both frameworks are useful; they answer different questions.
</div>

---

## Part 1 · what we learned

<div class="ok-box">
<b>Working interpretation (with explicit caveat).</b><br><br>
Under <b>Maynez 2020</b> (faithfulness vs source): <code>elaborate</code>
hallucinates more at every step (OFS: 13% → 21%), but the share that is
<b>direct contradiction</b> stays small (1.2–1.5% of all facts).<br>
Most growth is <b>non-grounded additions</b>, not contradictions.<br><br>
Under <b>Huang 2024</b> (factuality vs world): we cannot say yet — the 90%
neutral bucket may contain substantial <b>factuality hallucination</b> we
did not detect.
</div>

**Next steps for this thread:**

- Retest on **300q** with adequate sample size (much larger n)
- Run on **FictionalQA**, where parametric leak is structurally blocked
- Sample-based world-truth fact-check on `neutral` claims (SAFE-style)

---

# Part 2
# 15q ↔ 300q overlap

### Same questions, different rewriting behaviour
### Why model dtype matters

---

## The setup difference

| Run | Server | Model dtype |
|-----|--------|-------------|
| 15q (pilot) | Homer | <span class="green">bfloat16</span> |
| 300q       | Lisa   | <span class="red">4-bit NF4</span> |

Same model (OLMo-3.1-32B-Instruct), same prompts (OpenRewriteEval),
same instructions × 3 phrasings × 3 runs.

The **only thing that changed** between the two runs is the model dtype.
4-bit was chosen on Lisa because of the 24GB VRAM constraint on the 3090.

<div class="callout">
There are <b>10 qid in overlap</b> between 15q and 300q.<br>
On those 10 we can compare bf16 vs 4-bit on identical input — a natural experiment.
</div>

---

## Token count · `elaborate` stops elaborating in 4-bit

Mean tokens per step × instruction, **on the 10 overlap qid**:

**15q (bfloat16)** — `elaborate` *grows*:

| Instruction | Step 0 | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|--------|
| elaborate   | 2360   | 1499   | 1548   | <span class="green"><b>1621</b></span> |
| formality   | 2360   | 1316   | 1252   | 1216   |
| paraphrase  | 2360   |  890   |  832   |  796   |
| shorten     | 2360   |  574   |  478   |  439   |

**300q (4-bit NF4)** — `elaborate` *shrinks*:

| Instruction | Step 0 | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|--------|
| elaborate   | 2360   |  577   |  523   | <span class="red"><b>538</b></span> |
| formality   | 2360   |  871   |  705   |  609   |
| paraphrase  | 2360   |  492   |  384   |  335   |
| shorten     | 2360   |  432   |  332   |  285   |

In bf16, `elaborate` adds tokens (1499 → 1621). In 4-bit it removes them
(577 → 538) — even more compressed than `paraphrase`.

---

## Answer F1 · degradation is amplified in 4-bit

Mean F1 on the same 10 overlap qid, **all chains** (including F1=0 ones):

**15q (bfloat16)**

| Instruction | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|
| elaborate   | 0.283  | 0.332  | <span class="green">0.347</span> |
| formality   | 0.312  | 0.287  | 0.287  |
| paraphrase  | 0.287  | 0.326  | 0.348  |
| shorten     | 0.320  | 0.254  | 0.291  |

**300q (4-bit)**

| Instruction | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|
| elaborate   | 0.274  | 0.188  | <span class="red">0.181</span> |
| formality   | 0.311  | 0.265  | 0.280  |
| paraphrase  | 0.260  | 0.224  | 0.226  |
| shorten     | 0.253  | 0.234  | 0.226  |

Same input → F1 collapses faster in 4-bit because the rewriting removes
more text → more lost facts.

---

## OFS · groundedness comparison (overlap qid)

OFS init_score, mean per step × instruction, on the qid in overlap:

**15q (bfloat16)**

| Instruction | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|
| elaborate   | 0.908  | 0.897  | <span class="green">0.881</span> |
| formality   | 0.914  | 0.913  | 0.917  |
| paraphrase  | 0.921  | 0.904  | 0.919  |
| shorten     | 0.907  | 0.893  | 0.894  |

**300q (4-bit)**

| Instruction | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|
| elaborate   | 0.915  | 0.857  | <span class="red">0.900</span> |
| formality   | 0.917  | 0.896  | 0.886  |
| paraphrase  | 0.940  | 0.917  | 0.868  |
| shorten     | 0.913  | 0.906  | 0.887  |

<div class="warn-box">
<b>Small-sample caveat.</b> The 300q OFS run is still in progress
(55/179 qid completed). Overlap with 15q on OFS is only <b>5 qid</b>
(not 10). Numbers are noisy and trends should not be over-interpreted.
</div>

The differences are within noise for most cells. No clean
quantization-related OFS signal at this sample size.

---

## Mean token length · all 300q (179 qid)

Reference table for the full 300q run, mean tokens per step × instruction:

| Instruction | Step 0 | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|--------|
| elaborate   | 2434   | <span class="red">710</span>  | 647   | <span class="red">626</span>  |
| formality   | 2434   | 935    | 722   | 631    |
| paraphrase  | 2434   | 570    | 430   | 369    |
| shorten     | 2434   | 465    | 357   | 313    |

**Compression ratio vs E₀** (step3 / step0):

| Instruction | Ratio |
|-------------|-------|
| elaborate   | 0.26  |
| formality   | 0.26  |
| paraphrase  | 0.15  |
| shorten     | 0.13  |

<div class="warn-box">
On the full 300q (179 qid), <code>elaborate</code> compresses to <b>26%</b> of
E₀ — equal to <code>formality</code>, far more compressed than the 15q-bf16
baseline. The pattern matches the overlap-qid finding: <b>quantization is
making <code>elaborate</code> behave like a compressor</b>.
</div>

<span class="note">N = 537 chains per (instruction, step) cell. Std dev on
step 1 ranges 236 (shorten) to 497 (elaborate) — see full results CSV.</span>

---

## Cross-check · gpt-4o-mini does not compress

I ran a few `elaborate` rewriting prompts on **gpt-4o-mini** with the same
source texts.

<div class="ok-box">
<b>gpt-4o-mini does not compress aggressively.</b><br>
Output length stays close to or above E₀, as expected from the instruction.
</div>

This rules out:

- Prompt-template artifact (same prompts, different model → no compression)
- Generation-config bug (same code path used for gpt-4o-mini)

It points squarely at the **quantized OLMo** as the responsible factor.

---

## Part 2 · what we learned

<div class="warn-box">
A non-trivial portion of the "degradation trend" reported on 300q is
likely an <b>artifact of 4-bit quantization on the rewriting model</b>,
not intrinsic factual degradation from iterative rewriting.
</div>

Three converging pieces of evidence:

1. <span class="red"><code>elaborate</code></span> behavior <b>flips</b>: expand (bf16) → compress (4-bit), same qid
2. F1 drops more in 4-bit on the same qid
3. gpt-4o-mini does not reproduce the compression with the same prompts

**To close it.** Controlled run on 5–10 qid, bf16 vs 4-bit, same seed.
Then decide: re-run 300q in bf16 (slower, on Homer), or keep 4-bit and
quantify+disclose the bias.

---

# Part 3
# Two new datasets

### NewsQA · FictionalQA
### Pipelines ready, awaiting GPU

---

## Why two more datasets

MuSiQue alone has known confounds:

- **Multi-hop reasoning chain** — F1 drop can come from the chain breaking,
  not the rewriting itself
- **Parametric memory** — the QA model may know facts independently of E₀
  (King John, Henry III, etc.)

Two new datasets triangulate the rewriting effect:

| Dataset | Reasoning | Source | Memory leak |
|---------|-----------|--------|-------------|
| MuSiQue | Multi-hop (2/3/4) | Wiki + distractors | possible |
| **NewsQA** | Single-hop extractive | Real CNN articles | possible |
| **FictionalQA** | Single-hop | Synthetic events | <span class="green">blocked by design</span> |

<div class="callout">
Together they let us separate three confounds: <b>multi-hop reasoning</b>
(only MuSiQue), <b>extractive vs reasoning</b> (NewsQA vs MuSiQue), and
<b>parametric memory</b> (FictionalQA blocks it).
</div>

---

## NewsQA (Maluuba)

**Source.** `combined-newsqa-data-v1.csv` — CNN articles + crowdsourced questions.

**Selection.** One question per story, the *best validated answerable*:

- ≥2 crowdworkers agreed on the answer span
- Excluded `is_answer_absent` and `is_question_bad`
- All validated spans kept as aliases (`||`-joined) — max-F1 over the set
  replicates the official metric

**What it controls for.** Single-hop, extractive, real-news text.
Isolates the rewriting effect from the multi-hop reasoning noise of MuSiQue.

**Scripts** (`scripts/newsqa/`):
`rewriting_pipeline_newsqa.py` · `answer_f1_eval_newsqa.py` ·
`openfactscore_eval_newsqa.py`

<span class="note">Same I/O schema as MuSiQue pipeline, same instruction set
(OpenRewriteEval, Shu et al. 2023).</span>

---

## FictionalQA

**Source.** `jwkirchenbauer/fictionalqa` (HuggingFace) — synthetic events.
1500 documents (5 webtext styles × 300 events).

**Selection.** Per-doc best question with:

- `grade_blind == 0` — LLM **cannot** answer without the document
- `grade_informed == 1` — LLM **can** answer with the document
- duplicate filter; tie-break on shortest natural answer

**What it controls for.** The fact is **fictional** → the QA model cannot
answer from prior knowledge. Any F1 > 0 is a clean measurement of
fact-preservation through rewriting. **OFS drops here would be pure
hallucination signal**, independent of the QA model's memory.

**Scripts** (`scripts/fictionalqa/`):
`rewriting_pipeline_fictionalqa.py` · `answer_f1_eval_fictionalqa.py` ·
`openfactscore_eval_fictionalqa.py`

---

## Pipeline state

<div class="ok-box">
Both pipelines are <b>written, schema-coherent</b> with the MuSiQue setup,
and have been <b>validated end-to-end on gpt-4o-mini</b> as the rewriting
backbone.<br>
Smoke tests pass: rewriting generates, F1 evaluator returns sensible
numbers, OFS evaluator runs to completion.
</div>

What is shared across the three pipelines:

- Output CSV schema: `qid, question, gold_answer, gold_answer_aliases, group, instruction_type, run, instruction_used, step, text, n_tokens`
- Instruction set: 4 instruction types × 3 phrasings (OpenRewriteEval, Shu et al. 2023)
- Generation config: `temperature=0`, deterministic
- Evaluators reuse the same OFS / F1 logic

**What is missing.** The actual run with the target rewriting model
(OLMo-3.1-32B-Instruct, for consistency with MuSiQue).

<div class="warn-box">
<b>Bottleneck: GPU.</b> Homer/Lisa/Bart are currently busy with the 300q
runs. NewsQA + FictionalQA scheduled next, when a node frees up.
</div>

---

# Part 4
# Next steps

---

## Priority order

1. <span class="red"><b>Quantization control run</b></span><br>
   5–10 qid, bf16 vs 4-bit, same seed.<br>
   <i>Unblocks interpretation of all 300q numbers.</i>

2. <span class="blue"><b>Complete the elaborate hallucination taxonomy</b></span><br>
   ~1250 claims left, ~15 min, ~$0.30. Resume supported.

3. <span class="orange"><b>NewsQA + FictionalQA full runs</b></span><br>
   Pipelines ready, schemas coherent. Run as soon as GPU frees up.

4. <span class="green"><b>Sample-based world-truth check on neutral claims</b></span><br>
   SAFE-style, on 100–200 sampled neutral claims. Bridges the
   faithfulness/factuality gap.

---

## Open questions

- **300q dtype.** Re-run in bf16 on Homer (slower) or keep 4-bit and
  disclose the confound? Tradeoff: time vs interpretability.

- **NewsQA scope.** Article-length cutoff? NewsQA articles are typically
  much shorter than MuSiQue's 20-paragraph context — should we cap or pad?

- **FictionalQA styles.** Keep all 5 webtext styles, or focus on a single
  one (e.g. "news" for closest comparison with NewsQA) for the first run?

<br>

<div class="callout">
<b>Documents and outputs:</b><br>
<code>results/15q/exploratory_notes_and_next_steps.md</code> — full write-up<br>
<code>results/15q/elaborate_hallucination_taxonomy.csv</code> — judge outputs (2725/3981)<br>
<code>results/15q/elaborate_gpt_analysis.{csv,md}</code> — first-pass qualitative judge<br>
<code>scripts/15q/hallucination_taxonomy_eval.py</code> — pipeline script
</div>
