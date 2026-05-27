# Reclassify statistics — OpenAI judge (2hop__14092_8311)

Single-qid analysis on 797 NOT_SUPPORTED claims; judge = gpt-4o-mini (temperature=0, 5-category prompt).

> **Caveat.** All tests below have the claim as the unit of analysis. Since we only have one qid, *do not* read these results as population-level claims about MuSiQue 2-hop questions. They describe the error structure within this single qid.

## 1. Marginal distribution of judge labels

| label         |   n |    % |
|:--------------|----:|-----:|
| SUPPORTED     | 296 | 37.1 |
| DISTORTED     | 208 | 26.1 |
| CONTRADICTION | 130 | 16.3 |
| UNVERIFIABLE  |  91 | 11.4 |
| INVENTED      |  72 |  9   |

- **Real rewrite errors** (CONTRADICTION + INVENTED + DISTORTED) = **410** (51.4% of the 797 NOT_SUPPORTED).
- **AFV false positives** (SUPPORTED) = **296** (37.1%).
- **Ambiguous / UNVERIFIABLE** = **91** (11.4%).

## 2. χ² independence — label × step

Contingency table (counts):

| label         |   1 |   2 |   3 |
|:--------------|----:|----:|----:|
| SUPPORTED     | 133 |  90 |  73 |
| DISTORTED     |  87 |  73 |  48 |
| CONTRADICTION |  43 |  45 |  42 |
| UNVERIFIABLE  |  26 |  31 |  34 |
| INVENTED      |  19 |  16 |  37 |

- χ² = **32.50**, dof = 8, p = **7.56e-05**
- Conclusion: the **mix of error categories shifts significantly across steps**.

## 3. χ² independence — label × instruction

Contingency table:

| label         |   elaborate |   shorten |   formality |   paraphrase |
|:--------------|------------:|----------:|------------:|-------------:|
| SUPPORTED     |          79 |        61 |          50 |          106 |
| DISTORTED     |          49 |        50 |          75 |           34 |
| CONTRADICTION |          45 |        24 |          30 |           31 |
| UNVERIFIABLE  |          50 |        14 |           7 |           20 |
| INVENTED      |          41 |         7 |          12 |           12 |

- χ² = **95.58**, dof = 12, p = **4.07e-15**
- Conclusion: **error categories are not uniformly distributed across instructions**. Standardised residuals reveal which cells deviate the most.

Pearson standardised residuals (|z| > 2 ⇒ over/under-represented):

| label         |   elaborate |   shorten |   formality |   paraphrase |
|:--------------|------------:|----------:|------------:|-------------:|
| SUPPORTED     |       -1.92 |      0.4  |       -1.82 |         3.53 |
| DISTORTED     |       -2.4  |      1.46 |        4.39 |        -2.61 |
| CONTRADICTION |        0.3  |     -0.29 |        0.3  |        -0.37 |
| UNVERIFIABLE  |        3.62 |     -0.9  |       -2.89 |        -0.66 |
| INVENTED      |        3.51 |     -1.89 |       -0.94 |        -1.48 |

## 4. elaborate vs others on INVENTED — Fisher exact

|           |   not_INVENTED |   INVENTED |
|:----------|---------------:|-----------:|
| other     |            502 |         31 |
| elaborate |            223 |         41 |

- P(INVENTED | elaborate)   = 41/264 = **15.5%**
- P(INVENTED | not elaborate) = 31/533 = **5.8%**
- Fisher exact: odds ratio = **2.98**, p = **1.81e-05**
- Conclusion: elaborate **produces INVENTED claims at a different rate** than other instructions.

## 5. Cochran-Armitage trend — INVENTED proportion across steps 1→2→3

|   step |   n |   n_inv |   p_inv |
|-------:|----:|--------:|--------:|
|      1 | 308 |      19 |   0.062 |
|      2 | 255 |      16 |   0.063 |
|      3 | 234 |      37 |   0.158 |

- Cochran-Armitage z = **3.72**, p = **0.000197** (positive z ⇒ INVENTED rate grows with step)

## 6. AFV false-positive bucket (SUPPORTED) vs instruction

| instruction_type   |   not_SUPPORTED |   SUPPORTED |   P(SUPPORTED) |
|:-------------------|----------------:|------------:|---------------:|
| elaborate          |             185 |          79 |          0.299 |
| shorten            |              95 |          61 |          0.391 |
| formality          |             124 |          50 |          0.287 |
| paraphrase         |              97 |         106 |          0.522 |

- χ² = **31.17**, dof = 3, p = **7.81e-07**
- Conclusion: **the AFV's false-positive rate is not uniform across instructions** — Gemma-3-4B mislabels paraphrase-style rewrites more often than others.

## 7. Bootstrap 95% CI on the headline numbers

- **Real-error rate within NOT_SUPPORTED** = 51.4%  (95% CI [47.9, 54.8])
- **AFV false-positive rate within NOT_SUPPORTED** = 37.1%  (95% CI [33.8, 40.5])
- **Raw OFS** (qid)          = 0.757  (2489/3286)
- **Calibrated OFS** (qid)   = **0.848**  (95% CI [0.839, 0.856])
- Δ = +9.0 percentage points of OFS underestimation by the Gemma AFV on this qid.
