# 300q - inference summary

Random structure: `(1|qid)`. Step contrasts: planned, Holm-adjusted.


## bs_base

| family   | test                    | contrast          |   n_pairs |   mean_diff |   effect_size |        p_raw |   p_holm |
|:---------|:------------------------|:------------------|----------:|------------:|--------------:|-------------:|---------:|
| bs_base  | Friedman over runs      | run (within qid)  |       nan |         nan |        0.0559 | 7.71473e-09  |      nan |
| bs_base  | Friedman (step omnibus) | step in [1, 2, 3] |       297 |         nan |      nan      | 8.59741e-106 |      nan |



## bs_baseline

| family      | test                                    | contrast     |   n_pairs |   mean_diff |   effect_size |       p_raw |      p_holm |
|:------------|:----------------------------------------|:-------------|----------:|------------:|--------------:|------------:|------------:|
| bs_baseline | Wilcoxon paired (planned step contrast) | step1->step2 |       297 |     -0.0084 |         0.988 | 2.55626e-49 | 5.11252e-49 |
| bs_baseline | Wilcoxon paired (planned step contrast) | step2->step3 |       297 |     -0.0042 |         0.959 | 1.54124e-46 | 1.54124e-46 |



## bs_cons

| family   | test                    | contrast          |   n_pairs |   mean_diff |   effect_size |        p_raw |   p_holm |
|:---------|:------------------------|:------------------|----------:|------------:|--------------:|-------------:|---------:|
| bs_cons  | Friedman over runs      | run (within qid)  |       nan |         nan |        0.0222 | 0.00735439   |      nan |
| bs_cons  | Friedman (step omnibus) | step in [1, 2, 3] |       297 |         nan |      nan      | 3.88069e-123 |      nan |



## bs_consecutive

| family         | test                                    | contrast     |   n_pairs |   mean_diff |   effect_size |       p_raw |      p_holm |
|:---------------|:----------------------------------------|:-------------|----------:|------------:|--------------:|------------:|------------:|
| bs_consecutive | Wilcoxon paired (planned step contrast) | step1->step2 |       297 |      0.0846 |          1    | 1.88322e-50 | 3.76643e-50 |
| bs_consecutive | Wilcoxon paired (planned step contrast) | step2->step3 |       297 |      0.0146 |          0.97 | 1.50638e-47 | 1.50638e-47 |



## f1

| family   | test                                    | contrast             |   n_pairs |   mean_diff |   effect_size |       p_raw |        p_holm |
|:---------|:----------------------------------------|:---------------------|----------:|------------:|--------------:|------------:|--------------:|
| f1       | Wilcoxon paired (planned step contrast) | step0->step1         |       297 |     -0.1686 |        0.377  | 3.62479e-09 |   1.08744e-08 |
| f1       | Wilcoxon paired (planned step contrast) | step1->step2         |       297 |     -0.0297 |        0.019  | 0.758221    |   0.929557    |
| f1       | Wilcoxon paired (planned step contrast) | step2->step3         |       297 |     -0.0171 |        0.044  | 0.464778    |   0.929557    |
| f1       | Friedman over runs                      | run (within qid)     |       nan |    nan      |        0.0307 | 0.028403    | nan           |
| f1       | Friedman (step omnibus)                 | step in [0, 1, 2, 3] |       297 |    nan      |      nan      | 1.51637e-09 | nan           |



## hop-bs_base

| family      | test                                    | contrast   |   n_pairs |   mean_diff |   effect_size |       p_raw |      p_holm |
|:------------|:----------------------------------------|:-----------|----------:|------------:|--------------:|------------:|------------:|
| hop-bs_base | Wilcoxon step_min vs step_max (per hop) | 2-hop      |       100 |     -0.0138 |           nan | 6.31161e-18 | 1.89348e-17 |
| hop-bs_base | Wilcoxon step_min vs step_max (per hop) | 3-hop      |        97 |     -0.0126 |           nan | 1.29646e-17 | 2.59293e-17 |
| hop-bs_base | Wilcoxon step_min vs step_max (per hop) | 4-hop      |       100 |     -0.0113 |           nan | 2.01956e-17 | 2.59293e-17 |



## hop-bs_cons

| family      | test                                    | contrast   |   n_pairs |   mean_diff |   effect_size |       p_raw |      p_holm |
|:------------|:----------------------------------------|:-----------|----------:|------------:|--------------:|------------:|------------:|
| hop-bs_cons | Wilcoxon step_min vs step_max (per hop) | 2-hop      |       100 |      0.0961 |           nan | 3.89656e-18 | 1.16897e-17 |
| hop-bs_cons | Wilcoxon step_min vs step_max (per hop) | 3-hop      |        97 |      0.0967 |           nan | 1.21812e-17 | 1.21812e-17 |
| hop-bs_cons | Wilcoxon step_min vs step_max (per hop) | 4-hop      |       100 |      0.1045 |           nan | 3.89656e-18 | 1.16897e-17 |



## hop-f1

| family   | test                                    | contrast   |   n_pairs |   mean_diff |   effect_size |       p_raw |      p_holm |
|:---------|:----------------------------------------|:-----------|----------:|------------:|--------------:|------------:|------------:|
| hop-f1   | Wilcoxon step_min vs step_max (per hop) | 2-hop      |       100 |     -0.2333 |           nan | 4.97757e-05 | 0.000149327 |
| hop-f1   | Wilcoxon step_min vs step_max (per hop) | 3-hop      |        97 |     -0.232  |           nan | 6.98732e-05 | 0.000149327 |
| hop-f1   | Wilcoxon step_min vs step_max (per hop) | 4-hop      |       100 |     -0.1817 |           nan | 0.00793537  | 0.00793537  |



## metric-correlation

| family             | test    | contrast                     |   n_pairs |   mean_diff |   effect_size |       p_raw |   p_holm |
|:-------------------|:--------|:-----------------------------|----------:|------------:|--------------:|------------:|---------:|
| metric-correlation | rm_corr | answer_f1 ~ factscore        |       nan |         nan |         0.094 | 4.25446e-05 |      nan |
| metric-correlation | rm_corr | answer_f1 ~ bert_f1_baseline |       nan |         nan |         0.193 | 2.26993e-17 |      nan |
| metric-correlation | rm_corr | factscore ~ bert_f1_baseline |       nan |         nan |         0.252 | 7.11401e-29 |      nan |



## ofs

| family   | test                                    | contrast          |   n_pairs |   mean_diff |   effect_size |       p_raw |        p_holm |
|:---------|:----------------------------------------|:------------------|----------:|------------:|--------------:|------------:|--------------:|
| ofs      | Wilcoxon paired (planned step contrast) | step1->step2      |        55 |     -0.0166 |        0.794  | 3.06683e-07 |   6.13366e-07 |
| ofs      | Wilcoxon paired (planned step contrast) | step2->step3      |        55 |     -0.0131 |        0.658  | 2.15761e-05 |   2.15761e-05 |
| ofs      | Friedman over runs                      | run (within qid)  |       nan |    nan      |        0.0543 | 0.104921    | nan           |
| ofs      | Friedman (step omnibus)                 | step in [1, 2, 3] |        55 |    nan      |      nan      | 3.75065e-12 | nan           |

