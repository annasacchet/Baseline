# 300q - inference summary

Random structure: `(1|qid)`. Step contrasts: planned, Holm-adjusted.


## bs_base

| family   | test                    | contrast          |   n_pairs |   mean_diff |   effect_size |       p_raw |   p_holm |
|:---------|:------------------------|:------------------|----------:|------------:|--------------:|------------:|---------:|
| bs_base  | Friedman over runs      | run (within qid)  |       nan |         nan |        0.0463 | 0.000313828 |      nan |
| bs_base  | Friedman (step omnibus) | step in [1, 2, 3] |        15 |         nan |      nan      | 3.05902e-07 |      nan |



## bs_baseline

| family      | test                                    | contrast     |   n_pairs |   mean_diff |   effect_size |       p_raw |     p_holm |
|:------------|:----------------------------------------|:-------------|----------:|------------:|--------------:|------------:|-----------:|
| bs_baseline | Wilcoxon paired (planned step contrast) | step1->step2 |        15 |     -0.0155 |             1 | 6.10352e-05 | 0.00012207 |
| bs_baseline | Wilcoxon paired (planned step contrast) | step2->step3 |        15 |     -0.0071 |             1 | 6.10352e-05 | 0.00012207 |



## bs_cons

| family   | test                    | contrast          |   n_pairs |   mean_diff |   effect_size |       p_raw |   p_holm |
|:---------|:------------------------|:------------------|----------:|------------:|--------------:|------------:|---------:|
| bs_cons  | Friedman over runs      | run (within qid)  |       nan |         nan |        0.0303 | 0.00769899  |      nan |
| bs_cons  | Friedman (step omnibus) | step in [1, 2, 3] |        15 |         nan |      nan      | 3.05902e-07 |      nan |



## bs_consecutive

| family         | test                                    | contrast     |   n_pairs |   mean_diff |   effect_size |       p_raw |     p_holm |
|:---------------|:----------------------------------------|:-------------|----------:|------------:|--------------:|------------:|-----------:|
| bs_consecutive | Wilcoxon paired (planned step contrast) | step1->step2 |        15 |      0.0569 |             1 | 6.10352e-05 | 0.00012207 |
| bs_consecutive | Wilcoxon paired (planned step contrast) | step2->step3 |        15 |      0.0168 |             1 | 6.10352e-05 | 0.00012207 |



## f1

| family   | test                                    | contrast             |   n_pairs |   mean_diff |   effect_size |    p_raw |     p_holm |
|:---------|:----------------------------------------|:---------------------|----------:|------------:|--------------:|---------:|-----------:|
| f1       | Wilcoxon paired (planned step contrast) | step0->step1         |        15 |      0      |        0.35   | 0.204357 |   0.613071 |
| f1       | Wilcoxon paired (planned step contrast) | step1->step2         |        15 |     -0.0056 |        0      | 1        |   1        |
| f1       | Wilcoxon paired (planned step contrast) | step2->step3         |        15 |     -0.0111 |        0.125  | 0.637352 |   1        |
| f1       | Friedman over runs                      | run (within qid)     |       nan |    nan      |        0.0103 | 0.263597 | nan        |
| f1       | Friedman (step omnibus)                 | step in [0, 1, 2, 3] |        15 |    nan      |      nan      | 0.937149 | nan        |



## hop-bs_base

| family      | test                                    | contrast   |   n_pairs |   mean_diff |   effect_size |   p_raw |   p_holm |
|:------------|:----------------------------------------|:-----------|----------:|------------:|--------------:|--------:|---------:|
| hop-bs_base | Wilcoxon step_min vs step_max (per hop) | 2-hop      |         5 |     -0.0214 |           nan |  0.0625 |   0.1875 |
| hop-bs_base | Wilcoxon step_min vs step_max (per hop) | 3-hop      |         5 |     -0.0208 |           nan |  0.0625 |   0.1875 |
| hop-bs_base | Wilcoxon step_min vs step_max (per hop) | 4-hop      |         5 |     -0.0254 |           nan |  0.0625 |   0.1875 |



## hop-bs_cons

| family      | test                                    | contrast   |   n_pairs |   mean_diff |   effect_size |   p_raw |   p_holm |
|:------------|:----------------------------------------|:-----------|----------:|------------:|--------------:|--------:|---------:|
| hop-bs_cons | Wilcoxon step_min vs step_max (per hop) | 2-hop      |         5 |      0.0693 |           nan |  0.0625 |   0.1875 |
| hop-bs_cons | Wilcoxon step_min vs step_max (per hop) | 3-hop      |         5 |      0.0843 |           nan |  0.0625 |   0.1875 |
| hop-bs_cons | Wilcoxon step_min vs step_max (per hop) | 4-hop      |         5 |      0.0676 |           nan |  0.0625 |   0.1875 |



## hop-f1

| family   | test                                    | contrast   |   n_pairs |   mean_diff |   effect_size |   p_raw |   p_holm |
|:---------|:----------------------------------------|:-----------|----------:|------------:|--------------:|--------:|---------:|
| hop-f1   | Wilcoxon step_min vs step_max (per hop) | 2-hop      |         5 |       -0    |           nan |   1     |        1 |
| hop-f1   | Wilcoxon step_min vs step_max (per hop) | 3-hop      |         5 |       -0.05 |           nan |   0.875 |        1 |
| hop-f1   | Wilcoxon step_min vs step_max (per hop) | 4-hop      |         5 |       -0    |           nan |   0.75  |        1 |



## hop-ofs

| family   | test                                    | contrast   |   n_pairs |   mean_diff |   effect_size |   p_raw |   p_holm |
|:---------|:----------------------------------------|:-----------|----------:|------------:|--------------:|--------:|---------:|
| hop-ofs  | Wilcoxon step_min vs step_max (per hop) | 2-hop      |         5 |     -0.01   |           nan |  0.625  |   0.625  |
| hop-ofs  | Wilcoxon step_min vs step_max (per hop) | 3-hop      |         5 |     -0.0188 |           nan |  0.0625 |   0.1875 |
| hop-ofs  | Wilcoxon step_min vs step_max (per hop) | 4-hop      |         5 |     -0.0258 |           nan |  0.0625 |   0.1875 |



## metric-correlation

| family             | test    | contrast                     |   n_pairs |   mean_diff |   effect_size |     p_raw |   p_holm |
|:-------------------|:--------|:-----------------------------|----------:|------------:|--------------:|----------:|---------:|
| metric-correlation | rm_corr | answer_f1 ~ factscore        |       nan |         nan |         0.009 | 0.83171   |      nan |
| metric-correlation | rm_corr | answer_f1 ~ bert_f1_baseline |       nan |         nan |         0.112 | 0.0101581 |      nan |
| metric-correlation | rm_corr | factscore ~ bert_f1_baseline |       nan |         nan |        -0.029 | 0.513442  |      nan |



## ofs

| family   | test                                    | contrast          |   n_pairs |   mean_diff |   effect_size |      p_raw |      p_holm |
|:---------|:----------------------------------------|:------------------|----------:|------------:|--------------:|-----------:|------------:|
| ofs      | Wilcoxon paired (planned step contrast) | step1->step2      |        15 |     -0.0083 |        0.5    | 0.0946045  |   0.0946045 |
| ofs      | Wilcoxon paired (planned step contrast) | step2->step3      |        15 |     -0.0099 |        0.617  | 0.0353394  |   0.0706787 |
| ofs      | Friedman over runs                      | run (within qid)  |       nan |    nan      |        0.0348 | 0.126607   | nan         |
| ofs      | Friedman (step omnibus)                 | step in [1, 2, 3] |        15 |    nan      |      nan      | 0.00127263 | nan         |

