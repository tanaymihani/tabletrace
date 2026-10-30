# Label QA against the MPEA dataset

The MPEA dataset (Borg et al. 2020, *Scientific Data* 7:430) compiles mechanical properties of
multi-principal element alloys from the literature. Commit `d31e2510cc` of
CitrineInformatics/MPEA_dataset. Matching is one-to-one on property and value within 1%.

| paper | labels | MPEA values (4 properties) | matched |
|---|---|---|---|
| chen2014 | 18 | 20 | 11 |
|  | not in the labels: ultimate_strength 1500 (Al0.25 Mo1 Nb1 Ti1 V1, test C) | | |
|  | not in the labels: ultimate_strength 1800 (Al0.5 Mo1 Nb1 Ti1 V1, test C) | | |
|  | not in the labels: elongation 11 (Al0.5 Mo1 Nb1 Ti1 V1, test C) | | |
|  | not in the labels: ultimate_strength 1300 (Al0.75 Mo1 Nb1 Ti1 V1, test C) | | |
|  | not in the labels: elongation 8 (Al0.75 Mo1 Nb1 Ti1 V1, test C) | | |
|  | not in the labels: ultimate_strength 1400 (Al1 Mo1 Nb1 Ti1 V1, test C) | | |
|  | not in the labels: elongation 3 (Al1 Mo1 Nb1 Ti1 V1, test C) | | |
|  | not in the labels: ultimate_strength 2050 (Mo1 Nb1 Ti1 V1, test C) | | |
|  | not in the labels: elongation 26 (Mo1 Nb1 Ti1 V1, test C) | | |
| yao2016 | 10 | 4 | 2 |
|  | not in the labels: ultimate_strength 2400 (Mo1 Nb1 Ta1 V1, test C) | | |
|  | not in the labels: elongation 21 (Mo1 Nb1 Ta1 V1, test C) | | |
| tseng2018 | 38 | 27 | 27 |
| zyka2019 | 67 | 26 | 26 |
| haas2019 | 30 | 30 | 30 |
| rickman2019 | 7 | 7 | 7 |
| sanchez2019 | 15 | 10 | 10 |

Totals: 113 of 124 MPEA values match a label.
