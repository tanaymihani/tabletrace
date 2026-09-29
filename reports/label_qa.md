# Label QA against LitXAlloy

LitXAlloy (LitXBench, Radical AI, 2026) curates measurements from the same journal papers,
including values reported in the text. Commit `ac93f65005`. Matching is one-to-one on
property and value within 1%, because the two label sets name materials differently
(printed row label here, composition plus processing history there).

| paper | our labels | LitXAlloy values (4 properties) | matched | LitXAlloy table-sourced values matched |
|---|---|---|---|---|
| chen2014 | 18 | 18 | 18 | 17/17 |
| yao2016 | 10 | 2 | 1 | 0/0 |
|  | not in our labels: elongation ~21 percent (MoNbTaV; source: The alloy exhibits about 21% compression strain before fract) | | | |
| tseng2018 | 38 | 27 | 27 | 27/27 |
| zyka2019 | 67 | 19 | 19 | 0/0 |
| haas2019 | 30 | 20 | 20 | 20/20 |
| rickman2019 | 7 | 7 | 7 | 0/0 |
| sanchez2019 | 15 | 15 | 10 | 10/10 |
|  | not in our labels: vickers_hardness 235 HV (Al65Cu5Fe5Mg5Si15Zn5; source: n/a) | | | |
|  | not in our labels: vickers_hardness 260 HV (Al65Cu5Mg5Ni5Si15Zn5; source: n/a) | | | |
|  | not in our labels: vickers_hardness 200 HV (Al70Cr5Cu5Mg5Si10Zn5; source: n/a) | | | |
|  | not in our labels: vickers_hardness 264 HV (Al70Cu5Mg5Mn5Si10Zn5; source: n/a) | | | |
|  | not in our labels: vickers_hardness 220 HV (Al70Cu5Mg5Si10Zn5Zr5; source: n/a) | | | |

Totals: 185 labels here, 108 LitXAlloy values, 102 matched; 74 of 74 LitXAlloy values whose source cites a table are in our labels.
