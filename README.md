# nsa_cybersecurity_metric_spring2022

**National Security Agency Cybersecurity Collaboration Center: An Effort to Create a Cybersecurity Risk Evaluation Metric**

Ryan Ripper — Data Science II: Applied Statistical Learning, Spring 2022, Georgetown University

Data wrangling and unsupervised machine learning over U.S. federal contract subawards, aimed at identifying Defense Industrial Base companies that are plausible cybersecurity targets.

## Background

[Hacking for Defense](https://www.h4d.us/) (H4D) is a Department of Defense–sponsored course at Georgetown University in which students work directly with the Defense and Intelligence Communities on mission-critical problems. The Spring 2022 cohort was tasked by the **NSA Cybersecurity Collaboration Center** with creating a standardized metric for the cybersecurity risk posed by defense contractors — a way to prioritize mitigation across a Defense Industrial Base of more than 100,000 companies, and to scale up the small cybersecurity-as-a-service pilots the NSA runs today.

This repository holds the data science half of that effort.

## Research question and pivot

The original goal was a supervised risk model. That turned out not to be buildable from the available data:

- There is no federal law requiring disclosure of data breaches, so no reliable labeled list of past cybersecurity attacks exists — and with no dependent variable, no supervised model.
- Even with such a list, the class distribution would be extreme, and oversampling the positive class would badly overfit.
- Enriching USAspending with third-party firmographics required joining on D-U-N-S numbers (the Dun & Bradstreet API attempt did not pan out) or on company name (naming conventions across sources are not consistent enough to trust).

So the project pivoted: rather than *predicting* attacks, use **unsupervised** methods to surface companies that stand out — by contract volume, by award size, or as statistical anomalies — as candidates for additional government cybersecurity support.

The analysis focuses on **hypersonics** as the emerging-technology slice, chosen because the technology is destabilizing (rapid nuclear delivery), because the U.S. is a lagging rather than leading player, and because subaward-level data reaches past the primes (Raytheon, Northrop Grumman, Lockheed Martin) to the wider supplier base. Hypersonic subawards are compared against all subawards throughout.

## Data

**[USAspending](https://www.usaspending.gov/)** — all federal contract **subawards** for calendar year 2021, the most recent full year at the time. Subawards rather than prime awards, deliberately: they expose the developers below the primes.

Hypersonic contracts are isolated by matching `hypersonic` in `prime_award_description`.

The dataset ships as `H4D_data.zip`, tracked with **Git LFS**.

## Analysis

Work proceeded in three passes, each with its own notebook plus an HTML render.

**Preliminary (`h4d_prelim_analysis_033022.ipynb`)** — a single-contract case study: the Raytheon Company hypersonic missile development award (prime award `HR001117C0025`, read from `Contract_HR001117C0025_Sub-Awards_1.csv` — a separate, narrower export, not the all-subawards file). Subcontracts within that prime award are grouped by subawardee name, description, and business type to compare average subaward amount and subaward count. Results in `results/*.csv`.

**Follow-up (`h4d_followup_analysis_040722.ipynb`)** — widened to all CY 2021 awards. Ranks the top subaward descriptions and the companies most frequently attached to them, for hypersonic contracts and for all contracts, with and without parent-company rollup. Results in `results/*.csv`.

**Machine learning (`h4d_ml_analysis_042622.ipynb`)** — the modeling pass:
- Missingness profiling with `missingno`. `subawardee_business_types` is the worst column, missing on 9,020 of 162,207 rows; across all 16 retained columns, 9.397% of observations have at least one NA — small enough to drop outright, taking the dataset to 146,964 rows
- **K-Means clustering** (k=5) fit on the full dummy-encoded hypersonic feature matrix, with the resulting labels plotted onto a subaward-count vs. average-subaward-amount scatter
- **t-SNE** dimensionality reduction to visualize the high-dimensional feature space, with K-Means labels mapped onto it
- **Anomaly detection** via an Isolation Forest (PyCaret), visualized as a 3D t-SNE outlier plot (`results/anomaly_plot.html`)

## Findings

The cybersecurity risk *metric* was not achieved — the data does not support it. What the analysis did produce is a shortlist of DIB companies that look like plausible targets, several of which independently showed weak IT posture:

- **Optical Sciences Corporation** — custom sensor test equipment and engineering support for the U.S. Army, NASA, and private aerospace/defense firms; Huntsville, AL. Tied for first (9 subawards, alongside Microtechnologies LLC) among companies attached to the top hypersonic subaward descriptions.
- **Bendix Commercial Vehicle Systems LLC** — surfaced across all contracts; during investigation the company's site was found to be serving without a valid SSL certificate for several days (April 11–12, 2022), leaving browser-to-server transmissions interceptable.
- **Honeywell International Inc.** and **Carahsoft Technology Corp.** — the two outliers in the hypersonic scatter, on average award amount and on subaward count respectively.
- **Lockheed Martin Corporation** — flagged by the anomaly detector, and unsurprising: its logistical position as one of the largest U.S. aerospace developers makes it a standing target.

The anomaly detector's hits correlated strongly with subaward *description* — matching anomalous descriptions back onto the full hypersonic set reproduced roughly the same company list, suggesting the descriptions themselves drove the detection and that further work is needed to establish a genuine link.

## Requirements

**Python 3.8 required.** Dependencies are pinned to the versions the analysis was originally run with (April 2022) — in particular `pycaret==2.3.5`, whose API (`setup(..., silent=True)`) was removed in PyCaret 3.x, so a newer stack will not run these notebooks as written.

```bash
pip install -r requirements.txt
```

`pycaret.anomaly` supplies the Isolation Forest used for anomaly detection; the rest is standard PyData plus `plotnine` (ggplot2 grammar) and `missingno` (missingness visualization). If you just want to read the results, every notebook has a pre-rendered `.html` next to it — no install needed.

## Setup

```bash
git clone https://github.com/ryanripper/h4d_spring2022.git
cd h4d_spring2022
git lfs pull          # H4D_data.zip is tracked with Git LFS
unzip H4D_data.zip
```

Data paths differ by notebook, so check before running:

| Notebook | Reads |
| --- | --- |
| `FINAL_PROJECT/Ryan_Ripper_Final_Project.ipynb` | `All_Contracts_Subawards_2022-04-12_H19M40S39_1.csv` from the working directory |
| `H4D_ANALYSIS/h4d_prelim_analysis/…` | `../data/Contract_HR001117C0025_Sub-Awards_1.csv` (the Raytheon prime award export) |
| `H4D_ANALYSIS/h4d_followup_analysis/…` | Both of the above — the Raytheon export first, then `../data/All_Subawards_2022-04-12_H19M40S42878092/All_Contracts_Subawards_2022-04-12_H19M40S39_1.csv` |
| `H4D_ANALYSIS/h4d_ml_analysis/…` | `../data/All_Subawards_2022-04-12_H19M40S42878092/All_Contracts_Subawards_2022-04-12_H19M40S39_1.csv` |

The `H4D_ANALYSIS` notebooks expect a sibling `data/` directory at the repository root; unzip `H4D_data.zip` there, or adjust the paths.

Then open any notebook with `jupyter lab` / `jupyter notebook`. Every notebook also has a pre-rendered `.html` next to it if you just want to read the output.

## Project structure

```
.
├── FINAL_PROJECT/
│   ├── Ryan_Ripper_Final_Project.ipynb        # The full write-up: intro → 8 tables, 5 figures → conclusion
│   ├── Ryan_Ripper_Final_Project.html         # Rendered version
│   ├── Ryan_Ripper_Final_Project_Plan.pdf/.docx
│   ├── Final_Project_Discussion.docx
│   └── NSA_Presentation/
│       ├── Ryan_Ripper_Final_Project.pptx     # Slide deck
│       └── Screenshots/                       # Supporting screenshots
├── H4D_ANALYSIS/
│   ├── h4d_prelim_analysis/                   # 3/30/22 — Raytheon hypersonic case study
│   ├── h4d_followup_analysis/                 # 4/7/22 — top descriptions & companies, all + hypersonic
│   ├── h4d_ml_analysis/                       # 4/26/22 — K-Means, t-SNE, Isolation Forest
│   │   └── results/anomaly_plot.html          # 3D t-SNE outlier plot
│   └── Hacking for Defense - 3-23-22.docx     # Meeting notes
├── H4D_data.zip                               # USAspending CY 2021 subawards (Git LFS)
├── requirements.txt
└── README.md
```

Each `H4D_ANALYSIS/*` subdirectory contains its notebook, an HTML render, and a `results/` folder. The prelim and follow-up `results/` folders hold exported CSVs; the ML one holds the anomaly plot instead.

## Tech stack

pandas, NumPy, scikit-learn (KMeans, TSNE), PyCaret (anomaly detection / Isolation Forest), plotnine, matplotlib, missingno, Jupyter.

## Author

Ryan Ripper — Georgetown University, Spring 2022
