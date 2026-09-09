# Beyond 1 Through 5: A Data-Driven Rework of Typical Basketball Player Archetypes

## DS 3001 Group Project

### Team
- Caroline Clippinger (mgu6bs)
- Malyeka Ali (sub6ap)
- Owen Kearney (uak4ux)
- Bryson Rios (Brios3)

## Project Overview

Traditional basketball positions—point guard, shooting guard, small forward, power forward, and center—are increasingly less descriptive of how players actually perform on the court. This project uses data-driven methods to investigate whether modern NBA players naturally fit the traditional 1-through-5 positional framework or whether statistical archetypes provide a better representation.

## Research Question

Can NBA player performance data be used to identify meaningful player archetypes that differ from traditional basketball positions?

## Goals

1. Explore the relationship between traditional positions and measurable player statistics.
2. Identify groups of players with similar statistical profiles.
3. Compare data-driven clusters with conventional positions.
4. Evaluate whether the resulting archetypes provide a more useful description of modern basketball players.

## Planned Methodology

The project will involve data cleaning and exploratory analysis, feature selection and standardization, dimensionality reduction/visualization where appropriate, and unsupervised classification. Candidate approaches include K-means clustering and Latent Profile Analysis (LPA). Cluster quality and interpretability will be evaluated rather than assuming that five clusters are automatically correct.

## Key Assumptions

- Player statistics are reasonable proxies for on-court roles and playing styles.
- The selected dataset is sufficiently representative of the players and seasons being studied.
- Relevant variables can be standardized so that differences in measurement scale do not dominate clustering.
- Player roles can be meaningfully summarized using combinations of observable statistical features.
- Traditional listed positions provide a useful baseline for comparison, but are not treated as ground truth.
- Players with limited playing time or missing data may need to be excluded or handled separately to avoid unstable statistical profiles.
- Cluster labels are descriptive rather than definitive classifications of players.

## Repository Structure

```text
DS3001-Project/
├── README.md
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── src/
│   └── README.md
├── analysis/
│   └── README.md
└── report/
    └── README.md
```

## Reproducibility

Raw datasets should be placed in `data/` when permitted by the dataset's licensing and course requirements. Analysis code and notebooks should document preprocessing, feature selection, modeling choices, and results so that the analysis can be reproduced.

## Status

Project repository initialized. Analysis, data, notebooks, and final report materials will be added as the project develops.
