# Beyond 1 Through 5: A Data-Driven Rework of Typical Basketball Player Archetypes

## DS 3001 Group Project

### Team
- Caroline Clippinger (mgu6bs)
- Malyeka Ali (sub6ap)
- Owen Kearney (uak4ux)
- Bryson Rios (cwb3ud)
- Olivia Sun ()

## Project Overview

The traditional basketball positions are point guard (1), shooting guard (2), small forward (3), power forward (4), and center (5). However, these positions are not as descriptive of how players actually perform on the court. This project uses data-driven methods to investigate whether modern NBA players naturally fit the traditional 1-5 positional framework or whether statistical archetypes provide a better representation.

## Research Question

Can NBA player performance data be used to identify meaningful player archetypes that differ from traditional basketball positions?

## Goals

1. Explore the relationship between traditional positions and measurable player statistics.
2. Identify groups of players with similar statistical profiles.
3. Compare data-driven clusters with conventional positions.
4. Evaluate whether the resulting archetypes provide a more useful description of modern basketball players.

## Planned Methodology

The target outcome of this project is to sort basketball players into groups based on specific features or profile indicators that better capture their play style. To achieve this, we will use either Latent Profile Analysis (LPA) or K-Means Clustering to identify groups of players with similar statistical profiles. LPA would allow us to examine each player’s probability of belonging to different groups, which may better capture players who are dynamic and capable of adopting multiple play styles. In contrast, K-Means Clustering assigns each player to a single cluster, making the results easier to interpret but potentially eliminating some of the nuance in player roles. The features used in the model will be based on four main playing metrics: ball dominance, playmaking, shooting profile, and rebounding/defensive play. These metrics will be represented through variables such as playtime, possession time, assists, turnovers, shot attempts and shot types, scoring breakdowns, rebounds, blocks, steals, and contested shots. By using these features to identify patterns among players, the model will allow us to determine whether data-driven player groups provide a more accurate representation of playing styles than traditional basketball positions.

## Key Assumptions

- In-game statistics are a valid representation of a player’s play style.
- Players with similar statistics generally have similar playing styles.
- The selected features, such as ball dominance, playmaking, shooting, and rebounding/defense, are useful indicators of playing style.
- The model will create meaningful groups of players that can be interpreted in real-world basketball terms.
- The groups created by the model will provide a useful way to compare players beyond traditional positions.

## Constraints

- This is a semester-long project, which limits the scope of the final product and the amount of time available for analysis.
- We are using multiple datasets, so we must ensure that values are matched to the same player and the same season across datasets.
- Differences in how statistics are recorded across datasets may limit which variables we can use.
- The available statistics may not capture every aspect of a player's playing style, such as coaching, team system, or role within a specific lineup.



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

## Datasets

Raw datasets should be placed in `data/` when permitted by the dataset's licensing and course requirements. Analysis code and notebooks should document preprocessing, feature selection, modeling choices, and results so that the analysis can be reproduced.


