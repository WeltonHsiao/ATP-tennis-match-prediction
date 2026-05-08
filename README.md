# ATP Tennis Match Prediction

This repository contains R code for a Data Science Research Project on pre-match ATP tennis outcome prediction.

## Project

The project compares logistic regression, Elo rating models, rolling-window logistic regression, and lag-enhanced rolling logistic regression models using ATP men's singles match data from 2000–2019.

## Data

Raw ATP match data can be downloaded from Jeff Sackmann's tennis_atp repository:

https://github.com/JeffSackmann/tennis_atp

Place the annual CSV files in:

data/raw/

## Scripts

Run the scripts in order:

1. scripts/01_load_clean_data.R
2. scripts/02_exploratory_data_analysis.R
3. scripts/03_logistic_regression_models.R
4. scripts/04_elo_models.R
5. scripts/05_rolling_window_models.R
6. scripts/06_lag_enhanced_models.R

## Outputs

- figures/: visualisations used in the report
- results/: model performance tables
- data/processed/: cleaned and processed datasets

## Author

Welton Hsiao  
Master of Data Science  
Adeliade University
