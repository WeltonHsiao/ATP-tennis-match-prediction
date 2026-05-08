# ATP Tennis Match Prediction

This repository contains R code for a Data Science research project on ATP men's tennis match prediction.

## Project Overview

The project compares several predictive approaches for pre-match ATP singles outcomes, including:

- Logistic regression
- Elo rating models
- Rolling-window logistic regression
- Lag-enhanced rolling logistic regression

The models are evaluated using ATP men's singles match data from 2000–2019.

## Data

Raw ATP match data can be downloaded from Jeff Sackmann's public repository:

https://github.com/JeffSackmann/tennis_atp

Place the annual CSV files inside:

data/raw/

## Repository Structure

data/
├── raw/
├── processed/

outputs/
├── figures/
├── tables/

scripts/
├── 01_load_clean_data.R
├── 02_exploratory_data_analysis.R
├── 03_logistic_regression_models.R
├── 04_elo_models.R
├── 05_rolling_window_models.R
├── 06_lag_enhanced_models.R

## Outputs 
- figures/: visualisations used in the report
- results/: model performance tables
- data/processed/: cleaned and processed datasets 

## Author

Welton Hsiao 
Master of Data Science 
Adelaide University
