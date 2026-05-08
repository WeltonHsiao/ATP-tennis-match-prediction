# =========================================================
# 03_logistic_regression_models.R
# Logistic Regression Models for ATP Match Prediction
# =========================================================
#
# Purpose:
# Train and evaluate baseline and extended logistic regression
# models for ATP tennis match prediction.
#
# Models:
#   1. Baseline logistic regression:
#      - ranking-point difference only
#
#   2. Extended logistic regression:
#      - ranking-point difference
#      - seed difference
#      - tournament level
#      - playing surface
#
# Train period: 2000–2018
# Test period : 2019
#
# =========================================================

suppressPackageStartupMessages({
  library(tidymodels)
  library(dplyr)
  library(forcats)
  library(lubridate)
})

set.seed(123)

# ---------------------------------------------------------
# 0) Load cleaned data and create output folders
# ---------------------------------------------------------

df_clean <- readRDS("data/processed/df_clean.rds")

if (!dir.exists("results")) {
  dir.create("results", recursive = TRUE)
}

if (!dir.exists("figures")) {
  dir.create("figures", recursive = TRUE)
}

# ---------------------------------------------------------
# 1) Construct modelling dataset
# ---------------------------------------------------------

df_model <- df_clean %>%
  mutate(
    year = year(tourney_date),
    
    # Outcome: 1 if the higher-ranked player won
    y_num = as.integer(winner_rank < loser_rank),
    y = factor(y_num, levels = c(0, 1)),
    
    # Ranking points for higher- and lower-ranked players
    hi_pts = if_else(
      winner_rank < loser_rank,
      winner_rank_points,
      loser_rank_points,
      missing = NA_real_
    ),
    
    lo_pts = if_else(
      winner_rank < loser_rank,
      loser_rank_points,
      winner_rank_points,
      missing = NA_real_
    ),
    
    # Main predictor: ranking-point difference
    x = hi_pts - lo_pts,
    
    # Contextual predictors
    seed_diff = as.integer(
      coalesce(winner_seeded, 0L) - coalesce(loser_seeded, 0L)
    ),
    
    tourney_level = fct_other(
      factor(tourney_level),
      keep = c("G", "M", "A", "F", "D")
    ),
    
    surface = fct_other(
      factor(surface),
      keep = c("Hard", "Clay", "Grass")
    )
  ) %>%
  filter(
    !is.na(year),
    !is.na(y),
    !is.na(x),
    !is.na(surface),
    !is.na(tourney_level)
  )

# ---------------------------------------------------------
# 2) Chronological train-test split
# ---------------------------------------------------------

train <- df_model %>%
  filter(year >= 2000, year <= 2018)

test <- df_model %>%
  filter(year == 2019)

stopifnot(nrow(train) > 0, nrow(test) > 0)

# ---------------------------------------------------------
# 3) Model recipes
# ---------------------------------------------------------

# Baseline model: ranking-point difference only
rec_base <- recipe(y ~ x, data = train) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Extended model: ranking-point difference + contextual variables
rec_ext <- recipe(
  y ~ x + seed_diff + tourney_level + surface,
  data = train
) %>%
  step_unknown(tourney_level, surface) %>%
  step_other(tourney_level, surface, threshold = 0.005) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ---------------------------------------------------------
# 4) Logistic model specification
# ---------------------------------------------------------

logit_spec <- logistic_reg(mode = "classification") %>%
  set_engine("glm")

# ---------------------------------------------------------
# 5) Workflows and model fitting
# ---------------------------------------------------------

wf_base <- workflow() %>%
  add_model(logit_spec) %>%
  add_recipe(rec_base)

wf_ext <- workflow() %>%
  add_model(logit_spec) %>%
  add_recipe(rec_ext)

fit_base <- fit(wf_base, data = train)
fit_ext  <- fit(wf_ext,  data = train)

# ---------------------------------------------------------
# 6) Predictions on 2019 test set
# ---------------------------------------------------------

pred_base <- predict(fit_base, test, type = "prob") %>%
  bind_cols(predict(fit_base, test, type = "class")) %>%
  rename(
    .pred = .pred_1,
    .class = .pred_class
  )

pred_ext <- predict(fit_ext, test, type = "prob") %>%
  bind_cols(predict(fit_ext, test, type = "class")) %>%
  rename(
    .pred = .pred_1,
    .class = .pred_class
  )

eval_base <- bind_cols(test %>% select(y), pred_base)
eval_ext  <- bind_cols(test %>% select(y), pred_ext)

# Save model predictions for later model comparison
logistic_predictions <- test %>%
  select(year, y, x, seed_diff, tourney_level, surface) %>%
  mutate(
    pred_logit_base = eval_base$.pred,
    pred_logit_ext  = eval_ext$.pred
  )

write.csv(
  logistic_predictions,
  "results/logistic_predictions_2019.csv",
  row.names = FALSE
)

# ---------------------------------------------------------
# 7) Evaluation metrics
# ---------------------------------------------------------

eval_metrics <- function(df) {
  y_num <- as.integer(as.character(df$y))
  p <- pmin(pmax(df$.pred, 1e-15), 1 - 1e-15)
  
  acc <- mean(df$.class == df$y)
  ll <- -mean(y_num * log(p) + (1 - y_num) * log(1 - p))
  brier <- mean((p - y_num)^2)
  
  c(
    Accuracy = acc,
    LogLoss = ll,
    Brier = brier
  )
}

m_base <- eval_metrics(eval_base)
m_ext  <- eval_metrics(eval_ext)

logistic_results <- tibble(
  Model = c(
    "Baseline logistic regression (x only)",
    "Extended logistic regression (x + seed_diff + level + surface)"
  ),
  Accuracy = c(m_base["Accuracy"], m_ext["Accuracy"]),
  LogLoss = c(m_base["LogLoss"], m_ext["LogLoss"]),
  Brier = c(m_base["Brier"], m_ext["Brier"])
) %>%
  mutate(
    across(
      c(Accuracy, LogLoss, Brier),
      ~ round(.x, 4)
    )
  )

cat("\n=== Logistic regression comparison on 2019 test set ===\n")
print(logistic_results)

write.csv(
  logistic_results,
  "results/logistic_regression_results.csv",
  row.names = FALSE
)

# ---------------------------------------------------------
# 8) Calibration plot for extended logistic regression model
# ---------------------------------------------------------

calib_tbl <- eval_ext %>%
  mutate(bin = ntile(.pred, 10)) %>%
  group_by(bin) %>%
  summarise(
    n = n(),
    mean_pred = mean(.pred),
    obs_rate = mean(as.integer(as.character(y))),
    .groups = "drop"
  )

p_logit_calibration <- ggplot(calib_tbl, aes(mean_pred, obs_rate)) +
  geom_point(size = 2) +
  geom_line(alpha = 0.8) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = 2
  ) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "Calibration — Extended Logistic Regression",
    x = "Mean predicted probability",
    y = "Observed win rate"
  ) +
  theme_minimal(base_size = 12)

print(p_logit_calibration)

ggsave(
  filename = "figures/logit_calibration.png",
  plot = p_logit_calibration,
  width = 7,
  height = 5,
  dpi = 300
)

message("\nLogistic regression modelling completed.")
message("Results saved to results/logistic_regression_results.csv")
message("Predictions saved to results/logistic_predictions_2019.csv")
message("Calibration plot saved to figures/logit_calibration.png")