# =========================================================
# 04_elo_models.R
# Elo Rating Models for ATP Match Prediction
# =========================================================
#
# Purpose:
# Implement a baseline Elo rating model for ATP tennis match
# prediction. The model updates player ratings sequentially
# using matches from 2000–2018 and evaluates predictions on
# the 2019 test season.
#
# Model:
#   - Single-pool Elo
#   - Fixed K-factor
#   - Default initial rating = 1500
#
# Evaluation:
#   - Accuracy
#   - Log-loss
#   - Brier score
#   - Calibration plot
#
# =========================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(forcats)
  library(broom)
  library(ggplot2)
})

# ---------------------------------------------------------
# 0) Configuration
# ---------------------------------------------------------

K_BASE <- 32
DEFAULT_RATING <- 1500

df_clean <- readRDS("data/processed/df_clean.rds")

if (!dir.exists("results")) {
  dir.create("results", recursive = TRUE)
}

if (!dir.exists("figures")) {
  dir.create("figures", recursive = TRUE)
}

# ---------------------------------------------------------
# 1) Build sequential Elo dataset
# ---------------------------------------------------------

df_seq_base <- df_clean %>%
  transmute(
    tourney_date = as.Date(tourney_date),
    year = year(tourney_date),
    
    winner_id,
    loser_id,
    
    winner_rank = as.integer(winner_rank),
    loser_rank = as.integer(loser_rank)
  ) %>%
  filter(
    !is.na(tourney_date),
    !is.na(winner_id),
    !is.na(loser_id),
    !is.na(winner_rank),
    !is.na(loser_rank)
  ) %>%
  arrange(tourney_date)

train_b <- df_seq_base %>%
  filter(year >= 2000, year <= 2018)

test_b <- df_seq_base %>%
  filter(year == 2019)

stopifnot(nrow(train_b) > 0, nrow(test_b) > 0)

# ---------------------------------------------------------
# 2) Elo rating storage
# ---------------------------------------------------------

elo_pool <- new.env(parent = emptyenv())

get_rating <- function(player_id) {
  key <- as.character(player_id)
  
  if (!exists(key, envir = elo_pool, inherits = FALSE)) {
    assign(key, DEFAULT_RATING, envir = elo_pool)
  }
  
  get(key, envir = elo_pool, inherits = FALSE)
}

set_rating <- function(player_id, value) {
  assign(
    as.character(player_id),
    as.numeric(value),
    envir = elo_pool
  )
}

# Expected win probability from Elo ratings
elo_expected_prob <- function(rating_a, rating_b) {
  1 / (1 + 10 ^ ((rating_b - rating_a) / 400))
}

# ---------------------------------------------------------
# 3) Sequential Elo processor
# ---------------------------------------------------------

process_elo_block <- function(data, update = TRUE, K = K_BASE) {
  out <- vector("list", nrow(data))
  
  for (i in seq_len(nrow(data))) {
    row <- data[i, ]
    
    # Define player i as the higher-ranked player
    if (row$winner_rank < row$loser_rank) {
      id_i <- row$winner_id
      id_j <- row$loser_id
      y_i <- 1L
    } else {
      id_i <- row$loser_id
      id_j <- row$winner_id
      y_i <- 0L
    }
    
    # Pre-match Elo ratings
    rating_i <- get_rating(id_i)
    rating_j <- get_rating(id_j)
    
    p_i <- elo_expected_prob(rating_i, rating_j)
    
    out[[i]] <- tibble(
      tourney_date = row$tourney_date,
      year = row$year,
      
      id_i = id_i,
      id_j = id_j,
      
      elo_i_pre = rating_i,
      elo_j_pre = rating_j,
      
      p_elo = p_i,
      y = y_i
    )
    
    # Update ratings only when requested
    if (update) {
      score_i <- y_i
      score_j <- 1 - score_i
      
      set_rating(id_i, rating_i + K * (score_i - p_i))
      set_rating(id_j, rating_j + K * (score_j - (1 - p_i)))
    }
  }
  
  bind_rows(out)
}

# ---------------------------------------------------------
# 4) Train baseline Elo model on 2000–2018
# ---------------------------------------------------------

elo_pool <- new.env(parent = emptyenv())  # reset ratings

elo_train_baseline <- process_elo_block(
  data = train_b,
  update = TRUE,
  K = K_BASE
)

# ---------------------------------------------------------
# 5) Predict 2019 matches without rating updates
# ---------------------------------------------------------

elo_test_baseline <- process_elo_block(
  data = test_b,
  update = FALSE,
  K = K_BASE
)

# ---------------------------------------------------------
# 6) Evaluate baseline Elo model on 2019
# ---------------------------------------------------------

pred <- elo_test_baseline$p_elo
y <- elo_test_baseline$y

eps <- 1e-15
p <- pmin(pmax(pred, eps), 1 - eps)

elo_acc <- mean((p >= 0.5) == as.logical(y))
elo_logloss <- -mean(y * log(p) + (1 - y) * log(1 - p))
elo_brier <- mean((p - y)^2)

# Rule baseline: always predict higher-ranked player wins
rule_acc <- mean(y == 1)

baseline_elo_results <- tibble(
  Model = c(
    "Rule: always higher-ranked",
    "Baseline Elo (K = 32)"
  ),
  Accuracy = c(rule_acc, elo_acc),
  LogLoss = c(NA_real_, elo_logloss),
  Brier = c(NA_real_, elo_brier)
) %>%
  mutate(
    across(
      c(Accuracy, LogLoss, Brier),
      ~ round(.x, 4)
    )
  )

cat("\n=== Baseline Elo model on 2019 test set ===\n")
print(baseline_elo_results)

write.csv(
  baseline_elo_results,
  "results/baseline_elo_results.csv",
  row.names = FALSE
)

# Save Elo predictions for later comparison
write.csv(
  elo_test_baseline,
  "results/baseline_elo_predictions_2019.csv",
  row.names = FALSE
)

# ---------------------------------------------------------
# 7) Calibration plot for baseline Elo model
# ---------------------------------------------------------

calib_elo <- elo_test_baseline %>%
  mutate(bin = ntile(p_elo, 10)) %>%
  group_by(bin) %>%
  summarise(
    n = n(),
    mean_pred = mean(p_elo),
    obs_rate = mean(y),
    .groups = "drop"
  )

p_elo_calibration <- ggplot(calib_elo, aes(mean_pred, obs_rate)) +
  geom_point(size = 2) +
  geom_line(alpha = 0.8) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = 2
  ) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "Calibration — Baseline Elo",
    x = "Mean predicted probability",
    y = "Observed win rate"
  ) +
  theme_minimal(base_size = 13)

print(p_elo_calibration)

ggsave(
  filename = "figures/elo_calibration.png",
  plot = p_elo_calibration,
  width = 7,
  height = 5,
  dpi = 300
)

# ---------------------------------------------------------
# 8) Yearly training-period accuracy
# ---------------------------------------------------------

elo_yearly_accuracy <- elo_train_baseline %>%
  group_by(year) %>%
  summarise(
    accuracy = mean((p_elo >= 0.5) == y),
    n_matches = n(),
    .groups = "drop"
  )

p_elo_yearly_accuracy <- ggplot(elo_yearly_accuracy, aes(year, accuracy)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  labs(
    title = "Baseline Elo Accuracy by Year",
    x = "Year",
    y = "Accuracy"
  ) +
  theme_minimal(base_size = 13)

print(p_elo_yearly_accuracy)

ggsave(
  filename = "figures/elo_yearly_accuracy.png",
  plot = p_elo_yearly_accuracy,
  width = 7,
  height = 5,
  dpi = 300
)

write.csv(
  elo_yearly_accuracy,
  "results/elo_yearly_accuracy.csv",
  row.names = FALSE
)

message("\nBaseline Elo modelling completed.")
message("Results saved to results/baseline_elo_results.csv")
message("Predictions saved to results/baseline_elo_predictions_2019.csv")
message("Calibration plot saved to figures/elo_calibration.png")

# ---------------------------------------------------------
# 9) Contextual Elo model preparation
# ---------------------------------------------------------
#
# This section extends the single-pool Elo model by using
# pre-match Elo rating differences together with contextual
# variables in a logistic calibration model.
#
# Elo ratings are still updated only from match outcomes.
# Contextual variables are used only in the calibration stage.
# ---------------------------------------------------------

# Rebuild sequential frame with contextual variables
df_seq_context <- df_clean %>%
  transmute(
    tourney_date = as.Date(tourney_date),
    year = year(tourney_date),
    
    level = fct_other(
      factor(tourney_level),
      keep = c("G", "M", "A", "F", "D")
    ),
    
    surface = fct_other(
      factor(surface),
      keep = c("Hard", "Clay", "Grass")
    ),
    
    winner_seed = as.integer(coalesce(winner_seeded, 0L)),
    loser_seed  = as.integer(coalesce(loser_seeded, 0L)),
    
    winner_id,
    loser_id,
    
    winner_rank = as.integer(winner_rank),
    loser_rank  = as.integer(loser_rank)
  ) %>%
  filter(
    !is.na(tourney_date),
    !is.na(winner_id),
    !is.na(loser_id),
    !is.na(winner_rank),
    !is.na(loser_rank),
    !is.na(level),
    !is.na(surface)
  ) %>%
  arrange(tourney_date)

train_context <- df_seq_context %>%
  filter(year >= 2000, year <= 2018)

test_context <- df_seq_context %>%
  filter(year == 2019)

stopifnot(nrow(train_context) > 0, nrow(test_context) > 0)

# ---------------------------------------------------------
# 10) Contextual Elo processor
# ---------------------------------------------------------

# Reset Elo ratings for contextual Elo experiment
elo_pool <- new.env(parent = emptyenv())

process_contextual_elo_block <- function(data, update = TRUE, K = K_BASE) {
  out <- vector("list", nrow(data))
  
  for (k in seq_len(nrow(data))) {
    row <- data[k, ]
    
    # Define player i as the higher-ranked player
    if (row$winner_rank < row$loser_rank) {
      id_i <- row$winner_id
      id_j <- row$loser_id
      y_i <- 1L
      seed_i <- row$winner_seed
      seed_j <- row$loser_seed
    } else {
      id_i <- row$loser_id
      id_j <- row$winner_id
      y_i <- 0L
      seed_i <- row$loser_seed
      seed_j <- row$winner_seed
    }
    
    seed_diff <- as.integer(coalesce(seed_i, 0L) - coalesce(seed_j, 0L))
    
    rating_i <- get_rating(id_i)
    rating_j <- get_rating(id_j)
    
    p_i <- elo_expected_prob(rating_i, rating_j)
    
    out[[k]] <- tibble(
      tourney_date = row$tourney_date,
      year = row$year,
      
      level = row$level,
      surface = row$surface,
      
      id_i = id_i,
      id_j = id_j,
      
      seed_diff = seed_diff,
      deltaR = rating_i - rating_j,
      p_elo = p_i,
      y = y_i
    )
    
    if (update) {
      score_i <- y_i
      score_j <- 1 - score_i
      
      set_rating(id_i, rating_i + K * (score_i - p_i))
      set_rating(id_j, rating_j + K * (score_j - (1 - p_i)))
    }
  }
  
  bind_rows(out)
}

# ---------------------------------------------------------
# 11) Train contextual Elo calibration model
# ---------------------------------------------------------

# Reset Elo ratings for contextual Elo experiment
elo_pool <- new.env(parent = emptyenv())

context_train_predictions <- process_contextual_elo_block(
  data = train_context,
  update = TRUE,
  K = K_BASE
) %>%
  mutate(
    level = fct_relevel(level, "G"),
    surface = fct_relevel(surface, "Hard")
  )

# Logistic calibration model:
# y ~ Elo rating difference + contextual variables
contextual_elo_glm <- glm(
  y ~ deltaR + seed_diff + level + surface,
  data = context_train_predictions,
  family = binomial()
)

# ---------------------------------------------------------
# 12) Predict 2019 using contextual Elo calibration
# ---------------------------------------------------------

context_test_predictions <- process_contextual_elo_block(
  data = test_context,
  update = FALSE,
  K = K_BASE
) %>%
  mutate(
    level = fct_relevel(level, "G"),
    surface = fct_relevel(surface, "Hard"),
    p_context_elo = as.numeric(
      predict(contextual_elo_glm, newdata = ., type = "response")
    )
  )

write.csv(
  context_test_predictions,
  "results/contextual_elo_predictions_2019.csv",
  row.names = FALSE
)

# ---------------------------------------------------------
# 13) Evaluate contextual Elo model
# ---------------------------------------------------------

p <- pmin(pmax(context_test_predictions$p_context_elo, eps), 1 - eps)
y <- context_test_predictions$y

context_elo_acc <- mean((p >= 0.5) == as.logical(y))
context_elo_logloss <- -mean(y * log(p) + (1 - y) * log(1 - p))
context_elo_brier <- mean((p - y)^2)

contextual_elo_results <- tibble(
  Model = "Contextual Elo (deltaR + context)",
  Accuracy = round(context_elo_acc, 4),
  LogLoss = round(context_elo_logloss, 4),
  Brier = round(context_elo_brier, 4)
)

cat("\n=== Contextual Elo model on 2019 test set ===\n")
print(contextual_elo_results)

write.csv(
  contextual_elo_results,
  "results/contextual_elo_results.csv",
  row.names = FALSE
)

# ---------------------------------------------------------
# 14) Calibration plot for contextual Elo model
# ---------------------------------------------------------

context_elo_calib_tbl <- context_test_predictions %>%
  mutate(bin = ntile(p_context_elo, 10)) %>%
  group_by(bin) %>%
  summarise(
    n = n(),
    mean_pred = mean(p_context_elo),
    obs_rate = mean(y),
    .groups = "drop"
  )

p_context_elo_calibration <- ggplot(
  context_elo_calib_tbl,
  aes(mean_pred, obs_rate)
) +
  geom_point(size = 2) +
  geom_line(alpha = 0.8) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = 2
  ) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "Calibration — Contextual Elo",
    x = "Mean predicted probability",
    y = "Observed win rate"
  ) +
  theme_minimal(base_size = 13)

print(p_context_elo_calibration)

ggsave(
  filename = "figures/elo_context_calibration.png",
  plot = p_context_elo_calibration,
  width = 7,
  height = 5,
  dpi = 300
)

message("\nContextual Elo modelling completed.")
message("Results saved to results/contextual_elo_results.csv")
message("Predictions saved to results/contextual_elo_predictions_2019.csv")
message("Calibration plot saved to figures/elo_context_calibration.png")