# =========================================================
# 06_lag_enhanced_models.R
# Lag-Enhanced Rolling Logistic Regression
# =========================================================
#
# Purpose:
# Build lagged player win-rate features and evaluate whether
# recent historical performance improves rolling-window
# logistic regression models for ATP match prediction.
#
# Lag features:
#   - wr_diff_lag1: previous 1-season win-rate difference
#   - wr_diff_lag2: previous 2-season win-rate difference
#   - wr_diff_lag3: previous 3-season win-rate difference
#
# Evaluation:
#   - Accuracy
#   - Log-loss
#   - Brier score
#
# =========================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(purrr)
  library(lubridate)
  library(forcats)
  library(tidyr)
  library(ggplot2)
  library(broom)
})

# ---------------------------------------------------------
# 0) Configuration
# ---------------------------------------------------------

WINDOW_YEARS <- 3
START_YEAR <- 2003
END_YEAR <- 2019
K_LAGS <- 3

input_path <- "data/processed/df_clean.rds"
table_dir  <- "outputs/tables"
figure_dir <- "outputs/figures"

if (!file.exists(input_path)) {
  stop("Cleaned data not found. Please run scripts/01_load_clean_data.R first.")
}

if (!dir.exists(table_dir)) {
  dir.create(table_dir, recursive = TRUE)
}

if (!dir.exists(figure_dir)) {
  dir.create(figure_dir, recursive = TRUE)
}

df_clean <- readRDS(input_path)

# ---------------------------------------------------------
# 1) Build player-year win rates
# ---------------------------------------------------------

player_year_stats <- df_clean %>%
  mutate(year = year(tourney_date)) %>%
  transmute(year, winner_id, loser_id) %>%
  bind_rows(
    transmute(., year, player_id = winner_id, win = 1L),
    transmute(., year, player_id = loser_id, win = 0L)
  ) %>%
  group_by(player_id, year) %>%
  summarise(
    win_rate = mean(win),
    n_matches = n(),
    .groups = "drop"
  )

write.csv(
  player_year_stats,
  file.path(table_dir, "player_year_win_rates.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 2) Rebuild match-level modelling frame
# ---------------------------------------------------------

df_model_lag <- df_clean %>%
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
    
    x = hi_pts - lo_pts,
    
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
    ),
    
    higher_id = if_else(
      winner_rank < loser_rank,
      winner_id,
      loser_id
    ),
    
    lower_id = if_else(
      winner_rank < loser_rank,
      loser_id,
      winner_id
    )
  ) %>%
  filter(
    !is.na(year),
    !is.na(y),
    !is.na(x),
    !is.na(surface)
  ) %>%
  select(
    year,
    y,
    y_num,
    x,
    seed_diff,
    tourney_level,
    surface,
    higher_id,
    lower_id
  )

# ---------------------------------------------------------
# 3) Add lagged win-rate difference features
# ---------------------------------------------------------

for (k in 1:K_LAGS) {
  higher_k <- player_year_stats %>%
    transmute(
      higher_id = player_id,
      year = year + k,
      higher_wr = win_rate
    )
  
  lower_k <- player_year_stats %>%
    transmute(
      lower_id = player_id,
      year = year + k,
      lower_wr = win_rate
    )
  
  df_model_lag <- df_model_lag %>%
    left_join(higher_k, by = c("higher_id", "year")) %>%
    left_join(lower_k, by = c("lower_id", "year")) %>%
    mutate(
      !!paste0("wr_diff_lag", k) :=
        coalesce(higher_wr, 0.5) - coalesce(lower_wr, 0.5)
    ) %>%
    select(-higher_wr, -lower_wr)
}

# ---------------------------------------------------------
# 4) Remove early seasons with incomplete lag history
# ---------------------------------------------------------

df_model_lag_before_remove <- df_model_lag

df_model_lag <- df_model_lag %>%
  filter(year >= START_YEAR)

# Save lag modelling datasets
saveRDS(
  df_model_lag_before_remove,
  "data/processed/df_model_lag_before_remove.rds"
)

saveRDS(
  df_model_lag,
  "data/processed/df_model_lag.rds"
)

write.csv(
  df_model_lag,
  file.path(table_dir, "df_model_lag_summary_data.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 5) Quick check
# ---------------------------------------------------------

glimpse(
  df_model_lag %>%
    select(
      year,
      y_num,
      x,
      seed_diff,
      tourney_level,
      surface,
      starts_with("wr_diff_lag")
    )
)

message("\nLagged win-rate features constructed.")
message("Lag model data saved to:")
message(" - data/processed/df_model_lag_before_remove.rds")
message(" - data/processed/df_model_lag.rds")
message(" - outputs/tables/df_model_lag_summary_data.csv")

# ---------------------------------------------------------
# 6) Check lag feature quality
# ---------------------------------------------------------

lag_check_by_year <- df_model_lag %>%
  group_by(year) %>%
  summarise(
    n_matches = n(),
    
    mean_lag1 = mean(wr_diff_lag1, na.rm = TRUE),
    mean_lag2 = mean(wr_diff_lag2, na.rm = TRUE),
    mean_lag3 = mean(wr_diff_lag3, na.rm = TRUE),
    
    zero_rate_lag1 = mean(wr_diff_lag1 == 0, na.rm = TRUE),
    zero_rate_lag2 = mean(wr_diff_lag2 == 0, na.rm = TRUE),
    zero_rate_lag3 = mean(wr_diff_lag3 == 0, na.rm = TRUE),
    
    .groups = "drop"
  )

cat("\n=== Lag feature quality check by year ===\n")
print(lag_check_by_year)

write.csv(
  lag_check_by_year,
  file.path(table_dir, "lag_check_by_year.csv"),
  row.names = FALSE
)

# Reshape lag features after removing 2000–2002
lag_long <- df_model_lag %>%
  select(year, starts_with("wr_diff_lag")) %>%
  pivot_longer(
    cols = starts_with("wr_diff_lag"),
    names_to = "lag_term",
    values_to = "wr_diff"
  )

# Reshape lag features before removing 2000–2002
lag_long_before <- df_model_lag_before_remove %>%
  select(year, starts_with("wr_diff_lag")) %>%
  pivot_longer(
    cols = starts_with("wr_diff_lag"),
    names_to = "lag_term",
    values_to = "wr_diff"
  )

# ---------------------------------------------------------
# 7) Lag distributions before removing early seasons
# ---------------------------------------------------------

p_lag_before_remove <- lag_long_before %>%
  filter(year <= 2005) %>%
  ggplot(aes(x = wr_diff)) +
  geom_histogram(bins = 30, boundary = 0, fill = "grey40") +
  facet_grid(lag_term ~ year) +
  labs(
    title = "Early-Year Lag Feature Distributions Before Removing 2000–2002",
    subtitle = "Early seasons show incomplete lag history and more neutral values",
    x = "Lagged win-rate difference",
    y = "Number of matches"
  ) +
  theme_minimal(base_size = 12)

print(p_lag_before_remove)

ggsave(
  filename = file.path(figure_dir, "lag_before_remove.png"),
  plot = p_lag_before_remove,
  width = 10,
  height = 6,
  dpi = 300
)

# ---------------------------------------------------------
# 8) Lag distributions after removing early seasons
# ---------------------------------------------------------

p_lag_after_remove <- lag_long %>%
  filter(year <= 2008) %>%
  ggplot(aes(x = wr_diff)) +
  geom_histogram(bins = 30, boundary = 0, fill = "grey40") +
  facet_grid(lag_term ~ year) +
  labs(
    title = "Early-Year Lag Feature Distributions After Removing 2000–2002",
    subtitle = "Lag features are checked after removing incomplete early seasons",
    x = "Lagged win-rate difference",
    y = "Number of matches"
  ) +
  theme_minimal(base_size = 12)

print(p_lag_after_remove)

ggsave(
  filename = file.path(figure_dir, "lag_after_remove.png"),
  plot = p_lag_after_remove,
  width = 10,
  height = 6,
  dpi = 300
)

# Optional full yearly lag distribution plot
p_lag_full_distribution <- ggplot(lag_long, aes(x = wr_diff)) +
  geom_histogram(bins = 30, boundary = 0, fill = "grey40") +
  facet_grid(lag_term ~ year) +
  labs(
    title = "Yearly Distribution of Lagged Win-Rate Difference Features",
    subtitle = "After removing 2000–2002 due to incomplete lag history",
    x = "Lagged win-rate difference",
    y = "Number of matches"
  ) +
  theme_minimal(base_size = 11)

print(p_lag_full_distribution)

ggsave(
  filename = file.path(figure_dir, "lag_full_distribution.png"),
  plot = p_lag_full_distribution,
  width = 14,
  height = 6,
  dpi = 300
)

# ---------------------------------------------------------
# 9) Mean lag feature values over time
# ---------------------------------------------------------

lag_mean_by_year <- df_model_lag %>%
  group_by(year) %>%
  summarise(
    wr_diff_lag1 = mean(wr_diff_lag1, na.rm = TRUE),
    wr_diff_lag2 = mean(wr_diff_lag2, na.rm = TRUE),
    wr_diff_lag3 = mean(wr_diff_lag3, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = starts_with("wr_diff_lag"),
    names_to = "lag_term",
    values_to = "mean_wr_diff"
  )

cat("\n=== Mean lag feature values by year ===\n")
print(lag_mean_by_year)

write.csv(
  lag_mean_by_year,
  file.path(table_dir, "lag_mean_by_year.csv"),
  row.names = FALSE
)

# Bar plot version
p_lag_mean_bar <- ggplot(
  lag_mean_by_year,
  aes(x = factor(year), y = mean_wr_diff, fill = lag_term)
) +
  geom_col(position = "dodge") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Average Lagged Win-Rate Difference by Year",
    subtitle = "After removing 2000–2002",
    x = "Year",
    y = "Average lagged win-rate difference",
    fill = "Lag term"
  ) +
  theme_minimal(base_size = 12)

print(p_lag_mean_bar)

ggsave(
  filename = file.path(figure_dir, "lag_mean_by_year_bar.png"),
  plot = p_lag_mean_bar,
  width = 10,
  height = 5,
  dpi = 300
)

# Line plot version used in the report
p_lag_over_time <- ggplot(
  lag_mean_by_year,
  aes(x = year, y = mean_wr_diff, color = lag_term)
) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Average Lagged Win-Rate Difference Over Time",
    subtitle = "After removing 2000–2002",
    x = "Year",
    y = "Average lagged win-rate difference",
    color = "Lag term"
  ) +
  theme_minimal(base_size = 12)

print(p_lag_over_time)

ggsave(
  filename = file.path(figure_dir, "lag_over_time.png"),
  plot = p_lag_over_time,
  width = 8,
  height = 5,
  dpi = 300
)

message("\nLag feature quality checks completed.")
message("Results saved to:")
message(" - outputs/tables/lag_mean_by_year.csv")
message("Figures saved to:")
message(" - outputs/figures/lag_before_remove.png")
message(" - outputs/figures/lag_after_remove.png")
message(" - outputs/figures/lag_full_distribution.png")
message(" - outputs/figures/lag_mean_by_year_bar.png")
message(" - outputs/figures/lag_over_time.png")

# ---------------------------------------------------------
# 10) Helper: standardise lag variables using training data
# ---------------------------------------------------------

scale_with_train <- function(train_df, test_df, vars) {
  means <- sapply(train_df[vars], mean, na.rm = TRUE)
  sds <- sapply(train_df[vars], sd, na.rm = TRUE)
  
  # Avoid division by zero
  sds[is.na(sds) | sds == 0] <- 1
  
  for (v in vars) {
    train_df[[v]] <- (train_df[[v]] - means[[v]]) / sds[[v]]
    test_df[[v]] <- (test_df[[v]] - means[[v]]) / sds[[v]]
  }
  
  list(
    train = train_df,
    test = test_df
  )
}

# ---------------------------------------------------------
# 11) Fit one lag-enhanced rolling model
# ---------------------------------------------------------

fit_one_lag_model <- function(train_data, test_data) {
  lag_vars <- c("wr_diff_lag1", "wr_diff_lag2", "wr_diff_lag3")
  
  scaled <- scale_with_train(train_data, test_data, lag_vars)
  train_s <- scaled$train
  test_s <- scaled$test
  
  model <- glm(
    y_num ~ x + seed_diff + tourney_level +
      wr_diff_lag1 + wr_diff_lag2 + wr_diff_lag3,
    data = train_s,
    family = binomial()
  )
  
  pred <- predict(model, newdata = test_s, type = "response")
  
  eps <- 1e-15
  p <- pmin(pmax(pred, eps), 1 - eps)
  y <- test_s$y_num
  
  metrics <- tibble(
    test_year = unique(test_s$year),
    n_train = nrow(train_s),
    n_test = nrow(test_s),
    accuracy = mean((p >= 0.5) == (y == 1)),
    logloss = -mean(y * log(p) + (1 - y) * log(1 - p)),
    brier = mean((p - y)^2)
  )
  
  lag_coefs <- tidy(model) %>%
    filter(term %in% lag_vars) %>%
    select(term, estimate, std.error, statistic, p.value) %>%
    mutate(test_year = unique(test_s$year)) %>%
    relocate(test_year)
  
  list(
    metrics = metrics,
    lag_coefs = lag_coefs
  )
}

# ---------------------------------------------------------
# 12) Rolling-window evaluation for lag-enhanced model
# ---------------------------------------------------------

lag_model_out <- map(
  START_YEAR:END_YEAR,
  function(test_year) {
    train_data <- df_model_lag %>%
      filter(year >= test_year - WINDOW_YEARS, year < test_year)
    
    test_data <- df_model_lag %>%
      filter(year == test_year)
    
    if (nrow(train_data) > 100 && nrow(test_data) > 100) {
      fit_one_lag_model(train_data, test_data)
    } else {
      NULL
    }
  }
)

lag_model_results <- lag_model_out %>%
  discard(is.null) %>%
  map_dfr("metrics")

lag_model_coefs <- lag_model_out %>%
  discard(is.null) %>%
  map_dfr("lag_coefs")

lag_model_summary <- lag_model_results %>%
  summarise(
    mean_accuracy = mean(accuracy, na.rm = TRUE),
    mean_logloss = mean(logloss, na.rm = TRUE),
    mean_brier = mean(brier, na.rm = TRUE)
  ) %>%
  mutate(
    across(
      everything(),
      ~ round(.x, 4)
    )
  )

cat("\n=== Lag-enhanced rolling logistic results ===\n")
print(lag_model_results)

cat("\n=== Lag-enhanced rolling logistic summary ===\n")
print(lag_model_summary)

cat("\n=== Lag feature coefficients ===\n")
print(lag_model_coefs)

write.csv(
  lag_model_results,
  file.path(table_dir, "lag_model_results.csv"),
  row.names = FALSE
)

write.csv(
  lag_model_summary,
  file.path(table_dir, "lag_model_summary.csv"),
  row.names = FALSE
)

write.csv(
  lag_model_coefs,
  file.path(table_dir, "lag_model_coefficients.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 13) Baseline rolling model without lag features
# ---------------------------------------------------------

fit_one_baseline_model <- function(train_data, test_data) {
  model <- glm(
    y_num ~ x + seed_diff + tourney_level,
    data = train_data,
    family = binomial()
  )
  
  pred <- predict(model, newdata = test_data, type = "response")
  
  eps <- 1e-15
  p <- pmin(pmax(pred, eps), 1 - eps)
  y <- test_data$y_num
  
  tibble(
    test_year = unique(test_data$year),
    n_train = nrow(train_data),
    n_test = nrow(test_data),
    accuracy = mean((p >= 0.5) == (y == 1)),
    logloss = -mean(y * log(p) + (1 - y) * log(1 - p)),
    brier = mean((p - y)^2)
  )
}

baseline_out <- map(
  START_YEAR:END_YEAR,
  function(test_year) {
    train_data <- df_model_lag %>%
      filter(year >= test_year - WINDOW_YEARS, year < test_year)
    
    test_data <- df_model_lag %>%
      filter(year == test_year)
    
    if (nrow(train_data) > 100 && nrow(test_data) > 100) {
      fit_one_baseline_model(train_data, test_data)
    } else {
      NULL
    }
  }
)

baseline_results <- baseline_out %>%
  discard(is.null) %>%
  map_dfr(~ .x)

cat("\n=== Rolling logistic baseline without lag features ===\n")
print(baseline_results)

write.csv(
  baseline_results,
  file.path(table_dir, "lag_baseline_results.csv"),
  row.names = FALSE
)

message("\nLag-enhanced rolling model fitted.")
message("Lag model outputs saved to outputs/tables/ and outputs/figures/.")

# ---------------------------------------------------------
# 14) Compare original rolling model and lag-enhanced model
# ---------------------------------------------------------

compare_lag_summary <- bind_rows(
  baseline_results %>%
    mutate(Model = "Original rolling logistic regression"),
  
  lag_model_results %>%
    mutate(Model = "Lag-enhanced rolling logistic regression")
) %>%
  group_by(Model) %>%
  summarise(
    mean_accuracy = mean(accuracy, na.rm = TRUE),
    mean_logloss = mean(logloss, na.rm = TRUE),
    mean_brier = mean(brier, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    across(
      c(mean_accuracy, mean_logloss, mean_brier),
      ~ round(.x, 4)
    )
  )

cat("\n=== Original vs lag-enhanced rolling model summary ===\n")
print(compare_lag_summary)

write.csv(
  compare_lag_summary,
  file.path(table_dir, "lag_model_comparison_summary.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 15) Combine yearly rolling results
# ---------------------------------------------------------

compare_lag_results <- bind_rows(
  baseline_results %>%
    mutate(Model = "Original rolling logistic regression"),
  
  lag_model_results %>%
    mutate(Model = "Lag-enhanced rolling logistic regression")
)

cat("\n=== Original vs lag-enhanced yearly results ===\n")
print(compare_lag_results)

write.csv(
  compare_lag_results,
  file.path(table_dir, "lag_model_comparison_yearly.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 16) Plot yearly performance comparison
# ---------------------------------------------------------

p_lag_compare_models <- compare_lag_results %>%
  pivot_longer(
    cols = c(accuracy, logloss, brier),
    names_to = "metric",
    values_to = "value"
  ) %>%
  ggplot(aes(
    x = test_year,
    y = value,
    color = Model
  )) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  facet_wrap(~ metric, scales = "free_y") +
  labs(
    title = "Original vs Lag-Enhanced Rolling Logistic Models",
    x = "Test year",
    y = "Metric value",
    color = "Model"
  ) +
  theme_minimal(base_size = 13)

print(p_lag_compare_models)

ggsave(
  filename = file.path(figure_dir, "lag_compare_models.png"),
  plot = p_lag_compare_models,
  width = 9,
  height = 5,
  dpi = 300
)

# ---------------------------------------------------------
# 17) Plot lag-feature coefficients over time
# ---------------------------------------------------------

p_lag_feature_coefficients <- lag_model_coefs %>%
  ggplot(aes(
    x = test_year,
    y = estimate,
    color = term
  )) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(
    title = "Lag Feature Coefficients Over Time",
    x = "Test year",
    y = "Coefficient estimate",
    color = "Lag term"
  ) +
  theme_minimal(base_size = 13)

print(p_lag_feature_coefficients)

ggsave(
  filename = file.path(figure_dir, "lag_feature_coefficients.png"),
  plot = p_lag_feature_coefficients,
  width = 8,
  height = 5,
  dpi = 300
)

# ---------------------------------------------------------
# 18) Final messages
# ---------------------------------------------------------

cat("\n=== Final lag-enhanced model summary ===\n")
print(lag_model_summary)

cat("\n=== Final comparison summary ===\n")
print(compare_lag_summary)

message("\nLag-enhanced rolling logistic analysis completed.")
message("Comparison tables saved to:")
message(" - outputs/tables/lag_baseline_results.csv")
message(" - outputs/tables/lag_model_results.csv")
message(" - outputs/tables/lag_model_summary.csv")
message(" - outputs/tables/lag_model_coefficients.csv")
message(" - outputs/tables/lag_model_comparison_summary.csv")
message(" - outputs/tables/lag_model_comparison_yearly.csv")
message("Figures saved to:")
message(" - outputs/figures/lag_compare_models.png")
message(" - outputs/figures/lag_feature_coefficients.png")
