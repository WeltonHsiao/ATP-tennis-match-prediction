# =========================================================
# 05_rolling_window_models.R
# Rolling-Window Logistic Regression
# =========================================================
#
# Purpose:
# Evaluate rolling-window logistic regression models for ATP
# tennis match prediction. Models are trained using only
# recent historical observations and tested on the following
# season.
#
# Main model:
#   - 3-year rolling window
#
# Predictors:
#   - ranking-point difference
#   - seed difference
#   - tournament level
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
  library(ggplot2)
  library(tidyr)
  library(forcats)
  library(lubridate)
})

# ---------------------------------------------------------
# 0) Configuration
# ---------------------------------------------------------

WINDOW_YEARS <- 3
START_YEAR <- 2003
END_YEAR <- 2019

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
# 1) Prepare modelling frame
# ---------------------------------------------------------

df_roll <- df_clean %>%
  mutate(
    year = year(tourney_date),
    
    # Outcome: 1 if the higher-ranked player won
    y = as.integer(winner_rank < loser_rank),
    
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
    )
  ) %>%
  transmute(
    year,
    y,
    x,
    seed_diff,
    tourney_level = as.factor(tourney_level)
  ) %>%
  filter(
    !is.na(year),
    !is.na(y),
    !is.na(x),
    !is.na(seed_diff),
    !is.na(tourney_level)
  )

# ---------------------------------------------------------
# 2) Helper function: fit and evaluate one rolling window
# ---------------------------------------------------------

fit_one_window <- function(train_data, test_data) {
  model <- glm(
    y ~ x + seed_diff + tourney_level,
    data = train_data,
    family = binomial()
  )
  
  pred <- predict(model, newdata = test_data, type = "response")
  
  eps <- 1e-15
  p <- pmin(pmax(pred, eps), 1 - eps)
  y <- test_data$y
  
  tibble(
    test_year = unique(test_data$year),
    n_train = nrow(train_data),
    n_test = nrow(test_data),
    accuracy = mean((p >= 0.5) == (y == 1)),
    logloss = -mean(y * log(p) + (1 - y) * log(1 - p)),
    brier = mean((p - y)^2)
  )
}

# ---------------------------------------------------------
# 3) Rolling loop over years
# ---------------------------------------------------------

rolling_results <- map_dfr(
  START_YEAR:END_YEAR,
  function(y_test) {
    y_train_start <- y_test - WINDOW_YEARS
    
    train_data <- df_roll %>%
      filter(year >= y_train_start, year < y_test)
    
    test_data <- df_roll %>%
      filter(year == y_test)
    
    if (nrow(train_data) > 100 && nrow(test_data) > 100) {
      fit_one_window(train_data, test_data)
    } else {
      NULL
    }
  }
)

# ---------------------------------------------------------
# 4) Inspect numeric results
# ---------------------------------------------------------

print(rolling_results)

write.csv(
  rolling_results,
  file.path(table_dir, "rolling_results_3_year_window.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 5) Visualize 3-year rolling-window performance over time
# ---------------------------------------------------------

p_rolling_results <- rolling_results %>%
  pivot_longer(
    cols = c(accuracy, logloss, brier),
    names_to = "metric",
    values_to = "value"
  ) %>%
  ggplot(aes(x = test_year, y = value, color = metric)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(
    title = "Rolling-Window Results",
    x = "Test year",
    y = "Metric value",
    color = "Metric"
  ) +
  theme_minimal(base_size = 13)

print(p_rolling_results)

ggsave(
  filename = file.path(figure_dir, "rolling_results_3_year_window.png"),
  plot = p_rolling_results,
  width = 8,
  height = 5,
  dpi = 300
)

# ---------------------------------------------------------
# 6) Sensitivity analysis: compare rolling-window sizes
# ---------------------------------------------------------

window_grid <- c(1, 2, 3, 4, 5)

run_rolling_window <- function(window_years, data, end_year = 2019, min_n = 100) {
  start_year <- min(data$year, na.rm = TRUE) + window_years
  
  map_dfr(start_year:end_year, function(test_year) {
    train_data <- data %>%
      filter(year >= test_year - window_years, year < test_year)
    
    test_data <- data %>%
      filter(year == test_year)
    
    if (nrow(train_data) > min_n && nrow(test_data) > min_n) {
      fit_one_window(train_data, test_data) %>%
        mutate(window_years = window_years)
    } else {
      NULL
    }
  })
}

rolling_all <- map_dfr(
  window_grid,
  ~ run_rolling_window(window_years = .x, data = df_roll)
)

cat("\n=== Rolling-window results across candidate window sizes ===\n")
print(rolling_all)

write.csv(
  rolling_all,
  file.path(table_dir, "rolling_all_window_sizes.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 7) Summary table by window size
# ---------------------------------------------------------

rolling_summary <- rolling_all %>%
  group_by(window_years) %>%
  summarise(
    mean_accuracy = mean(accuracy, na.rm = TRUE),
    sd_accuracy = sd(accuracy, na.rm = TRUE),
    mean_logloss = mean(logloss, na.rm = TRUE),
    sd_logloss = sd(logloss, na.rm = TRUE),
    mean_brier = mean(brier, na.rm = TRUE),
    sd_brier = sd(brier, na.rm = TRUE),
    n_test_years = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_accuracy), mean_logloss, mean_brier)

cat("\n=== Rolling-window summary by window size ===\n")
print(rolling_summary)

write.csv(
  rolling_summary,
  file.path(table_dir, "rolling_window_summary.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 8) Plot performance across window sizes
# ---------------------------------------------------------

p_window_size_results <- rolling_all %>%
  pivot_longer(
    cols = c(accuracy, logloss, brier),
    names_to = "metric",
    values_to = "value"
  ) %>%
  ggplot(aes(
    x = test_year,
    y = value,
    color = factor(window_years)
  )) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 1.8) +
  facet_wrap(~ metric, scales = "free_y", ncol = 1) +
  labs(
    title = "Rolling-Window Performance Across Window Sizes",
    x = "Test year",
    y = "Metric value",
    color = "Window size"
  ) +
  theme_minimal(base_size = 13)

print(p_window_size_results)

ggsave(
  filename = file.path(figure_dir, "window_size_performance_over_time.png"),
  plot = p_window_size_results,
  width = 8,
  height = 8,
  dpi = 300
)

# ---------------------------------------------------------
# 9) Plot average performance by window size
# ---------------------------------------------------------

p_window_size_comparison <- rolling_summary %>%
  pivot_longer(
    cols = c(mean_accuracy, mean_logloss, mean_brier),
    names_to = "metric",
    values_to = "value"
  ) %>%
  ggplot(aes(
    x = factor(window_years),
    y = value,
    group = metric
  )) +
  geom_line() +
  geom_point(size = 2) +
  facet_wrap(~ metric, scales = "free_y", ncol = 1) +
  labs(
    title = "Average Rolling-Window Performance by Window Size",
    x = "Window size (years)",
    y = "Average metric value"
  ) +
  theme_minimal(base_size = 13)

print(p_window_size_comparison)

ggsave(
  filename = file.path(figure_dir, "window_size_comparison.png"),
  plot = p_window_size_comparison,
  width = 7,
  height = 8,
  dpi = 300
)

message("\nRolling-window sensitivity analysis completed.")
message("Results saved to:")
message(" - outputs/tables/rolling_results_3_year_window.csv")
message(" - outputs/tables/rolling_all_window_sizes.csv")
message(" - outputs/tables/rolling_window_summary.csv")
message("Figures saved to:")
message(" - outputs/figures/rolling_results_3_year_window.png")
message(" - outputs/figures/window_size_performance_over_time.png")
message(" - outputs/figures/window_size_comparison.png")
