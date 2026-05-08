# =========================================================
# 02_exploratory_data_analysis.R
# Exploratory Data Analysis for ATP Tennis Match Prediction
# =========================================================
#
# Purpose:
# Conduct exploratory data analysis on the cleaned ATP match
# dataset. This script examines ranking distributions, player
# attributes, rank-based win patterns, contextual effects, and
# temporal trends.
#
# Input:
# data/processed/df_clean.rds
#
# Output:
# outputs/figures/eda/
#
# =========================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(forcats)
  library(scales)
  library(janitor)
  library(lubridate)
})

# ---------------------------------------------------------
# 0) Configuration
# ---------------------------------------------------------

input_path <- "data/processed/df_clean.rds"
figure_dir <- "outputs/figures/eda"

if (!dir.exists(figure_dir)) {
  dir.create(figure_dir, recursive = TRUE)
}

# Load cleaned data
df_clean <- readRDS(input_path)

# Global ggplot theme
theme_set(theme_minimal(base_size = 12))
update_geom_defaults("bar", list(color = NA))

# ---------------------------------------------------------
# 1) Quick structure and sanity checks
# ---------------------------------------------------------

glimpse(df_clean)

summary(df_clean$surface)

summary(
  dplyr::select(
    df_clean,
    winner_rank,
    loser_rank,
    winner_rank_points,
    loser_rank_points
  )
)

# ---------------------------------------------------------
# 2) EDA feature construction
# ---------------------------------------------------------

df_eda <- df_clean %>%
  mutate(
    year = year(tourney_date),
    
    winner_rank_f = if_else(is.na(winner_rank), 9999L, winner_rank),
    loser_rank_f  = if_else(is.na(loser_rank),  9999L, loser_rank),
    rank_diff     = loser_rank_f - winner_rank_f,
    
    rank_points_diff =
      coalesce(winner_rank_points, 0) -
      coalesce(loser_rank_points, 0),
    
    higher_rank_won = if_else(
      !is.na(winner_rank) &
        !is.na(loser_rank) &
        winner_rank < loser_rank,
      1L,
      0L,
      missing = NA_integer_
    ),
    
    winner_seeded = as.logical(coalesce(winner_seeded, 0L)),
    loser_seeded  = as.logical(coalesce(loser_seeded,  0L)),
    
    surface = fct_na_value_to_level(surface, level = "Unknown"),
    tourney_level = fct_relevel(
      as.factor(tourney_level),
      c("G", "M", "F", "A", "D")
    )
  ) %>%
  select(-winner_rank_f, -loser_rank_f)

# ---------------------------------------------------------
# 3) Distribution checks
# ---------------------------------------------------------

p_winner_rank <- ggplot(df_eda, aes(winner_rank)) +
  geom_histogram(binwidth = 25, fill = "grey40") +
  labs(
    title = "Winner Rank Distribution",
    x = "Winner rank (smaller = stronger)",
    y = "Count"
  )

print(p_winner_rank)

ggsave(
  filename = file.path(figure_dir, "winner_rank_distribution.png"),
  plot = p_winner_rank,
  width = 7,
  height = 5,
  dpi = 300
)

p_loser_rank <- ggplot(df_eda, aes(loser_rank)) +
  geom_histogram(binwidth = 25, fill = "grey40") +
  labs(
    title = "Loser Rank Distribution",
    x = "Loser rank (smaller = stronger)",
    y = "Count"
  )

print(p_loser_rank)

ggsave(
  filename = file.path(figure_dir, "loser_rank_distribution.png"),
  plot = p_loser_rank,
  width = 7,
  height = 5,
  dpi = 300
)

p_rank_points <- df_eda %>%
  pivot_longer(
    cols = c(winner_rank_points, loser_rank_points),
    names_to = "player_role",
    values_to = "rank_points"
  ) %>%
  ggplot(aes(rank_points, fill = player_role)) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 40) +
  scale_x_continuous(labels = label_comma()) +
  labs(
    title = "ATP Rank Points Distribution",
    x = "Rank points",
    y = "Count",
    fill = "Player role"
  )

print(p_rank_points)

ggsave(
  filename = file.path(figure_dir, "rank_points_distribution.png"),
  plot = p_rank_points,
  width = 7,
  height = 5,
  dpi = 300
)

message("\nEDA figures saved to:")
message(" - outputs/figures/eda/winner_rank_distribution.png")
message(" - outputs/figures/eda/loser_rank_distribution.png")
message(" - outputs/figures/eda/rank_points_distribution.png")

# Winner age distribution
p_winner_age <- ggplot(df_eda, aes(winner_age)) +
  geom_histogram(bins = 40, fill = "grey40") +
  labs(
    title = "Winner Age Distribution",
    x = "Age (years)",
    y = "Count"
  )

print(p_winner_age)

ggsave(
  filename = file.path(figure_dir, "winner_age_distribution.png"),
  plot = p_winner_age,
  width = 7,
  height = 5,
  dpi = 300
)

# Winner height distribution
p_winner_height <- df_eda %>%
  filter(!is.na(winner_ht), is.finite(winner_ht)) %>%
  ggplot(aes(winner_ht)) +
  geom_histogram(bins = 40, fill = "grey40") +
  coord_cartesian(xlim = c(150, 220)) +
  labs(
    title = "Winner Height Distribution",
    x = "Height (cm)",
    y = "Count"
  )

print(p_winner_height)

ggsave(
  filename = file.path(figure_dir, "winner_height_distribution.png"),
  plot = p_winner_height,
  width = 7,
  height = 5,
  dpi = 300
)

# ---------------------------------------------------------
# 4) Baseline pattern: higher-ranked player win rate
# ---------------------------------------------------------

higher_win_rate <- mean(df_eda$higher_rank_won, na.rm = TRUE)

cat(
  "Share of matches where the higher-ranked player won: ",
  percent(higher_win_rate, accuracy = 0.1),
  "\n"
)

# Win rate by absolute rank-difference bins
plot_gap_df <- df_eda %>%
  filter(
    !is.na(winner_rank),
    !is.na(loser_rank),
    winner_rank > 0,
    loser_rank > 0
  ) %>%
  mutate(
    abs_rank_diff = abs(loser_rank - winner_rank),
    higher_rank_won = as.integer(winner_rank < loser_rank),
    gap_bin = cut(
      abs_rank_diff,
      breaks = c(-Inf, 1, 5, 10, 25, 50, 100, 200, Inf),
      labels = c(
        "0–1", "2–5", "6–10", "11–25", "26–50",
        "51–100", "101–200", ">200"
      ),
      right = TRUE
    )
  ) %>%
  filter(!is.na(gap_bin)) %>%
  group_by(gap_bin) %>%
  summarise(
    n = n(),
    win_rate = mean(higher_rank_won, na.rm = TRUE),
    se = sqrt(pmax(win_rate * (1 - win_rate) / n, 0)),
    lo = pmax(0, win_rate - 1.96 * se),
    hi = pmin(1, win_rate + 1.96 * se),
    .groups = "drop"
  )

p_rank_winrate <- ggplot(plot_gap_df, aes(gap_bin, win_rate)) +
  geom_col(width = 0.85, fill = "grey40") +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    width = 0.25,
    linewidth = 0.6
  ) +
  geom_text(
    aes(label = paste0(percent(win_rate, 1), "\nn=", n)),
    vjust = -0.3,
    size = 3
  ) +
  coord_cartesian(ylim = c(0.45, 1.0)) +
  labs(
    title = "Win Rate of the Higher-Ranked Player by Rank Difference",
    x = "Absolute rank gap between players",
    y = "Win rate of higher-ranked player"
  )

print(p_rank_winrate)

# Main report figure
ggsave(
  filename = "outputs/figures/rank_winrate.png",
  plot = p_rank_winrate,
  width = 8,
  height = 5,
  dpi = 300
)

# EDA copy
ggsave(
  filename = file.path(figure_dir, "rank_winrate.png"),
  plot = p_rank_winrate,
  width = 8,
  height = 5,
  dpi = 300
)

# Smooth relationship between win probability and absolute rank gap
p_rank_gap_smooth <- df_eda %>%
  filter(
    !is.na(winner_rank),
    !is.na(loser_rank),
    winner_rank > 0,
    loser_rank > 0
  ) %>%
  mutate(
    abs_rank_diff = abs(loser_rank - winner_rank),
    higher_rank_won = as.integer(winner_rank < loser_rank)
  ) %>%
  ggplot(aes(abs_rank_diff, higher_rank_won)) +
  geom_smooth(
    method = "glm",
    method.args = list(family = "binomial"),
    formula = y ~ splines::ns(x, 3)
  ) +
  scale_x_continuous(
    trans = "log1p",
    breaks = c(1, 5, 10, 25, 50, 100, 200, 500)
  ) +
  labs(
    title = "Higher-Ranked Wins vs. Absolute Rank Gap",
    x = "Absolute rank gap (log scale)",
    y = "Win probability"
  )

print(p_rank_gap_smooth)

ggsave(
  filename = file.path(figure_dir, "rank_gap_smooth.png"),
  plot = p_rank_gap_smooth,
  width = 8,
  height = 5,
  dpi = 300
)

# ---------------------------------------------------------
# 5) Context effects: surface, tournament level, and seeding
# ---------------------------------------------------------

# Higher-ranked win rate by surface
p_surface_winrate <- df_eda %>%
  group_by(surface) %>%
  summarise(
    win_rate = mean(higher_rank_won, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  ggplot(aes(surface, win_rate)) +
  geom_col(fill = "grey40") +
  geom_text(
    aes(label = paste0(percent(win_rate, 1), "\n(n=", n, ")")),
    vjust = -0.2,
    size = 3
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    title = "Higher-Ranked Win Rate by Surface",
    x = "Surface",
    y = "Win rate"
  )

print(p_surface_winrate)

ggsave(
  filename = file.path(figure_dir, "surface_winrate.png"),
  plot = p_surface_winrate,
  width = 7,
  height = 5,
  dpi = 300
)

# Higher-ranked win rate by tournament level
p_level_winrate <- df_eda %>%
  group_by(tourney_level) %>%
  summarise(
    win_rate = mean(higher_rank_won, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  ggplot(aes(tourney_level, win_rate)) +
  geom_col(fill = "grey40") +
  geom_text(
    aes(label = paste0(percent(win_rate, 1), "\n(n=", n, ")")),
    vjust = -0.2,
    size = 3
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    title = "Higher-Ranked Win Rate by Tournament Level",
    x = "Tournament level",
    y = "Win rate"
  )

print(p_level_winrate)

ggsave(
  filename = file.path(figure_dir, "tournament_level_winrate.png"),
  plot = p_level_winrate,
  width = 7,
  height = 5,
  dpi = 300
)

# Higher-ranked win rate when the match winner was seeded
p_seeded_winrate <- df_eda %>%
  group_by(winner_seeded = coalesce(winner_seeded, FALSE)) %>%
  summarise(
    win_rate = mean(higher_rank_won, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  ggplot(aes(winner_seeded, win_rate)) +
  geom_col(fill = "grey40") +
  geom_text(
    aes(label = paste0(percent(win_rate, 1), "\n(n=", n, ")")),
    vjust = -0.2,
    size = 3
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    title = "Higher-Ranked Win Rate when Winner is Seeded",
    x = "Winner seeded?",
    y = "Win rate"
  )

print(p_seeded_winrate)

ggsave(
  filename = file.path(figure_dir, "winner_seeded_winrate.png"),
  plot = p_seeded_winrate,
  width = 7,
  height = 5,
  dpi = 300
)

# ---------------------------------------------------------
# 6) Time trends: motivation for time-aware models
# ---------------------------------------------------------

# Yearly win rate of the higher-ranked player
yearly_higher_rank_winrate <- df_eda %>%
  group_by(year) %>%
  summarise(
    win_rate = mean(higher_rank_won, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

p_temporal_trends <- ggplot(yearly_higher_rank_winrate, aes(year, win_rate)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  scale_y_continuous(labels = percent) +
  labs(
    title = "Yearly Win Rate of Higher-Ranked Player",
    x = "Year",
    y = "Win rate"
  )

print(p_temporal_trends)

# Main report figure
ggsave(
  filename = "outputs/figures/temporal_trends.png",
  plot = p_temporal_trends,
  width = 8,
  height = 5,
  dpi = 300
)

# EDA copy
ggsave(
  filename = file.path(figure_dir, "temporal_trends.png"),
  plot = p_temporal_trends,
  width = 8,
  height = 5,
  dpi = 300
)

# Yearly mean absolute rank difference
yearly_rank_diff <- df_eda %>%
  group_by(year) %>%
  summarise(
    mean_abs_rank_diff = mean(abs(rank_diff), na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

p_yearly_rank_diff <- ggplot(yearly_rank_diff, aes(year, mean_abs_rank_diff)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  labs(
    title = "Yearly Mean Absolute Rank Difference",
    x = "Year",
    y = "Mean absolute rank difference"
  )

print(p_yearly_rank_diff)

ggsave(
  filename = file.path(figure_dir, "yearly_mean_abs_rank_diff.png"),
  plot = p_yearly_rank_diff,
  width = 8,
  height = 5,
  dpi = 300
)

# ---------------------------------------------------------
# 7) Summary tables
# ---------------------------------------------------------

table_dir <- "outputs/tables"

if (!dir.exists(table_dir)) {
  dir.create(table_dir, recursive = TRUE)
}

# Top 10 tournaments by match count
top_tournaments <- df_eda %>%
  count(tourney_name, sort = TRUE) %>%
  slice_head(n = 10)

print(top_tournaments)

write.csv(
  top_tournaments,
  file = file.path(table_dir, "top_10_tournaments.csv"),
  row.names = FALSE
)

# Match counts by surface and tournament level
surface_level_counts <- df_eda %>%
  count(surface, tourney_level, sort = TRUE)

print(surface_level_counts)

write.csv(
  surface_level_counts,
  file = file.path(table_dir, "surface_level_counts.csv"),
  row.names = FALSE
)

# Summary statistics by surface
surface_summary <- df_eda %>%
  group_by(surface) %>%
  summarise(
    across(
      c(winner_age, winner_ht, winner_rank, winner_rank_points),
      list(
        mean = ~ mean(.x, na.rm = TRUE),
        sd = ~ sd(.x, na.rm = TRUE)
      )
    ),
    .groups = "drop"
  )

print(surface_summary)

write.csv(
  surface_summary,
  file = file.path(table_dir, "surface_summary.csv"),
  row.names = FALSE
)

message("\nEDA completed.")
message("Figures saved to:")
message(" - outputs/figures/")
message(" - outputs/figures/eda/")

message("\nSummary tables saved to:")
message(" - outputs/tables/")
