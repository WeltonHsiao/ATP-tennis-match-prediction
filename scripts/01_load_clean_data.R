# =========================================================
# 01_load_clean_data.R
# Load and clean ATP match data
# =========================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(purrr)
  library(lubridate)
  library(stringr)
})

# ---------------------------------------------------------
# 0) Configuration
# ---------------------------------------------------------

data_dir <- "data/raw"
out_dir  <- "data/processed"

years <- 2000:2019
files <- file.path(data_dir, sprintf("atp_matches_%d.csv", years))

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

# ---------------------------------------------------------
# 1) Helper function for reading one file
# ---------------------------------------------------------

read_one <- function(path) {
  read_csv(
    path,
    show_col_types = FALSE,
    na = c("", "NA", "NaN")
  ) |>
    rename_with(~ str_replace_all(.x, "\\s+", "_"))
}

# ---------------------------------------------------------
# 2) Load annual ATP match files
# ---------------------------------------------------------

existing_files <- files[file.exists(files)]

if (length(existing_files) == 0L) {
  stop("No input files found. Please place ATP CSV files in data/raw/.")
}

message("Loaded files: ", paste(basename(existing_files), collapse = ", "))

df_raw <- map_dfr(existing_files, read_one, .id = "src_idx")

# ---------------------------------------------------------
# 3) Standardise variable types
# ---------------------------------------------------------

df_norm <- df_raw |>
  mutate(
    tourney_date = ymd(as.character(tourney_date)),
    
    across(
      .cols = c(match_num, winner_rank, loser_rank),
      .fns  = ~ suppressWarnings(as.integer(.x))
    ),
    
    across(
      .cols = c(winner_rank_points, loser_rank_points),
      .fns  = ~ suppressWarnings(as.numeric(.x))
    ),
    
    across(
      .cols = c(winner_age, loser_age, winner_ht, loser_ht),
      .fns  = ~ suppressWarnings(as.numeric(.x))
    )
  ) |>
  filter(!is.na(tourney_date)) |>
  arrange(tourney_date, match_num)

# ---------------------------------------------------------
# 4) Create seed indicators and retain modelling variables
# ---------------------------------------------------------

df_clean <- df_norm |>
  mutate(
    winner_seeded = if_else(!is.na(winner_seed) & winner_seed != "", 1L, 0L),
    loser_seeded  = if_else(!is.na(loser_seed)  & loser_seed  != "", 1L, 0L)
  ) |>
  transmute(
    tourney_id,
    tourney_name,
    surface,
    tourney_level,
    tourney_date,
    round,
    best_of,
    
    winner_id,
    loser_id,
    winner_rank,
    loser_rank,
    winner_rank_points,
    loser_rank_points,
    
    winner_hand,
    loser_hand,
    winner_age,
    loser_age,
    winner_ht,
    loser_ht,
    
    winner_seeded,
    loser_seeded
  ) |>
  filter(
    !is.na(winner_id),
    !is.na(loser_id),
    !is.na(winner_rank),
    !is.na(loser_rank),
    !is.na(winner_rank_points),
    !is.na(loser_rank_points),
    !is.na(surface),
    !is.na(winner_hand),
    !is.na(loser_hand)
  )

# ---------------------------------------------------------
# 5) Apply defensive quality filters
# ---------------------------------------------------------

df_clean <- df_clean |>
  filter(
    winner_rank > 0,
    loser_rank > 0,
    
    is.na(winner_ht)  | between(winner_ht, 150, 220),
    is.na(loser_ht)   | between(loser_ht, 150, 220),
    
    is.na(winner_age) | between(winner_age, 15, 45),
    is.na(loser_age)  | between(loser_age, 15, 45)
  )

# ---------------------------------------------------------
# 6) Save cleaned data
# ---------------------------------------------------------

saveRDS(df_clean, file.path(out_dir, "df_clean.rds"))

# ---------------------------------------------------------
# 7) Sanity checks
# ---------------------------------------------------------

message("\nRows x columns:")
print(dim(df_clean))

message("\nMissing values per column:")
print(colSums(is.na(df_clean)))

message("\nSurface breakdown:")
print(table(df_clean$surface, useNA = "ifany"))

message(
  "\nDate range: ",
  format(min(df_clean$tourney_date, na.rm = TRUE)),
  " to ",
  format(max(df_clean$tourney_date, na.rm = TRUE))
)

message("\nSaved cleaned data to: data/processed/df_clean.rds")