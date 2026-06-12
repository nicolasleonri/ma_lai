# Load required libraries
library(dplyr)
library(readr)

# Read the CSV file
getwd()
data <- read_csv("./results/csv/evaluation.csv")
data <- read_csv("./test.csv")
summary(data)

# Function to calculate harmonic mean
harmonic_mean <- function(x) {
  return(length(x) / sum(1/x))
}

# Group by Filename and calculate harmonic mean for numeric columns
result <- data %>%
  group_by(Config, OCR_module, LLM_model) %>%
  summarise(across(c(Accuracy, Precision, Recall, F1_Score), ~harmonic_mean(.[. != 0])), .groups = "drop")

top10_accuracy <- result %>% arrange(desc(Accuracy))
top10_f1score <- result %>% arrange(desc(F1_Score))

# View the result
print(top10_f1score)
print(top10_accuracy)
print(result)

# Optionally, write the result to a new CSV file
write_csv(top10_f1score, "./results/csv/evaluation_results.csv")
