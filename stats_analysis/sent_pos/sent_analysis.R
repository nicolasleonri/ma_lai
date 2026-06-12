####### IMPORTS ####### 
if (!require(irr)) install.packages("irr")
if (!require(caret)) install.packages("caret")
library(irr)
library(caret)
library(lubridate)
library(ggplot2)
library(dplyr)
library(psych)
library(tidyr)
library(purrr)
library(stringr)
####### PREPROCESSING #######
save_df <- function(df, nombre_archivo = NULL, prefijo = "df_", carpeta = NULL) {
  if (!is.null(carpeta) && !dir.exists(carpeta)) {
    dir.create(carpeta, recursive = TRUE)
  }
  
  if (is.null(nombre_archivo)) {
    timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
    nombre_archivo <- paste0(prefijo, timestamp, ".csv")
  } else if (!grepl("\\.csv$", nombre_archivo)) {
    nombre_archivo <- paste0(nombre_archivo, ".csv")
  }
  
  if (!is.null(carpeta)) {
    ruta_completa <- file.path(carpeta, nombre_archivo)
  } else {
    ruta_completa <- nombre_archivo
  }
  
  write.csv(df, file = ruta_completa, row.names = FALSE, fileEncoding = "UTF-8")
  
  cat("✓ Dataframe guardado:", ruta_completa, "\n")
  cat("  - Dimensiones:", nrow(df), "filas ×", ncol(df), "columnas\n")
  cat("  - Tamaño:", format(object.size(df), units = "auto"), "\n")
  
  return(ruta_completa)
}

fix_invalid_dates <- function(date_vector, format = "%d/%m/%Y") {
  parsed_dates <- as.Date(date_vector, format = format)
  failed_indices <- which(is.na(parsed_dates))
  if (length(failed_indices) == 0) {
    cat("No invalid dates found!\n")
    return(parsed_dates)
  }
  cat("Found", length(failed_indices), "invalid dates to fix\n")
  for (i in failed_indices) {
    original_date <- date_vector[i]
    cat("Processing:", original_date)
    if (grepl("^\\d{1,2}/\\d{1,2}/\\d{4}$", original_date)) {
      parts <- strsplit(original_date, "/")[[1]]
      day <- as.numeric(parts[1])
      month <- as.numeric(parts[2])
      year <- as.numeric(parts[3])
      
      max_day_in_month <- as.numeric(format(
        as.Date(paste(year, month + 1, "01", sep = "-")) - 1, 
        "%d"
      ))
      # Handle December (month 12)
      if (month == 12) {
        max_day_in_month <- as.numeric(format(
          as.Date(paste(year + 1, "01", "01", sep = "-")) - 1, 
          "%d"
        ))
      }
      new_day <- max_day_in_month
      corrected_date_str <- sprintf("%02d/%02d/%04d", new_day, month, year)
      
      corrected_date <- as.Date(corrected_date_str, format = format)
      
      if (!is.na(corrected_date)) {
        parsed_dates[i] <- corrected_date
        cat(" -> Fixed to:", corrected_date_str, "\n")
      } else {
        cat(" -> Still invalid after correction\n")
      }
    } else {
      cat(" -> Unrecognized format, skipping\n")
    }
  }
  return(parsed_dates)
}

preprocess_df <- function(input) {
  output <- read.csv(input, header=TRUE, sep=";", dec = ".", fileEncoding = "UTF-8")
  output <- output[0:22] # cuts to sentiment analysis
  output$date_fixed <- fix_invalid_dates(output$date, "%Y-%m-%d")
  output$date <- output$date_fixed
  output$date_fixed <- NULL
  output$newspaper <- as.factor(output$newspaper)
  output$model_0_topic <- as.factor(output$model_0_topic)
  output$model_0_topic_label <- as.factor(output$model_0_topic_label)
  output$model_1_topic <- as.factor(output$model_1_topic)
  output$model_1_topic_label <- as.factor(output$model_1_topic_label)
  output$model_2_topic <- as.factor(output$model_2_topic)
  output$model_2_topic_label <- as.factor(output$model_2_topic_label)
  output$sabert_headline_label <- as.factor(output$sabert_headline_label)
  output$robertuito_headline_label <- as.factor(output$robertuito_headline_label)
  output$UMUTeam_headline_label <- as.factor(output$UMUTeam_headline_label)
  output$agreed_headline_label <- as.factor(output$agreed_headline_label)
  return(output)
}

df <- preprocess_df("./results_pos_2025-11-09.csv")
summary(df)
head(df)

####### GENERAL RESULTS #######
sentiment_summary <- df %>%
  group_by(agreed_headline_label) %>%
  summarise(
    muestra = n(),
    porcentaje = round(n() / nrow(df) * 100, 4)
  ) %>%
  mutate(
    hwa = sum(muestra) / sum(muestra / porcentaje),
    desviacion = porcentaje - hwa
  ) 
hwa_promedio <- sum(sentiment_summary$muestra) / sum(sentiment_summary$muestra / sentiment_summary$muestra)
hwa_porcentaje <- sum(sentiment_summary$muestra) / sum(sentiment_summary$muestra / sentiment_summary$porcentaje)

datos_desviacion <- sentiment_summary %>%
  mutate(
    tipo_desviacion = ifelse(desviacion > 0, "Por encima del promedio", "Por debajo del promedio"),
    desviacion_absoluta = round(abs(desviacion), 2)
  )
datos_desviacion$newspaper <- c("Correo", "El Comercio", "Gestión", "Ojo", "Perú21", "Publimetro", "Trome")

grafico_desviacion <- ggplot(datos_desviacion, 
                             aes(x = reorder(agreed_headline_label, desviacion), 
                                 y = desviacion, 
                                 fill = tipo_desviacion)) +
  geom_col(alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", size = 0.5) +
  geom_text(aes(label = sprintf("%+.2f%%", desviacion)),
            hjust = ifelse(datos_desviacion$desviacion > 0, 0, 0.5),
            size = 3.25, fontface = "bold") +
  coord_flip() +
  scale_fill_manual(values = c("Por encima del promedio" = "#59A14F", 
                               "Por debajo del promedio" = "#E15759")) +
  labs(title = "Desviación respecto al promedio general",
       subtitle = paste("Línea roja = Promedio general (", sprintf("%.2f%%", hwa_porcentaje), ")"),
       x = "",
       y = "Desviación del promedio (%)",
       fill = "Posición relativa") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom")










sentiment_summary <- sentiment_summary %>%
  bind_rows(
    tibble(
      agreed_headline_label = "Total",
      muestra = sum(.$muestra),
      porcentaje = sum(.$porcentaje), 
      hwa = NA_real_, 
      desviacion = NA_real_
    )
  ) %>%
  bind_rows(
    tibble(
      agreed_headline_label = "Promedio",
      muestra = hwa_promedio,
      porcentaje = hwa_porcentaje, 
      hwa = mean(.$hwa), 
      desviacion = NA_real_
    )
  ) 
sentiment_summary$hwa <- NULL
print(sentiment_summary)
save_df(sentiment_summary, prefijo = "sentiment_summary_")




levels(sentiment_summary$agreed_headline_label) <- c("Negativo", "Neutral", "Positivo")
sentiment_summary <- df %>%
  count(agreed_headline_label) %>%
  mutate(
    percentage = n / sum(n) * 100,
    label = paste0(round(percentage, 1), "%\n", format(n, big.mark = ","), " artículos")
  ) %>%
  arrange(desc(agreed_headline_label))


grafico_general_barras <- ggplot(sentiment_summary, aes(x = agreed_headline_label, y = porcentaje, fill = agreed_headline_label)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = label), 
            vjust = -0.5, 
            color = "black", fontface = "bold", size = 4.5,
            lineheight = 0.8) +
  scale_fill_manual(values = c("Negativo" = "#e74c3c", "Neutral" = "#f39c12", "Positivo" = "#2ecc71")) +
  labs(
    title = "Distribución general del sentimiento en titulares",
    subtitle = "Porcentaje y número absoluto de artículos por categoría de sentimiento",
    x = "Categoría de Sentimiento",
    y = "Porcentaje (%)",
    fill = "Sentimiento"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    legend.position = "none",
    axis.text.x = element_text(size = 11)
  ) +
  ylim(0, max(sentiment_summary$percentage) * 1.2)  # Más espacio para las etiquetas

print(grafico_general_barras)

sentiment_por_periodico <- df %>%
  group_by(newspaper, agreed_headline_label) %>%
  summarise(
    muestra = n(),
    .groups = 'drop'
  ) %>%
  group_by(newspaper) %>%
  mutate(
    muestra_periodico = sum(muestra),
    perc_news = round(muestra / muestra_periodico * 100, 4),
    hwa_perc_news = sum(muestra_periodico) / sum(muestra_periodico / perc_news),
    desv_perc_news = perc_news - hwa_perc_news,
  ) %>%
  group_by(newspaper) %>%
  mutate(
    perc_global = round(muestra_periodico / nrow(df) * 100, 4),
  ) %>% 
  group_by(agreed_headline_label) %>%
  mutate(
    hwa_perc_global = sum(muestra_periodico) / sum(muestra_periodico / perc_global),
    desv_perc_global = perc_global - hwa_perc_global
  )

print(sentiment_por_periodico)
save_df(sentiment_por_periodico, prefijo = "sentiment_por_periodico_")
sum(sentiment_por_periodico$muestra_periodico)
sum(sentiment_por_periodico$muestra) / sum(sentiment_por_periodico$muestra / sentiment_por_periodico$hwa_perc_news)
sum(sentiment_por_periodico$muestra) / sum(sentiment_por_periodico$muestra / sentiment_por_periodico$muestra_periodico)
sum(sentiment_por_periodico$hwa_perc_news)/21

####### CORRELATIONS #######
correlation_detailed <- df %>%
  select(ends_with("_headline_score")) %>%
  corr.test()
cat("\nDetailed Correlation Analysis with P-values:\n")
print(correlation_detailed$r, digits = 3)
cat("\nP-values:\n")
print(correlation_detailed$p, digits = 3)

sentiment_labels <- df %>%
  select(ends_with("_headline_label")) %>%
  mutate(across(everything(), ~factor(., levels = c("Negative", "Neutral", "Positive"))))
cat("\nFleiss' Kappa (all models):\n")
fleiss_data <- as.matrix(sentiment_labels)
fleiss_kappa <- kappam.fleiss(fleiss_data)
print(fleiss_kappa)
cat("\nPercentage Agreement Matrix:\n")
agreement_matrix <- function(labels1, labels2) {
  mean(labels1 == labels2)
}
agree_mat <- matrix(0, nrow = 4, ncol = 4)
rownames(agree_mat) <- colnames(agree_mat) <- names(sentiment_labels)
for (i in 1:4) {
  for (j in 1:4) {
    agree_mat[i,j] <- round(agreement_matrix(sentiment_labels[[i]], sentiment_labels[[j]]), 3)
  }
}
print(agree_mat)

cat("\nConfusion Matrix: Majority Vote vs Individual Models\n")
for (model in names(sentiment_labels)[1:3]) {
  cat("\n", model, "vs Majority Vote:\n")
  cm <- confusionMatrix(sentiment_labels[[model]], sentiment_labels$agreed_headline_label)
  print(cm$table)
  cat("Accuracy:", round(cm$overall["Accuracy"], 3), "\n")
}

####### TIME RESULTS #######
yearly_sentiment <- df %>%
  mutate(year = floor_date(date, "year")) %>%
  group_by(year) %>%
  filter(year < "2020-01-01") %>%
  summarise(
    sentiment_index = mean(case_when(
      agreed_headline_label == "Positive" ~ 1,
      agreed_headline_label == "Negative" ~ -1,
      TRUE ~ 0
    )),
    article_count = n()
  ) %>%
  mutate(
    hwa = sum(article_count) / sum(article_count / sentiment_index),
    desviacion = sentiment_index - hwa,
  )
yearly_sentiment$hwa <- NULL
save_df(yearly_sentiment, prefijo = "yearly_sentiment_")

####### ARCHIVO #######

monthly_sentiment <- df %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(month) %>%
  summarise(
    sentiment_index = mean(case_when(
      agreed_headline_label == "Positive" ~ 1,
      agreed_headline_label == "Negative" ~ -1,
      TRUE ~ 0
    )),
    article_count = n()
  )

yearly_sentiment <- monthly_sentiment %>%
  mutate(year = year(month)) %>%
  group_by(year) %>%
  summarise(
    sentiment_index = mean(sentiment_index, na.rm = TRUE),
    article_count = n()
  )

newspaper_monthly <- df %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(month, newspaper) %>%
  summarise(
    sentiment_index = mean(case_when(
      agreed_headline_label == "Positive" ~ 1,
      agreed_headline_label == "Negative" ~ -1,
      TRUE ~ 0
    ))
  )

newspaper_yearly <- newspaper_monthly %>%
  mutate(year = year(month)) %>%
  group_by(year, newspaper) %>%
  summarise(
    sentiment_index = mean(sentiment_index, na.rm = TRUE)
  )

newspaper_yearly_filtered <- newspaper_yearly %>% filter(year != 2020)
yearly_sentiment_filtered <- yearly_sentiment %>% filter(year != 2020)

levels(newspaper_yearly_filtered$newspaper) <- c("Correo", "El Comercio", "Gestión", "Ojo", "Perú21", "Publimetro", "Trome")

evolucion_sentimiento <- ggplot(newspaper_yearly_filtered, aes(x = year, y = sentiment_index, color = newspaper, group = newspaper)) +
  geom_line(size = 1.2, alpha = 0.8) +
  geom_point(size = 2) +
  geom_line(data = yearly_sentiment_filtered, 
            aes(x = year, y = sentiment_index), 
            size = 1, color = "black", inherit.aes = FALSE) +
  geom_point(data = yearly_sentiment_filtered, 
             aes(x = year, y = sentiment_index), 
             color = "black", size = 1.5, inherit.aes = FALSE) +
  geom_text(data = newspaper_yearly_filtered %>% 
              group_by(newspaper) %>% 
              filter(year == max(year)), 
            aes(label = newspaper), 
            hjust = -0.1, size = 3, check_overlap = TRUE, nudge_x = 0.1) +
  geom_text(data = yearly_sentiment_filtered %>% filter(year == max(year)), 
            aes(x = year, y = sentiment_index, label = "Promedio General"), 
            hjust = -0.1, vjust = -0.5, color = "black", fontface = "bold", 
            size = 3.5, inherit.aes = FALSE, nudge_x = 0.1) +
  scale_x_continuous(
    breaks = seq(min(newspaper_yearly_filtered$year), max(newspaper_yearly_filtered$year), by = 1),
    limits = c(min(newspaper_yearly_filtered$year), max(newspaper_yearly_filtered$year) + 1.5)
  ) +
  scale_y_continuous(
    name = "Índice de Sentimiento",
    limits = c(-1, 1)
  ) +
  scale_color_discrete(name = "Periódico") +
  labs(
    title = "Evolución anual del sentimiento en titulares de artículos (2013-2019)",
    subtitle = "Línea negra = Promedio general anual de todos los periódicos\nÍndice: -1 (Negativo) a +1 (Positivo)",
    x = "Año",
    y = "Índice de Sentimiento"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "none"
  )
print(evolucion_sentimiento)

####### ABSOLUTE/RELATIVE RESULTS #######
newspaper_sentiment <- df %>%
  group_by(newspaper, agreed_headline_label) %>%
  summarise(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)
levels(newspaper_sentiment$agreed_headline_label) <- c("Negativo", "Neutral", "Positivo")
levels(newspaper_sentiment$newspaper) <- c("Correo", "El Comercio", "Gestión", "Ojo", "Perú21", "Publimetro", "Trome")

grafico_periodicos <- ggplot(newspaper_sentiment, aes(x = newspaper, y = percentage, fill = agreed_headline_label)) +
  geom_bar(stat = "identity", position = "fill") +
  geom_text(aes(label = paste0(round(percentage, 1), "%\n(", count, ")")), 
            position = position_fill(vjust = 0.5),
            size = 3.5, color = "white", fontface = "bold", lineheight = 0.8) +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_manual(values = c("Negativo" = "#e74c3c", "Neutral" = "#f39c12", "Positivo" = "#2ecc71")) +
  labs(
    title = "Distribución del sentimiento por periódico",
    subtitle = "Porcentaje y número absoluto de artículos por categoría de sentimiento",
    x = "Periódico",
    y = "Porcentaje",
    fill = "Sentimiento"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

print(grafico_periodicos)

# Calcular desviaciones por periódico
general_avg <- df %>%
  count(agreed_headline_label, name = "frecuencia") %>%
  mutate(
    avg_percentage = frecuencia / sum(frecuencia) * 100,
    ) %>%
  select(agreed_headline_label, avg_percentage)

levels(general_avg$agreed_headline_label) <- c("Negativo", "Neutral", "Positivo")

newspaper_deviation <- newspaper_sentiment %>%
  left_join(general_avg, by = "agreed_headline_label") %>%
  mutate(
    deviation = percentage - avg_percentage,
    deviation_label = ifelse(deviation >= 0, 
                             paste0("+", round(deviation, 1), "%"), 
                             paste0(round(deviation, 1), "%"))
  )

print(newspaper_deviation, n=100)
save_df(newspaper_deviation, prefijo="newspaper_deviation_")

# Heatmap de desviaciones
heatmap_desviaciones <- ggplot(newspaper_deviation, 
                               aes(x = newspaper, y = agreed_headline_label, fill = deviation)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = deviation_label), 
            color = "black", fontface = "bold", size = 3.5) +
  scale_fill_gradient2(low = "#a2ab58", mid = "white", high = "#a2ab58", 
                       midpoint = 0, 
                       name = "Desviación (%)") +
  labs(
    title = "Desviación del sentimiento respecto al promedio general",
    subtitle = "Porcentaje de diferencia por periódico y categoría de sentimiento",
    x = "Periódico",
    y = "Categoría de sentimiento"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  )
print(heatmap_desviaciones)

# Gráfico de barras de desviaciones con mejor separación
barras_desviaciones <- ggplot(newspaper_deviation, 
                              aes(x = newspaper, y = deviation, fill = agreed_headline_label)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  # Líneas verticales entre periódicos
  geom_vline(xintercept = seq(1.5, length(unique(newspaper_deviation$newspaper)) - 0.5, 1), 
             linetype = "solid", color = "gray80", size = 0.7) +
  geom_text(aes(label = deviation_label, 
                y = deviation + ifelse(deviation >= 0, 0.8, -0.8)), 
            position = position_dodge(width = 0.8),
            size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("Negativo" = "#e74c3c", "Neutral" = "#f39c12", "Positivo" = "#2ecc71")) +
  scale_x_discrete(expand = expansion(mult = 0.1)) +  # Más espacio en los extremos
  labs(
    title = "Desviación del sentimiento por periódico",
    subtitle = "Diferencia porcentual respecto al promedio general",
    x = "Periódico",
    y = "Desviación (%)",
    fill = "Sentimiento"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1, margin = margin(t = 5)),
    panel.grid.major.x = element_blank(),  # Remover grid vertical
    panel.grid.minor.x = element_blank(),
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)  # Márgenes adecuados
  ) +
  ylim(min(newspaper_deviation$deviation) - 2, max(newspaper_deviation$deviation) + 2)  # Espacio para etiquetas
print(barras_desviaciones)

resumen_desviaciones <- newspaper_deviation %>%
  group_by(newspaper) %>%
  summarise(
    desviacion_promedio_absoluta = mean(abs(deviation)),
    desviacion_maxima = max(deviation),
    desviacion_minima = min(deviation)
  ) %>%
  arrange(desc(desviacion_promedio_absoluta))

cat("Periódicos ordenados por mayor desviación promedio:\n")
print(resumen_desviaciones)

####### TOP10 TOPICS #######
summary(df)
topics_normalized <- topics_combined %>%
  count(newspaper, topic_label) %>%
  group_by(newspaper) %>%
  mutate(total_periodico = sum(n)) %>%
  ungroup() %>%
  mutate(proporcion = n / total_periodico)


top_10_normalized <- topics_normalized %>%
  group_by(topic_label) %>%
  summarise(
    frecuencia_total = sum(n),
    proporcion_promedio = weighted.mean(proporcion, total_periodico),
    n_periodicos = n_distinct(newspaper)
  ) %>%
  arrange(desc(proporcion_promedio)) %>%
  head(10)


topics_combined <- df %>%
  select(newspaper, 
         model_0_topic_label, 
         model_1_topic_label, 
         model_2_topic_label) %>%
  pivot_longer(
    cols = c(model_0_topic_label, model_1_topic_label, model_2_topic_label),
    names_to = "model",
    values_to = "topic_label"
  ) %>%
  filter(topic_label != "Unknown", !is.na(topic_label))  # Excluir temas desconocidos
top_topics <- topics_combined %>%
  count(topic_label, sort = TRUE) %>%
  head(20)
top_10_topics <- top_topics$topic_label
top_10_topics <- c(top_10_topics[0:4], top_10_topics[6], top_10_topics[8], top_10_topics[10:11], top_10_topics[13:14])

# Filtrar los temas seleccionados
selected_topics <- topics_combined %>%
  filter(topic_label %in% top_10_topics)

# Contar frecuencia de los temas seleccionados
selected_topics_count <- selected_topics %>%
  count(topic_label, sort = TRUE)

# Crear el gráfico de barras
grafico_top10 <- ggplot(selected_topics_count, aes(x = reorder(topic_label, n), y = n)) +
  geom_bar(stat = "identity", fill = "#3498db", alpha = 0.8) +
  geom_text(aes(label = n), 
            hjust = -0.2, 
            color = "black", fontface = "bold", size = 4) +
  coord_flip() +
  labs(
    title = "Top 10 temas más frecuentes",
    subtitle = "Selección basada en los temas más frecuentes",
    x = "Tema",
    y = "Número de Artículos"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.text.y = element_text(size = 9),
    panel.grid.major.y = element_blank()
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))  # Espacio para las etiquetas

print(grafico_top10)

# Cross-tabulation general entre temas top 10 y sentimiento
cross_sentiment_topics <- df %>%
  # Filtrar solo los temas top 10
  filter(
    model_0_topic_label %in% top_10_topics |
      model_1_topic_label %in% top_10_topics |
      model_2_topic_label %in% top_10_topics
  ) %>%
  # Combinar los tres modelos de temas
  pivot_longer(
    cols = c(model_0_topic_label, model_1_topic_label, model_2_topic_label),
    names_to = "model",
    values_to = "topic"
  ) %>%
  filter(topic %in% top_10_topics) %>%
  # Contar por tema y sentimiento
  count(topic, agreed_headline_label) %>%
  group_by(topic) %>%
  mutate(
    total_topic = sum(n),
    percentage = n / total_topic * 100
  ) %>%
  ungroup() %>%
  # Seleccionar y ordenar
  select(topic, agreed_headline_label, n, percentage) %>%
  arrange(topic, agreed_headline_label)

# Crear tabla pivote para visualización
cross_table_general <- cross_sentiment_topics %>%
  mutate(label = paste0(round(percentage, 1), "%\n(", n, ")")) %>%
  select(topic, agreed_headline_label, label) %>%
  pivot_wider(
    names_from = agreed_headline_label,
    values_from = label,
    values_fill = "0%\n(0)"
  )

cat("Cross-table: Top 10 Temas vs Sentimiento (General)\n")
print(cross_table_general)

# Preparar datos para el gráfico desde cross_table_general
datos_grafico <- cross_table_general %>%
  # Extraer números y porcentajes
  mutate(
    # Extraer porcentajes
    pct_neg = as.numeric(str_extract(Negative, "\\d+\\.?\\d*(?=%)")),
    pct_neu = as.numeric(str_extract(Neutral, "\\d+\\.?\\d*(?=%)")),
    pct_pos = as.numeric(str_extract(Positive, "\\d+\\.?\\d*(?=%)")),
    # Extraer números absolutos
    n_neg = as.numeric(str_extract(Negative, "(?<=\\()\\d+(?=\\))")),
    n_neu = as.numeric(str_extract(Neutral, "(?<=\\()\\d+(?=\\))")),
    n_pos = as.numeric(str_extract(Positive, "(?<=\\()\\d+(?=\\))"))
  ) %>%
  # Calcular total por tema
  mutate(total_tema = n_neg + n_neu + n_pos) %>%
  # Convertir a formato largo
  pivot_longer(
    cols = c(pct_neg, pct_neu, pct_pos),
    names_to = "sentimiento_tipo",
    values_to = "porcentaje"
  ) %>%
  mutate(
    sentimiento = case_when(
      sentimiento_tipo == "pct_neg" ~ "Negative",
      sentimiento_tipo == "pct_neu" ~ "Neutral",
      sentimiento_tipo == "pct_pos" ~ "Positive"
    ),
    n_articulos = case_when(
      sentimiento_tipo == "pct_neg" ~ n_neg,
      sentimiento_tipo == "pct_neu" ~ n_neu,
      sentimiento_tipo == "pct_pos" ~ n_pos
    ),
    etiqueta = paste0(round(porcentaje, 1), "%\n(", n_articulos, ")")
  )

grafico_top10_sentimiento <- datos_grafico %>%
  ggplot(aes(x = reorder(tema_corto, total_tema), y = porcentaje, fill = sentimiento)) +
  geom_col(alpha = 0.8, position = "fill") +
  geom_text(aes(label = ifelse(porcentaje > 5, etiqueta, "")), 
            position = position_fill(vjust = 0.5),
            size = 3, color = "white", fontface = "bold", lineheight = 0.8) +
  coord_flip() +
  scale_fill_manual(values = c("Negative" = "#e74c3c", "Neutral" = "#f39c12", "Positive" = "#2ecc71")) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Distribución de sentimiento en los 10 temas más frecuentes",
    subtitle = "Cada barra representa el 100% del sentimiento para ese tema",
    x = "",
    y = "Porcentaje",
    fill = "Sentimiento"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.text.y = element_text(size = 10),
    legend.position = "bottom"
  )

print(grafico_top10_sentimiento)

####### TOP3 TOPICS (NEWSPAPER) #######
procesar_topics_con_sentimiento <- function(df, periodico, n_topics = 10, excluir_indices_model0 = NULL, excluir_indices_model1 = NULL, excluir_indices_model2 = NULL) {
  
  # Filtrar por el periódico específico
  df_filtrado <- df %>% filter(newspaper == periodico)
  
  # Función auxiliar para procesar cada modelo con sentimiento
  procesar_modelo_con_sentimiento <- function(df, topic_id_col, topic_label_col, modelo_nombre, excluir_indices = NULL) {
    
    # Primero obtener los temas más frecuentes
    topics <- df %>%
      filter({{topic_id_col}} != -1) %>%
      filter(!grepl("UNKNOWN|unknown|^\\s*$", {{topic_label_col}})) %>%
      filter(!is.na({{topic_label_col}})) %>%
      filter({{topic_label_col}} != "") %>%
      count({{topic_label_col}}, sort = TRUE) %>%
      mutate(
        topic_label_clean = stringr::str_trim({{topic_label_col}}),
        topic_label_wrap = stringr::str_wrap(topic_label_clean, 40),
        indice = row_number(),
        modelo = modelo_nombre
      )
    
    # Aplicar exclusión por índices si se especifica
    if (!is.null(excluir_indices)) {
      topics <- topics %>% filter(!indice %in% excluir_indices)
    }
    
    topics_top <- topics %>% head(n_topics)
    
    # Ahora calcular sentimiento para cada tema top
    sentimiento_por_tema <- map_df(topics_top$topic_label_clean, function(tema) {
      df_tema <- df_filtrado %>% filter({{topic_label_col}} == tema)
      
      if(nrow(df_tema) > 0) {
        sentimiento <- df_tema %>%
          count(agreed_headline_label) %>%
          mutate(
            porcentaje = n / sum(n) * 100,
            tema = tema,
            modelo = modelo_nombre
          )
        return(sentimiento)
      } else {
        return(NULL)
      }
    })
    
    # Combinar temas con su sentimiento - renombrar columnas para evitar duplicados
    topics_top_renamed <- topics_top %>%
      rename(articulos_tema = n)
    
    sentimiento_renamed <- sentimiento_por_tema %>%
      rename(articulos_sentimiento = n)
    
    resultados <- topics_top_renamed %>%
      left_join(sentimiento_renamed, by = c("topic_label_clean" = "tema", "modelo" = "modelo")) %>%
      mutate(periodico = periodico)
    
    return(resultados)
  }
  
  # Procesar los tres modelos
  topics_model0 <- procesar_modelo_con_sentimiento(df_filtrado, model_0_topic, model_0_topic_label, "Modelo 0", excluir_indices_model0)
  topics_model1 <- procesar_modelo_con_sentimiento(df_filtrado, model_1_topic, model_1_topic_label, "Modelo 1", excluir_indices_model1)
  topics_model2 <- procesar_modelo_con_sentimiento(df_filtrado, model_2_topic, model_2_topic_label, "Modelo 2", excluir_indices_model2)
  
  # Combinar resultados
  resultados <- bind_rows(topics_model0, topics_model1, topics_model2)
  
  return(resultados)
}

# Ejecutar la función corregida
topics_elcomercio <- procesar_topics_con_sentimiento(
  economia_informal_df, 
  periodico = "elcomercio",
  excluir_indices_model0 = c(3, 4),
  excluir_indices_model1 = c(1,2,4,6),
  excluir_indices_model2 = c(3,4,5,6,7,8,9,10,11,12,13),
  n_topics = 3
)

topics_elcomercio <- procesar_topics_con_sentimiento(
  df, 
  periodico = "correo",
  #excluir_indices_model0 = c(3, 4),
  excluir_indices_model1 = c(3),
  #excluir_indices_model2 = c(3,4,5,6,7,8,9,10,11,12,13),
  n_topics = 3
)

topics_gestion<- procesar_topics_con_sentimiento(
  df, 
  periodico = "gestion",
  excluir_indices_model0 = c(3),
  excluir_indices_model1 = c(1,3,4,5),
  #excluir_indices_model2 = c(),
  n_topics = 3
)

topics_ojo <- procesar_topics_con_sentimiento(
  df, 
  periodico = "ojo",
  #excluir_indices_model0 = c(3),
  excluir_indices_model1 = c(2),
  excluir_indices_model2 = c(1,3),
  n_topics = 3
)

topics_trome <- procesar_topics_con_sentimiento(
  df, 
  periodico = "trome",
  #excluir_indices_model0 = c(),
  excluir_indices_model1 = c(2,3,4),
  excluir_indices_model2 = c(1,3),
  n_topics = 3
)

topics_peru21 <- procesar_topics_con_sentimiento(
  df, 
  periodico = "peru21",
  excluir_indices_model0 = c(2),
  excluir_indices_model1 = c(2,3,4),
  #excluir_indices_model2 = c(1,3),
  n_topics = 3
)

topics_publimetro <- procesar_topics_con_sentimiento(
  df, 
  periodico = "publimetro",
  #excluir_indices_model0 = c(),
  #excluir_indices_model1 = c(2,3,4),
  #excluir_indices_model2 = c(1,3),
  n_topics = 3
)

# Ver los resultados
cat("Temas de El Comercio con análisis de sentimiento:\n")
print(topics_elcomercio_sentimiento)

# Crear una tabla resumen más legible

resumen_elcomercio <- topics_publimetro %>%
  select(modelo, topic_label_clean, articulos_tema, agreed_headline_label, porcentaje, articulos_sentimiento) %>%
  pivot_wider(
    names_from = agreed_headline_label,
    values_from = c(articulos_sentimiento, porcentaje),
    names_sep = "_"
  ) %>%
  mutate(
    total_articulos = articulos_tema,
    etiqueta_neg = ifelse(!is.na(porcentaje_Negative), paste0(round(porcentaje_Negative, 1), "%\n(", articulos_sentimiento_Negative, ")"), "-"),
    etiqueta_neu = ifelse(!is.na(porcentaje_Neutral), paste0(round(porcentaje_Neutral, 1), "%\n(", articulos_sentimiento_Neutral, ")"), "-"),
    etiqueta_pos = ifelse(!is.na(porcentaje_Positive), paste0(round(porcentaje_Positive, 1), "%\n(", articulos_sentimiento_Positive, ")"), "-")
  ) %>%
  select(modelo, topic_label_clean, total_articulos, etiqueta_neg, etiqueta_neu, etiqueta_pos)

cat("\nResumen para El Comercio:\n")
print(resumen_elcomercio)

datos_porcentaje <- resumen_elcomercio %>%
  # Extraer los porcentajes de las etiquetas
  mutate(
    pct_neg = as.numeric(str_extract(etiqueta_neg, "\\d+\\.?\\d*(?=%)")),
    pct_neu = as.numeric(str_extract(etiqueta_neu, "\\d+\\.?\\d*(?=%)")), 
    pct_pos = as.numeric(str_extract(etiqueta_pos, "\\d+\\.?\\d*(?=%)")),
    pct_neg = ifelse(is.na(pct_neg), 0, pct_neg),
    pct_neu = ifelse(is.na(pct_neu), 0, pct_neu),
    pct_pos = ifelse(is.na(pct_pos), 0, pct_pos)
  ) %>%
  select(modelo, topic_label_clean, total_articulos, pct_neg, pct_neu, pct_pos) %>%
  pivot_longer(
    cols = c(pct_neg, pct_neu, pct_pos),
    names_to = "sentimiento_tipo",
    values_to = "porcentaje"
  ) %>%
  mutate(
    sentimiento = case_when(
      sentimiento_tipo == "pct_neg" ~ "Negative",
      sentimiento_tipo == "pct_neu" ~ "Neutral",
      sentimiento_tipo == "pct_pos" ~ "Positive"
    )
  )

datos_combinados <- datos_porcentaje %>%
  left_join(
    resumen_elcomercio %>% select(topic_label_clean, etiqueta_neg, etiqueta_neu, etiqueta_pos),
    by = "topic_label_clean"
  ) %>%
  mutate(
    etiqueta_completa = case_when(
      sentimiento == "Negative" ~ etiqueta_neg,
      sentimiento == "Neutral" ~ etiqueta_neu,
      sentimiento == "Positive" ~ etiqueta_pos
    )
  )

grafico_completo <- datos_combinados %>%
  ggplot(aes(x = reorder(topic_label_clean, total_articulos), y = porcentaje, fill = sentimiento)) +
  geom_col(alpha = 0.8, position = "fill") +
  geom_text(aes(label = ifelse(porcentaje > 5, etiqueta_completa, "")), 
            position = position_fill(vjust = 0.5),
            size = 2.8, color = "white", fontface = "bold", lineheight = 0.8) +
  coord_flip() +
  facet_wrap(~modelo, ncol = 1, scales = "free_y") +
  scale_fill_manual(values = c("Negative" = "#e74c3c", "Neutral" = "#f39c12", "Positive" = "#2ecc71")) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Distribución de sentimiento en top 3 campos semánticos",
    subtitle = "Periódico: Publimetro - Cada barra representa el 100% del sentimiento para ese tema",
    x = "",
    y = "Porcentaje",
    fill = "Sentimiento"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 10),
    axis.text.y = element_text(size = 8)
  )

print(grafico_completo)
