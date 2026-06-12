####### IMPORTS ####### 
if (!require(irr)) install.packages("irr")
if (!require(caret)) install.packages("caret")
if (!require(treemap)) install.packages("treemap")
library(irr)
library(caret)
library(lubridate)
library(ggplot2)
library(dplyr)
library(psych)
library(tidyr)
library(purrr)
library(stringr)
#library(waffle)
library(treemap)

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
# levels(df$newspaper) <- c("Correo", "El Comercio", "Gestión", "Ojo", "Perú21", "Publimetro", "Trome")
pos_columns <- names(df)[grepl("^pos_", names(df))]
df <- df %>%
  mutate(across(all_of(pos_columns), as.factor))

summary(df[0:22])
summary(df[23:30])

####### GENERAL RESULTS #######
frecuencia_apariciones <- df %>%
  summarise(across(starts_with("pos_"), ~sum(. == 1))) %>%
  pivot_longer(everything(), names_to = "construccion", values_to = "n_articulos")
rango_frecuencia <- frecuencia_apariciones %>%
  count(rango_frecuencia = cut(n_articulos, 
                               breaks = c(-1, 0, 1, 2, 5, 10, Inf),
                               labels = c("0", "1", "2", "3-5", "6-10", "11+"))) %>%
  mutate(
    porcentaje = n / sum(n) * 100,
    hwa = sum(n) / sum(n / porcentaje),
    desviacion = porcentaje - hwa
    ) %>%
  print(n = Inf)
rango_frecuencia$hwa <- NULL
save_df(rango_frecuencia, prefijo = "rango_frecuencia_")
sum(rango_frecuencia$n)
sum(rango_frecuencia$n) / sum(rango_frecuencia$n / rango_frecuencia$n)

# Crear el histograma por rangos de frecuencia
histograma_rangos <- ggplot(rango_frecuencia, aes(x = rango_frecuencia, y = n)) +
  geom_col(fill = "steelblue", alpha = 0.8, width = 0.7) +
  geom_text(aes(label = paste0(n, "\n(", round(porcentaje, 3), "%)")), 
            vjust = -0.3, size = 3.5, fontface = "bold") +
  labs(title = "Distribución de construcciones sintácticas por rango de repetición",
       subtitle = paste("Número total de construcciones:", sum(rango_frecuencia$n)),
       x = "Rango de repeticiones",
       y = "Frecuencia (número de casos)") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray40"),
    axis.text.x = element_text(size = 10, angle = 0, hjust = 0.5),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.y = element_blank()
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) # Espacio para las etiquetas
print(histograma_rangos)



conteo_rangos <- frecuencia_apariciones %>%
  count(rango_frecuencia = cut(n_articulos, 
                               breaks = c(0, 1, 3, 5, 10),
                               labels = c("1", "2-3", "4-5", "6-10"),
                               include.lowest = TRUE)) %>%
  mutate(porcentaje = n / sum(n) * 100)

histograma_con_porcentajes <- ggplot(conteo_rangos, aes(x = rango_frecuencia, y = n)) +
  geom_histogram(stat = "identity", fill = "#3498db", alpha = 0.8, 
                 color = "white", binwidth = 1) +
  geom_text(aes(label = paste0(n, "\n(", round(porcentaje, 1), "%)")), 
            vjust = -0.5, fontface = "bold", size = 3.5, lineheight = 0.8) +
  labs(
    title = "Histograma de construcciones POS",
    subtitle = "Distribución del número de construcciones en cada rango",
    x = "Rango de frecuencia (número de artículos)",
    y = "Número de construcciones POS"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 10)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))
print(histograma_con_porcentajes)

####### Tabla Newspaper #######
pos_por_periodico <- df %>%
  select(newspaper, starts_with("pos_")) %>%
  pivot_longer(
    cols = starts_with("pos_"),
    names_to = "construccion",
    values_to = "presente"
  ) %>%
  filter(presente == 1) %>%
  count(newspaper, name = "total_pos") %>%
  mutate(
    porcentaje = total_pos / sum(total_pos),
    wha = sum(total_pos) / sum(total_pos / porcentaje),
    desviacion = porcentaje - wha
  )
save_df(pos_por_periodico, prefijo = "pos_por_periodico_")
mean(pos_por_periodico$total_pos) / sum(pos_por_periodico$total_pos - pos_por_periodico$total_pos)

pos_por_periodico <- df %>%
  select(newspaper, starts_with("pos_")) %>%
  pivot_longer(
    cols = starts_with("pos_"),
    names_to = "construccion",
    values_to = "presente"
  ) %>%
  filter(presente == 1) %>%
  count(newspaper, name = "total_pos") %>%
  mutate(
    porcentaje = total_pos / sum(total_pos) * 100,  # Convertir a porcentaje
    wha = sum(total_pos) / sum(total_pos / porcentaje),
    desviacion = porcentaje - wha,
    desviacion_label = ifelse(desviacion >= 0, 
                              paste0("+", round(desviacion, 1)), 
                              as.character(round(desviacion, 1)))
  )
ggplot(pos_por_periodico, 
       aes(x = newspaper, y = desviacion, fill = newspaper)) +
  geom_bar(stat = "identity", width = 0.7, alpha = 0.8) +
  # Líneas verticales entre periódicos
  geom_vline(xintercept = seq(1.5, length(unique(pos_por_periodico$newspaper)) - 0.5, 1), 
             linetype = "solid", color = "gray80", size = 0.7) +
  geom_text(aes(label = desviacion_label, 
                y = desviacion + ifelse(desviacion >= 0, 0.5, -0.5)), 
            size = 3.5, fontface = "bold") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", size = 0.5) +
  scale_x_discrete(expand = expansion(mult = 0.1)) +
  labs(
    title = "Desviación de construcciones sintácticas por periódico",
    subtitle = paste("Diferencia porcentual respecto al promedio general (", 
                     round(unique(pos_por_periodico$wha), 1), "%)"),
    x = "Periódico",
    y = "Desviación (%)",
    fill = "Periódico",
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1, margin = margin(t = 5)),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    legend.position = "none",  # Ocultar leyenda ya que los colores están en el eje X
    plot.margin = margin(10, 10, 10, 10)
  ) +
  ylim(min(pos_por_periodico$desviacion) - 1, max(pos_por_periodico$desviacion) + 1)

# Crear tabla de POS por rango y periódico
pos_rango_periodico <- df %>%
  select(newspaper, starts_with("pos_")) %>%
  pivot_longer(
    cols = starts_with("pos_"),
    names_to = "construccion",
    values_to = "presente"
  ) %>%
  filter(presente == 1) %>%
  count(newspaper, construccion, name = "frecuencia") %>%
  mutate(
    rango = case_when(
      frecuencia == 1 ~ "1",
      frecuencia == 2 ~ "2", 
      frecuencia >= 3 & frecuencia <= 5 ~ "3-5",
      frecuencia >= 6 & frecuencia <= 10 ~ "6-10",
      TRUE ~ "11+"
    ),
    rango = factor(rango, levels = c("1", "2", "3-5", "6-10", "11+"))
  ) %>%
  count(newspaper, rango, name = "muestras") %>%
  group_by(newspaper) %>%
  mutate(
    total_periodico = sum(muestras),
    porcentaje_interno = round(muestras / total_periodico * 100, 2),
    hwa_interno = sum(total_periodico) / sum(total_periodico / porcentaje_interno),
    desviacion_interna = porcentaje_interno - hwa_interno
  ) 

frecuencia_apariciones_periodico <- df %>%
  select(newspaper, starts_with("pos_")) %>%
  pivot_longer(
    cols = starts_with("pos_"),
    names_to = "construccion",
    values_to = "presente"
  ) %>%
  filter(presente == 1) %>%
  count(newspaper, construccion, name = "n_articulos")

rango_frecuencia_periodico <- frecuencia_apariciones_periodico %>%
  mutate(
    rango = cut(n_articulos,
                breaks = c(-1, 0, 1, 2, 5, 10, Inf),
                labels = c("0", "1", "2", "3-5", "6-10", "11+"))
  ) %>%
  count(newspaper, rango, name = "n_construcciones") %>%
  group_by(newspaper) %>%
  mutate(
    total_construcciones = sum(n_construcciones),
    porcentaje = n_construcciones / total_construcciones * 100
  ) %>%
  ungroup()
hwa_por_rango <- rango_frecuencia_periodico %>%
  group_by(rango) %>%
  summarise(
    hwa_rango = sum(n_construcciones) / sum(n_construcciones / porcentaje)
  )
rango_frecuencia_periodico <- rango_frecuencia_periodico %>%
  left_join(hwa_por_rango, by = "rango") %>%
  mutate(
    desviacion = porcentaje - hwa_rango,
    desviacion_label = ifelse(desviacion >= 0, 
                              paste0("+", round(desviacion, 2)), 
                              as.character(round(desviacion, 2)))
  )
# Gráfico de desviación por periódico y rango
ggplot(rango_frecuencia_periodico, 
       aes(x = rango, y = desviacion, fill = newspaper)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_vline(xintercept = seq(1.5, length(unique(rango_frecuencia_periodico$rango)) - 0.5, 1), 
             linetype = "solid", color = "gray80", size = 0.7) +
  geom_text(aes(label = desviacion_label,
                y = desviacion + ifelse(desviacion >= 0, 0, -1)),
            position = position_dodge(width = 0.8),
            size = 2.8, fontface = "bold") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", size = 0.5) +
  labs(
    title = "Desviación de distribución de construcciones por repetición y periódico",
    subtitle = "Comparación con el promedio general de cada rango de repetición",
    x = "Rango de repetición",
    y = "Desviación (%)",
    fill = "Periódico",
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    legend.position = "bottom"
  )

pos_rango_periodico <- pos_rango_periodico %>%
  group_by(newspaper, rango) %>%
  mutate(
    perc_global = round(total_periodico / sum(pos_rango_periodico$muestras) * 100, 4),
  ) %>% 
  ungroup() %>%
  mutate(
    hwa_global = sum(total_periodico) / sum(total_periodico / perc_global),
    desviacion_global = perc_global - hwa_global
  )
print(pos_rango_periodico)
save_df(pos_rango_periodico, prefijo = "pos_rango_periodico_")

sum(pos_rango_periodico$muestras) / sum(pos_rango_periodico$muestras - pos_rango_periodico$total_periodico)
mean(pos_rango_periodico$hwa_global)


####### TOP10 #######
top10_pos <- frecuencia_apariciones %>%
  arrange(desc(n_articulos)) %>%
  head(10) %>%
  mutate(
    porcentaje_total = n_articulos / nrow(df) * 100,
    porcentaje_construcciones = n_articulos / sum(frecuencia_apariciones$n_articulos) * 100,
    rank = row_number()
  )
top10_pos_codes <- top10_pos$construccion
todas_pos_columns <- names(df)[grepl("^pos_", names(df))]
columnas_a_eliminar <- setdiff(todas_pos_columns, top10_pos_codes)
df_top10_columnas <- df %>%
  select(-all_of(columnas_a_eliminar))
df_top10 <- df_top10_columnas %>%
  filter(if_any(all_of(top10_pos_codes), ~ . == 1))
write.csv(df_top10, "top10_pos_metadata.csv", row.names = FALSE, fileEncoding = "UTF-8")

####### TOP10 x SENT #######
top10_sentimiento <- map_df(top10_pos$construccion, function(pos_col) {
  df %>%
    filter(.data[[pos_col]] == 1) %>%
    count(agreed_headline_label) %>%
    mutate(
      construccion = pos_col,
      total_construccion = sum(n),
      porcentaje = n / total_construccion * 100
    ) %>%
    select(construccion, agreed_headline_label, n, porcentaje)
})
top10_sentimiento_table <- top10_sentimiento %>%
  mutate(etiqueta = paste0(round(porcentaje, 1), "%\n(", n, ")")) %>%
  select(construccion, agreed_headline_label, etiqueta) %>%
  pivot_wider(
    names_from = agreed_headline_label,
    values_from = etiqueta,
    values_fill = "0%\n(0)"
  ) %>%
  left_join(top10_pos %>% select(construccion, n_articulos), by = "construccion") %>%
  arrange(desc(n_articulos)) %>%
  mutate(
    construccion_limpia = str_remove(construccion, "^pos_"),
    rank = row_number()
  ) %>%
  select(rank, construccion_limpia, n_articulos, Negative, Neutral, Positive)

####### TOP10 x NEWSPAPER #######
top10_newspaper <- map_df(top10_pos$construccion, function(pos_col) {
  df %>%
    filter(.data[[pos_col]] == 1) %>%
    count(newspaper) %>%
    mutate(
      construccion = pos_col,
      total_construccion = sum(n),
      porcentaje = n / total_construccion * 100
    ) %>%
    select(construccion, newspaper, n, porcentaje)
})

# Crear tabla pivote del top 10 por periódico
top10_newspaper_table <- top10_newspaper %>%
  mutate(etiqueta = paste0(round(porcentaje, 1), "%\n(", n, ")")) %>%
  select(construccion, newspaper, etiqueta) %>%
  pivot_wider(
    names_from = newspaper,
    values_from = etiqueta,
    values_fill = "0%\n(0)"
  ) %>%
  left_join(top10_pos %>% select(construccion, n_articulos), by = "construccion") %>%
  arrange(desc(n_articulos)) %>%
  mutate(
    construccion_limpia = str_remove(construccion, "^pos_"),
    rank = row_number()
  ) %>%
  select(rank, construccion_limpia, n_articulos, "El Comercio", "Correo", "Gestión", "Ojo", "Perú21", "Publimetro", "Publimetro")

cat("TOP 10 CONSTRUCCIONES POS POR PERIÓDICO:\n")
cat("========================================\n")
print(top10_newspaper_table, n = 10)

####### Distribucion x NEWSPAPER #######
distribucion_pos_periodico <- df %>%
  select(newspaper, starts_with("pos_")) %>%
  pivot_longer(
    cols = starts_with("pos_"),
    names_to = "construccion",
    values_to = "presente"
  ) %>%
  filter(presente == 1) %>%
  count(newspaper) %>%
  mutate(
    porcentaje = n / sum(n) * 100,
  )

sum(distribucion_pos_periodico$n)

distribucion_pos_periodico <- distribucion_pos_periodico %>%
  mutate(
    desviacion = porcentaje - mean(distribucion_pos_periodico$porcentaje) 
  )

grafico_absoluto <- ggplot(distribucion_pos_periodico, aes(x = reorder(newspaper, -n), y = n)) +
  geom_bar(stat = "identity", fill = "#3498db", alpha = 0.8) +
  geom_text(aes(label = paste0(n, "\n(", round(porcentaje, 1), "%)")),
            vjust = -0.5, fontface = "bold", size = 4, lineheight = 0.8) +
  labs(
    title = "Distribución total de construcciones POS por periódico",
    subtitle = "Número absoluto y porcentaje del total de 7102 apariciones",
    x = "Periódico",
    y = "Número de construcciones POS"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 10)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))
print(grafico_absoluto)

grafico_desviacion <- distribucion_pos_periodico %>%
  ggplot(aes(x = reorder(newspaper, -desviacion), y = desviacion, fill = desviacion)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 1) +
  geom_text(aes(label = paste0(ifelse(desviacion > 0, "+", ""), round(desviacion, 1), "%"),
                y = desviacion + ifelse(desviacion > 0, 0.5, -0.5)),
            fontface = "bold", size = 4) +
  scale_fill_gradient2(low = "#e74c3c", mid = "#f39c12", high = "#2ecc71", 
                       midpoint = 0, name = "Desviación (%)") +
  labs(
    title = "Desviación en el uso de construcciones POS por periódico",
    subtitle = "Diferencia porcentual respecto a la distribución esperada",
    x = "Periódico",
    y = "Desviación (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

print(grafico_desviacion)


      