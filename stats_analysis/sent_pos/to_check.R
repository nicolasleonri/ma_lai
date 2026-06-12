# install.packages("dplyr")
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
# install.packages("lubridate")
library(lubridate)
install.packages("patchwork")
library(patchwork)

############# Functions ########################
fix_invalid_dates <- function(date_vector, format = "%d/%m/%Y") {
  # Fix invalid dates by going back one day
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
      
      # Instead of just going back one day, find the last valid day of the month
      # Get the maximum valid day for this month/year combination
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
      # Set the day to the maximum valid day for that month
      new_day <- max_day_in_month
      # Create the corrected date string
      corrected_date_str <- sprintf("%02d/%02d/%04d", new_day, month, year)
      
      # Try to parse the corrected date
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
  output$date_fixed <- fix_invalid_dates(output$date, "%d/%m/%Y")
  output$date <- output$date_fixed
  output$date_fixed <- NULL
  output$newspaper <- as.factor(output$newspaper)
  output$model_0_topic <- as.factor(output$model_0_topic)
  output$model_0_topic_label <- as.factor(output$model_0_topic_label)
  output$model_1_topic <- as.factor(output$model_1_topic)
  output$model_1_topic_label <- as.factor(output$model_1_topic_label)
  output$model_2_topic <- as.factor(output$model_2_topic)
  output$model_2_topic_label <- as.factor(output$model_2_topic_label)
  return(output)
}

filter_by_keywords <- function(df, keywords) {
  pattern <- paste(keywords, collapse = "|")
  mask <- grepl(pattern, df$model_0_topic_label, ignore.case = TRUE) |
    grepl(pattern, df$model_1_topic_label, ignore.case = TRUE) |
    grepl(pattern, df$model_2_topic_label, ignore.case = TRUE)
  df[mask, ]
}

############### CODE ####################
# Read csv files
csv_files <- list.files(path = "./data", pattern = "*.csv", full.names = TRUE)
# Preprocess and reads csv files
df_list <- list()
for (i in seq_along(csv_files)) {
  cat("Preprocessing file", i, ":", basename(csv_files[i]), "\n")
  df_list[[i]] <- preprocess_df(csv_files[i])
}
# Initialize with the first dataframe
combined_df <- df_list[[1]]
# Get model prefixes dynamically (any column ending with "_topic")
model_cols <- grep("_topic$", names(combined_df), value = TRUE)
model_prefixes <- sub("_topic$", "", model_cols)
# Create mapping list: one mapping df per model
mapping_list <- lapply(model_prefixes, function(prefix) {
  unique(combined_df[c(paste0(prefix, "_topic"), paste0(prefix, "_topic_label"))])
})
names(mapping_list) <- model_prefixes
for (i in 2:length(df_list)) {
  current_df <- df_list[[i]]
  for (prefix in model_prefixes) {
    topic_col <- paste0(prefix, "_topic")
    label_col <- paste0(prefix, "_topic_label")
    # Current mapping for this model
    current_mapping <- mapping_list[[prefix]]
    # --- Find new labels not yet in mapping
    new_labels <- !current_df[[label_col]] %in% current_mapping[[label_col]]
    if (any(new_labels)) {
      new_rows <- unique(current_df[new_labels, c(topic_col, label_col)])
      # Convert to numeric safely
      new_rows[[topic_col]] <- as.numeric(as.character(new_rows[[topic_col]]))
      combined_max <- max(as.numeric(as.character(combined_df[[topic_col]])), na.rm = TRUE)
      # Shift topics so they start after existing ones
      new_rows[[topic_col]] <- new_rows[[topic_col]] + combined_max
      # Convert back to factor to match original type
      new_rows[[topic_col]] <- as.factor(new_rows[[topic_col]])
      # Append to mapping
      current_mapping <- rbind(current_mapping, new_rows)
    }
    # --- Remap topics in current_df to use updated mapping
    match_idx <- match(current_df[[label_col]], current_mapping[[label_col]])
    current_df[[topic_col]] <- current_mapping[[topic_col]][match_idx]
    # Update mapping_list with the new mapping
    mapping_list[[prefix]] <- current_mapping
  }
  # Update df_list and combined_df
  df_list[[i]] <- current_df
  combined_df <- rbind(combined_df, current_df)
}

################ EXTRACTION ######################
keywords_economia_informal <- c(
  # Core terms - broad matching
  "informal", "ambulant", "formal", "autoemple", "subemple", "cachuel",
  
  # Key phrases - informal economy
  "economia informal", "economía informal", 
  "economia sumergida", "economía sumergida",
  "economia subterránea", "economía subterránea",
  "economia popular", "economía popular",
  "economia en la sombra", "economía en la sombra",
  "economia de subsistencia", "economía de subsistencia",
  "sector informal", "sector formal",
  "mercado informal", "mercado formal", 
  "actividad informal", "actividad formal", 
  "comercio informal", "comercio formal",
  "unidad productiva", "microempresa", "pequeña empresa",
  "trabajo no declarado", "empleo no declarado",
  "trabajo declarado", "empleo declarado",
  "trabajo sin contrato", "sin contrato",
  "trabajo con contrato", "con contrato",
  "sin seguridad social", "sin beneficios laborales",
  "seguridad social", "beneficios laborales",
  "trabajador por cuenta propia", "cuenta propia",
  "trabajadora por cuenta propia", 
  "trabajo doméstico", "trabajador doméstico", "trabajadora doméstica",
  "trabajo domestico", "trabajador domestico", "trabajadora domestica",
  "trabajador familiar auxiliar", "trabajador familiar",
  "trabajadora familiar auxiliar", "trabajadora familiar", 
  "trabajador auxiliar", "trabajadora auxiliar",
  "precariedad laboral", "bajos ingresos", "baja productividad",
  "formalización laboral", "formalización", "proceso de formalización",
  "formalizacion laboral", "formalizacion", "proceso de formalizacion",
  "inspección laboral", "inspeccion laboral",
  "International Labour Organisation", "Organización Internacional del Trabajo",
  "OECD", "Hernando de Soto", "Arthur Lewis",
  "oecd", "hernando de soto", "arthur lewis",
  "Chacaltana", "Williams", "Lansky", "Chen",
  "Gamarra", "La Parada", "gamarra", "la parada", 
  "cachuelo", "vendedor callejero",
  "ENAHO", "régimen laboral especial", "MYPE",
  "enaho", "mype", "mototax", "combi",
  "mujer informal", "mujeres informales",
  "mujer formal", "mujeres formales",
  "inseguridad económica", "inseguridad economica",
  "política de formalización", "politica de formalizacion",
  "políticas de formalización", "politicas de formalizacion",
  "formalización laboral", "formalizacion laboral",
  "registro laboral", "micro y pequeñas empresas",
  "Keith Hart", "actividades economicas no registradas",
  "actividades económicas no registradas", "actividades economicas registradas",
  "actividades económicas registradas", "empleo sin contrato",
  "empleo con contrato", "ausencia de contrato fijo", 
  "sin afiliacion a seguros sociales", "sin afiliación a seguros sociales",
  "sin beneficios laborales", "sin protecciones sociales", 
  "precarización laboral", "precarizacion laboral",
  "trabajador por cuenta propia", "trabajadora por cuenta propia",
  "empresas no registradas", "empresas no constituidas", "unincorporated enterprises",
  "precariedad", "registro sunat", "inscripcion sunat", "inscripción sunat",
  "regimen laboral especial", "régimen laboral especial", "trabajo ambulante", 
  "comercio ambulatorio", "vendedor ambulante",  "vendedor callejero",
  "agricultura informal", "agricultura formal", 
  "mineria informal", "mineria formal",
  "minería informal", "minería formal",
  "zona rural informal", "jovenes informales", "jóvenes informales",
  "joven informal", "joven informal",
  "juventud informal", "juventud informal"
)
keywords_modernizante <- c(
  "sector dual", "Arthur Lewis", "premoderno", "subsistencia", "industrialización", "industrializacion",
  "desarrollo económico", "desarrollo economico", "absorción", "absorcion", "sector capitalista",
  "etapas de desarrollo", "transición", "transicion", "formalización", "formalizacion",
  "progreso", "modernización", "modernizacion", "residuo", "atraso"
)
keywords_estructuralista <- c(
  "marxismo", "capitalismo", "exclusión", "exclusion", "salarios bajos", "competencia laboral",
  "explotación", "explotacion", "plusvalía", "plusvalia", "reserva de mano de obra",
  "desigualdad estructural", "sistema económico", "sistema economico", "clase trabajadora",
  "precariado", "explotados", "informalidad estructural",
  "dependencia", "periferia", "centro", "migración rural-urbana", "migracion rural-urbana",
  "precarización", "precarizacion", "deslocalización", "deslocalizacion",
  "externalización", "externalizacion", "subcontratación", "subcontratacion"
)
keywords_neoliberal <- c(
  "Hernando de Soto", "burocracia", "regulaciones", "impuestos", "intervención estatal", "intervencion estatal",
  "flexibilidad", "autonomía", "autonomia", "libre mercado", "costos de formalización", "costos de formalizacion",
  "trámites", "tramites", "emprendimiento", "libertad económica", "libertad economica",
  "deregulación", "desregulación", "desregulacion", "mercado libre",
  "barreras regulatorias", "racionalidad", "elección individual", "eleccion individual",
  "evitar impuestos", "informalidad voluntaria"
)
keywords_posmoderna <- c(
  "redes de solidaridad", "antropología", "antropologia", "cultura", "reciprocidad", "comunidad",
  "capital social", "trueque", "mercados populares", "identidad", "tradición", "tradicion",
  "resistencia cultural", "economía alternativa", "economia alternativa",
  "redistribución", "redistribucion", "cooperación", "cooperacion",
  "valores comunitarios", "informalidad cultural", "prácticas locales", "practicas locales",
  "solidaridad", "ayni", "minka", "minga"
)
keywords_voluntarista <- c(
  "evasión", "evasion", "competencia desleal", "regulaciones ineficientes", "beneficios",
  "maximización", "maximizacion", "estrategia", "ventaja competitiva", "mercado libre",
  "abuso de controles", "ineficiencia estatal", "opción racional", "opcion racional",
  "beneficio individual", "eludir normas", "informalidad estratégica", "informalidad estrategica",
  "rentabilidad", "fraude", "subdeclaración", "subdeclaracion", "economía ilegal", "economia ilegal"
)
### Apply for each conceptual group ###
# economia_informal_df <- filter_by_keywords(combined_df, keywords_economia_informal)
perspectiva_modernizante_df <- filter_by_keywords(economia_informal_df, keywords_modernizante)
perspectiva_estructuralista_df <- filter_by_keywords(economia_informal_df, keywords_estructuralista)
perspectiva_neoliberal_df <- filter_by_keywords(economia_informal_df, keywords_neoliberal)
perspectiva_posmoderna_df <- filter_by_keywords(economia_informal_df, keywords_posmoderna)
perspectiva_voluntarista_df <- filter_by_keywords(economia_informal_df, keywords_voluntarista)

### Extract randomized/stratified samples ###
set.seed(123)  # For reproducibility

# Considers newspapers and years
sample_df <- economia_informal_df %>%
  mutate(year = format(date, "%Y")) %>%  # Extract year from date
  group_by(newspaper, year) %>%
  slice_sample(prop = 1200 / nrow(economia_informal_df)) %>%
  ungroup()

table(economia_informal_df$newspaper)
table(sample_df$newspaper)

# Shuffle the sample_df
sample_df <- sample_df %>%
  slice_sample(n = nrow(sample_df))

### Save as csv
write.csv(sample_df, file = "./sample_df.csv", row.names = FALSE)
write.csv(economia_informal_df, file = "./economia_informal_df.csv", row.names = FALSE)
write.table(lapply(economia_informal_df, as.character), 
            file = "./economia_informal_df.csv", 
            sep = ";", 
            dec = ".", 
            quote = TRUE, 
            row.names = FALSE,
            col.names = TRUE,
            qmethod = "double")
write.csv(perspectiva_modernizante_df, file = "./perspectiva_modernizante_df.csv", row.names = FALSE)
write.csv(perspectiva_estructuralista_df, file = "./perspectiva_estructuralista_df.csv", row.names = FALSE)
write.csv(perspectiva_neoliberal_df, file = "./perspectiva_neoliberal_df.csv", row.names = FALSE)
write.csv(perspectiva_posmoderna_df, file = "./perspectiva_posmoderna_df.csv", row.names = FALSE)
write.csv(perspectiva_voluntarista_df, file = "./perspectiva_voluntarista_df.csv", row.names = FALSE)

summary(economia_informal_df)

### Analysis of results

# 1. PORCENTAJE GENERAL
total_noticias <- nrow(combined_df)
muestra_noticias <- nrow(economia_informal_df)
porcentaje_general <- (muestra_noticias / total_noticias) * 100

resumen_general <- data.frame(
  Metric = c("Total de noticias", "Muestra filtrada", "Porcentaje de cobertura"),
  Value = c(
    format(total_noticias, big.mark = ","),
    format(muestra_noticias, big.mark = ","),
    sprintf("%.2f%%", porcentaje_general)
  )
)

# 2. PORCENTAJE POR PERIÓDICO
porcentaje_periodico <- combined_df %>%
  count(newspaper, name = "total") %>%
  left_join(
    economia_informal_df %>%
      count(newspaper, name = "muestra"),
    by = "newspaper"
  ) %>%
  mutate(
    muestra = ifelse(is.na(muestra), 0, muestra),
    porcentaje = (muestra / total) * 100,
    porcentaje_formateado = sprintf("%.2f%%", porcentaje)
  )

# Gráfico 1: Comparación general
datos_general <- data.frame(
  categoria = c("Artículos del CGEC13-20", "Muestra extraída"),
  cantidad = c(total_noticias, muestra_noticias),
  porcentaje = c(100, porcentaje_general)
)

grafico_general <- ggplot(datos_general, aes(x = categoria, y = cantidad)) +
  geom_col(aes(fill = categoria), alpha = 0.8, show.legend = FALSE) +
  geom_text(aes(label = paste(format(cantidad, big.mark = ","), 
                              "\n(", sprintf("%.2f%%", porcentaje), ")")),
            vjust = 0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("Artículos del CGEC13-20" = "#4E79A7", 
                               "Muestra extraída" = "#F28E2B")) +
  scale_y_continuous(labels = label_comma(big.mark = ".", decimal.mark = ",")) +
  labs(title = "Comparación entre número total de artículos y muestra extraída",
       subtitle = "Número absoluto y porcentaje de artículos periodísticos",
       x = "",
       y = "") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.text.x = element_text(size = 11))

# Gráfico 2: Desviación por periódico
promedio_general <- porcentaje_general

datos_desviacion <- porcentaje_periodico %>%
  mutate(
    desviacion = round(porcentaje, 2) - round(promedio_general, 2),
    tipo_desviacion = ifelse(desviacion > 0, "Por encima del promedio", "Por debajo del promedio"),
    desviacion_absoluta = round(abs(desviacion), 2)
  )
datos_desviacion$newspaper <- c("Correo", "El Comercio", "Gestión", "Ojo", "Perú21", "Publimetro", "Trome")

grafico_desviacion <- ggplot(datos_desviacion, 
                             aes(x = reorder(newspaper, desviacion), 
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
  labs(title = "Desviación por periódico respecto al promedio general",
       subtitle = paste("Línea roja = Promedio general (", sprintf("%.2f%%", promedio_general), ")"),
       x = "",
       y = "Desviación del promedio (%)",
       fill = "Posición relativa") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom")

# Gráfico 3: Evolución anual de la cobertura
levels(datos_evolucion$newspaper) <- c("Correo", "El Comercio", "Gestión", "Ojo", "Perú21", "Publimetro", "Trome")

etiquetas_final <- datos_evolucion %>%
  group_by(newspaper) %>%
  filter(año == max(año)) %>%
  ungroup()

promedio_anual <- datos_evolucion %>%
  group_by(año) %>%
  summarise(
    muestra_total = sum(muestra),
    total_total = sum(total),
    porcentaje_promedio = round((muestra_total / total_total) * 100, 2)
  )

evolucion_anual <- ggplot(datos_evolucion, aes(x = año, y = porcentaje, color = newspaper, group = newspaper)) +
  geom_line(size = 1.2, alpha = 0.8) +
  geom_point(size = 2) +
  geom_line(data = promedio_anual, 
            aes(x = año, y = porcentaje_promedio, color = "Promedio General"), 
            size = 1, color = "black", inherit.aes = FALSE) +
  geom_point(data = promedio_anual, 
             aes(x = año, y = porcentaje_promedio), 
             color = "black", size = 1.5, inherit.aes = FALSE) +
  geom_text(data = etiquetas_final, 
            aes(label = newspaper), 
            hjust = -0.1, size = 3, check_overlap = FALSE) +
  geom_text(data = promedio_anual %>% filter(año == max(año)), 
            aes(x = año, y = porcentaje_promedio, label = "Promedio General"), 
            hjust = -0.1, vjust = -0.5, color = "black", fontface = "bold", 
            size = 3.5, inherit.aes = FALSE) +
  scale_x_continuous(
    breaks = seq(min(datos_evolucion$año), max(datos_evolucion$año), by = 1),
    limits = c(min(datos_evolucion$año), max(datos_evolucion$año) + 1)
  ) +
  scale_y_continuous(
    name = "Porcentaje de cobertura (%)",
    limits = c(0, NA)
  ) +
  scale_color_discrete(name = "Periódico") +
  labs(
    title = "Evolución anual de la cobertura (hasta diciembre 2019)",
    subtitle = "Línea negra = Promedio general anual de todos los periódicos",
    x = "Año",
    y = "Porcentaje de cobertura (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "none"
  )

# Gráfico 4: TOPICS MÁS ASOCIADOS con economía informal
procesar_topics_excluir <- function(df, topic_id_column, topic_label_column, n_topics = 10, excluir_indices = NULL) {
  
  # Primera pasada: obtener todos los topics ordenados
  topics_completos <- df %>%
    filter({{topic_id_column}} != -1) %>%
    filter(!grepl("UNKNOWN|unknown|^\\s*$", {{topic_label_column}})) %>%
    filter(!is.na({{topic_label_column}})) %>%
    filter({{topic_label_column}} != "") %>%
    count({{topic_label_column}}, sort = TRUE) %>%
    mutate(
      topic_label_clean = stringr::str_trim({{topic_label_column}}),
      topic_label_wrap = stringr::str_wrap(topic_label_clean, 40),
      indice = row_number()  # Añadir índice para exclusión
    )
  
  # Filtrar excluyendo los índices no deseados
  if (!is.null(excluir_indices)) {
    topics_filtrados <- topics_completos %>%
      filter(!indice %in% excluir_indices)
  } else {
    topics_filtrados <- topics_completos
  }
  
  # Tomar los n_topics después de la exclusión
  topics_final <- topics_filtrados %>%
    head(n_topics)
  
  return(topics_final)
}

topics_model_0 <- procesar_topics_excluir(
  economia_informal_df, 
  model_0_topic, 
  model_0_topic_label, 
  n_topics = 10,
  excluir_indices = c(6, 8, 11)
)
grafico_topics_model_0 <- topics_model_0 %>%
  ggplot(aes(x = reorder(topic_label_wrap, n), y = n)) +
  geom_col(fill = "#76B7B2", alpha = 0.8) +
  coord_flip() +
  labs(title = "Top 10 campos semánticos identificados",
       subtitle = "Modelo: distiluse-base-multilingual-cased-v1",
       x = "", y = "Frecuencia") +
  theme_minimal()

topics_model_1 <- procesar_topics_excluir(
  economia_informal_df, 
  model_1_topic, 
  model_1_topic_label, 
  n_topics = 10,
  excluir_indices = c(1, 5, 6, 7, 10, 14, 15, 16)
)
grafico_topics_model_1 <- topics_model_1 %>%
  ggplot(aes(x = reorder(topic_label_wrap, n), y = n)) +
  geom_col(fill = "#76B7B2", alpha = 0.8) +
  coord_flip() +
  labs(title = "Top 10 campos semánticos identificados",
       subtitle = "Modelo: sentence_similarity_spanish_es",
       x = "", y = "Frecuencia") +
  theme_minimal()

topics_model_2 <- procesar_topics_excluir(
  economia_informal_df, 
  model_2_topic, 
  model_2_topic_label, 
  n_topics = 10,
  excluir_indices = c(4, 5, 7, 10, 12, 13, 15, 16)
)
grafico_topics_model_2 <- topics_model_2 %>%
  ggplot(aes(x = reorder(topic_label_wrap, n), y = n)) +
  geom_col(fill = "#76B7B2", alpha = 0.8) +
  coord_flip() +
  labs(title = "Top 10 campos semánticos identificados",
       subtitle = "Modelo: Linq-Embed-Mistral",
       x = "", y = "Frecuencia") +
  theme_minimal()

# Gráfico 5: TOPICS MÁS ASOCIADOS con economía informal por periodico
procesar_topics_un_periodico <- function(df, periodico, n_topics = 10, excluir_indices_model0 = NULL, excluir_indices_model1 = NULL, excluir_indices_model2 = NULL) {
  
  # Filtrar por el periódico específico
  df_filtrado <- df %>% filter(newspaper == periodico)
  
  # Función auxiliar para procesar cada modelo
  procesar_modelo <- function(df, topic_id_col, topic_label_col, modelo_nombre, excluir_indices = NULL) {
    
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
    
    return(topics %>% head(n_topics))
  }
  
  # Procesar los tres modelos
  topics_model0 <- procesar_modelo(df_filtrado, model_0_topic, model_0_topic_label, "Modelo 0", excluir_indices_model0)
  topics_model1 <- procesar_modelo(df_filtrado, model_1_topic, model_1_topic_label, "Modelo 1", excluir_indices_model1)
  topics_model2 <- procesar_modelo(df_filtrado, model_2_topic, model_2_topic_label, "Modelo 2", excluir_indices_model2)
  
  # Combinar resultados
  resultados <- bind_rows(topics_model0, topics_model1, topics_model2) %>%
    mutate(periodico = periodico)
  
  return(resultados)
}

topics_elcomercio <- procesar_topics_un_periodico(
  economia_informal_df, 
  periodico = "elcomercio",
  excluir_indices_model0 = c(3, 4),
  excluir_indices_model1 = c(1,2,4,6),
  excluir_indices_model2 = c(3,4,5,6,7,8,9,10,11,12,13),
  n_topics = 3
)

topics_correo <- procesar_topics_un_periodico(
  economia_informal_df, 
  periodico = "correo",
  #excluir_indices_model0 = c(3, 4),
  excluir_indices_model1 = c(3),
  #excluir_indices_model2 = c(3,4,5,6,7,8,9,10,11,12,13),
  n_topics = 3
)

topics_gestion <- procesar_topics_un_periodico(
  economia_informal_df, 
  periodico = "gestion",
  excluir_indices_model0 = c(3),
  excluir_indices_model1 = c(1,3,4,5),
  #excluir_indices_model2 = c(),
  n_topics = 3
)

topics_ojo <- procesar_topics_un_periodico(
  economia_informal_df, 
  periodico = "ojo",
  #excluir_indices_model0 = c(3),
  excluir_indices_model1 = c(2),
  excluir_indices_model2 = c(1,3),
  n_topics = 3
)

topics_trome <- procesar_topics_un_periodico(
  economia_informal_df, 
  periodico = "trome",
  #excluir_indices_model0 = c(),
  excluir_indices_model1 = c(2,3,4),
  excluir_indices_model2 = c(1,3),
  n_topics = 3
)

topics_peru21 <- procesar_topics_un_periodico(
  economia_informal_df, 
  periodico = "peru21",
  excluir_indices_model0 = c(2),
  excluir_indices_model1 = c(2,3,4),
  #excluir_indices_model2 = c(1,3),
  n_topics = 3
)

topics_publimetro <- procesar_topics_un_periodico(
  economia_informal_df, 
  periodico = "publimetro",
  #excluir_indices_model0 = c(),
  #excluir_indices_model1 = c(2,3,4),
  #excluir_indices_model2 = c(1,3),
  n_topics = 3
)

grafico_comparativo <- topics_publimetro %>%
  ggplot(aes(x = reorder(topic_label_wrap, n), y = n, fill = modelo)) +
  geom_col(alpha = 0.8) +
  coord_flip() +
  facet_wrap(~modelo, ncol = 1, scales = "free_y") +
  scale_fill_manual(values = c("Modelo 0" = "#2E86AB", "Modelo 1" = "#A23B72", "Modelo 2" = "#F18F01")) +
  labs(
    title = paste("Top 3 campos semánticos identificados"),
    subtitle = "Periódico: Publimetro",
    x = "",
    y = "Frecuencia"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 10)
  )

print(grafico_comparativo)

# 5. TABLA: Periódicos rankeados por cobertura
tabla_ranking <- datos_desviacion %>%
  arrange(desc(porcentaje)) %>%
  select(Periódico = newspaper, 
         `Total Noticias` = total, 
         `Muestra EI` = muestra, 
         `Porcentaje` = porcentaje,
         `Desviación` = desviacion) %>%
  mutate(across(c(Porcentaje, Desviación), ~ sprintf("%.2f%%", .)))

tabla_años <- economia_informal_df %>%
  mutate(año = year(date)) %>%
  count(año, name = "muestra") %>%
  left_join(
    combined_df %>%
      mutate(año = year(date)) %>%
      count(año, name = "total"),
    by = "año"
  ) %>%
  mutate(porcentaje = round((muestra / total) * 100, 2)) %>%
  arrange(desc(porcentaje)) %>%
  select(Año = año, `Total` = total, `Muestra` = muestra, `Porcentaje` = porcentaje)
