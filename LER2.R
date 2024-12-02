library(tidyverse)
library(corrplot)
library(broom)
library(caret)
library(randomForest)
library(shiny)

# === Chargement et préparation des données ===

titanic <- read.csv('U:/Documents/SAE DATAMINING/titanic.csv')

colnames(titanic) <- c(
  "survecu", "classe_billet", "sex", "age", "nb_famille_proche", 
  "nb_parents_enfants", "tarif", "port_embarquement", "classe", 
  "categorie_personne", "homme_adulte", "pont", "ville_embarquement", 
  "en_vie", "voyage_seul"
)

# Préparation des données
titanic_clean <- titanic %>%
  mutate(
    survecu = as.factor(survecu),
    classe_billet = as.factor(classe_billet),
    sex = factor(sex, levels = c("male", "female"))
  ) %>%
  drop_na(survecu, classe_billet, sex, age, tarif)

# Diviser les données en ensembles d'entraînement et de test
set.seed(0)
train_index <- createDataPartition(titanic_clean$survecu, p = 0.8, list = FALSE)
train_data <- titanic_clean[train_index, ]
test_data <- titanic_clean[-train_index, ]

# Modèle Random Forest
rf_model <- randomForest(
  survecu ~ sex + age + classe_billet + nb_famille_proche + nb_parents_enfants + tarif,
  data = train_data,
  ntree = 100, mtry = 3, importance = TRUE
)

# === Application Shiny ===

ui <- fluidPage(
  titlePanel("Prédiction de survie Titanic"),
  
  navbarPage("Menu",
             # Onglet Prédiction
             tabPanel("Prédiction", 
                      sidebarLayout(
                        sidebarPanel(
                          selectInput("classe_billet", "Classe du billet :", choices = c(1, 2, 3)),
                          selectInput("sex", "Sexe :", choices = c("male", "female")),
                          numericInput("age", "Âge :", value = 30),
                          numericInput("nb_famille_proche", "Nombre de frères/sœurs/conjoints :", value = 0),
                          numericInput("nb_parents_enfants", "Nombre de parents/enfants :", value = 0),
                          numericInput("tarif", "Tarif :", value = 50),
                          actionButton("predict", "Prédire la survie")
                        ),
                        mainPanel(
                          h4("Résultat :"),
                          textOutput("result")
                        )
                      )
             ),
             
             # Onglet Analyse Exploratoire
             tabPanel("Analyse Exploratoire", 
                      sidebarLayout(
                        sidebarPanel(
                          h4("Choisissez une analyse :"),
                          selectInput("explore_type", "Type de visualisation :",
                                      choices = c(
                                        "Résumé" = "summary",
                                        "Histogrammes" = "histograms",
                                        "Corrélation" = "correlation",
                                        "Boxplots" = "boxplots",
                                        "Distribution par variables catégoriques" = "categorical"
                                      )),
                          actionButton("update_graph", "Mettre à jour"),
                          
                          # Sélection pour les histogrammes
                          conditionalPanel(
                            condition = "input.explore_type == 'histograms'",
                            selectInput("hist_variable", "Variable pour l'histogramme :", 
                                        choices = c("age", "tarif", "nb_famille_proche", "nb_parents_enfants"))
                          ),
                          
                          # Sélection pour les variables catégoriques
                          conditionalPanel(
                            condition = "input.explore_type == 'categorical'",
                            selectInput("cat_variable", "Variable catégorique :", 
                                        choices = names(titanic_clean)[sapply(titanic_clean, is.factor)])
                          )
                        ),
                        mainPanel(
                          uiOutput("dynamic_content")
                        )
                      )
             )
  )
)

server <- function(input, output) {
  
  # === Prédiction ===
  observeEvent(input$predict, {
    new_data <- data.frame(
      classe_billet = factor(input$classe_billet, levels = levels(train_data$classe_billet)),
      sex = factor(input$sex, levels = c("male", "female")),
      age = input$age,
      nb_famille_proche = input$nb_famille_proche,
      nb_parents_enfants = input$nb_parents_enfants,
      tarif = input$tarif
    )
    
    prediction <- predict(rf_model, new_data, type = "class")
    result <- ifelse(prediction == 1, "La personne a survécu.", "La personne est décédée.")
    output$result <- renderText(result)
  })
  
  # === Analyse Exploratoire ===
  output$dynamic_content <- renderUI({
    req(input$explore_type)
    if (input$explore_type == "summary") {
      verbatimTextOutput("summary_output")
    } else if (input$explore_type == "histograms") {
      plotOutput("histogram_plot")
    } else if (input$explore_type == "correlation") {
      plotOutput("correlation_plot")
    } else if (input$explore_type == "boxplots") {
      tagList(
        plotOutput("boxplot_sex"),
        plotOutput("boxplot_age")
      )
    } else if (input$explore_type == "categorical") {
      plotOutput("categorical_plot")
    }
  })
  
  # Résumé
  output$summary_output <- renderPrint({
    summary(titanic_clean)
  })
  
  # Histogramme
  output$histogram_plot <- renderPlot({
    req(input$hist_variable)
    ggplot(titanic_clean, aes_string(x = input$hist_variable, fill = "survecu")) +
      geom_histogram(bins = 30, alpha = 0.7, position = "identity", color = "black") +
      labs(x = input$hist_variable, y = "Fréquence", title = paste("Histogramme de", input$hist_variable)) +
      scale_fill_manual(values = c("red", "green"))
  })
  
  # Corrélation
  output$correlation_plot <- renderPlot({
    corr_matrix <- cor(titanic_clean %>% select(age, tarif, nb_famille_proche, nb_parents_enfants))
    corrplot(corr_matrix, method = "circle")
  })
  
  # Boxplots
  output$boxplot_sex <- renderPlot({
    ggplot(titanic_clean, aes(x = sex, y = age, fill = sex)) +
      geom_boxplot() +
      labs(title = "Boxplot : âge par sexe")
  })
  
  output$boxplot_age <- renderPlot({
    ggplot(titanic_clean, aes(x = survecu, y = age, fill = survecu)) +
      geom_boxplot() +
      labs(title = "Boxplot : âge par survie")
  })
  
  # Distribution par variables catégoriques
  output$categorical_plot <- renderPlot({
    req(input$cat_variable)
    ggplot(titanic_clean, aes_string(x = input$cat_variable, fill = "survecu")) +
      geom_bar(position = "dodge") +
      labs(title = paste("Distribution de", input$cat_variable, "par survie"))
  })
}

# Lancer l'application Shiny
shinyApp(ui = ui, server = server)
