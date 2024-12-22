#### >>> 0. Chargement et préparation des données ####
library(shiny)
library(dplyr)
library(DT)
library(ggplot2)
library(shinydashboard)
library(plotly)
library(tidyverse)
library(corrplot)
library(caret)
library(randomForest)
library(e1071)
library(rpart)


#### >>> 1. Chargement et préparation des données ####
#### >> 1.1 Chargement du csv ####
titanic <- read.csv('donnée_fr.csv', sep = ';')

#### >> 1.2 préparation des données ####
titanic_clean <- titanic %>%
  mutate(
    survecu = as.factor(survecu),
    classe_billet = as.factor(classe_billet),
    sexe = factor(sexe, levels = c("male", "female"))
  ) %>%
  drop_na(survecu, classe_billet, sexe, age, tarif)

#### >> 1.3 Préparation du machine learning ####
set.seed(0)
train_index <- createDataPartition(titanic_clean$survecu, p = 0.8, list = FALSE)
train_data <- titanic_clean[train_index, ]
test_data <- titanic_clean[-train_index, ]

#### >> 1.4 Préparation des modèles ####
#### > 1.4.1 Modèle Random Forest ####
rf_model <- randomForest(
  survecu ~ sexe + age + classe_billet + nb_famille_proche + nb_parents_enfants + tarif,
  data = train_data, ntree = 100, mtry = 3, importance = TRUE
)

#### > 1.4.2 Modèle Régression Logistique ####
logistic_model <- glm(
  survecu ~ sexe + age + classe_billet + nb_famille_proche + nb_parents_enfants + tarif,
  data = train_data, family = binomial
)

#### > 1.4.3 Modèle Arbre de Décision ####
decision_tree_model <- rpart(
  survecu ~ sexe + age + classe_billet + nb_famille_proche + nb_parents_enfants + tarif,
  data = train_data, method = "class"
)

#### > 1.4.4 Modèle k-NN ####
knn_model <- train(
  survecu ~ sexe + age + classe_billet + nb_famille_proche + nb_parents_enfants + tarif,
  data = train_data, method = "knn", tuneLength = 10
)

#### >>> 2. Interface utilisateur ####
ui <- dashboardPage(
  dashboardHeader(title = "DataTanic"),
  
  #### >> 2.1 Barre latérale avec les onglets ####
  dashboardSidebar(
    sidebarMenu(
      menuItem("Exploration des données", tabName = "data_exploration", icon = icon("chart-bar")),
      menuItem("Prédiction", tabName = "prediction", icon = icon("magic"))
    )
  ),
  
  #### >> 2.2 Corps du tableau de bord ####
  dashboardBody(
    tabItems(
      #### > 2.2.1 Onglet Exploration des données ####
      tabItem(
        tabName = "data_exploration",
        fluidRow(
          box(
            # Choix de l'analyse
            title = "Analyse des données", width = 12,
            selectInput("explore_type", "Choisissez une analyse :", 
                        choices = c(
                          "Résumé" = "summary",
                          "Histogrammes" = "histograms",
                          "Corrélation" = "correlation",
                          "Boxplots" = "boxplots",
                          "Distribution par variables catégoriques" = "categorical",
                          "Visualisations originales" = "original_visualization"
                        )),
            conditionalPanel(
              condition = "input.explore_type == 'histograms'",
              selectInput("hist_variable", "Variable pour l'histogramme :", 
                          choices = c("age", "tarif", "nb_famille_proche", "nb_parents_enfants"))
            ),
            conditionalPanel(
              condition = "input.explore_type == 'boxplots'",
              selectInput("box_var", "Variable pour le boxplot :", 
                          choices = c("age", "tarif")),
              selectInput("box_group", "Grouper par :", 
                          choices = c("survecu", "sexe"))
            ),
            conditionalPanel(
              condition = "input.explore_type == 'categorical'",
              selectInput("cat_variable", "Variable catégorique :", 
                          choices = names(titanic_clean)[sapply(titanic_clean, is.factor)])
            )
          ),
          # Visualisation
          box(
            title = "Visualisation", width = 12,
            uiOutput("dynamic_content")
          )
        )
      ),
      
      #### > 2.2.2 Onglet Prédiction ####
      tabItem(
        tabName = "prediction",
        fluidRow(
          box(
            # Choix du modèle de prédiction et des paramètres
            title = "Prédiction", width = 4,
            selectInput("model", "Modèle à utiliser :", 
                        choices = c("Random Forest" = "rf", "Régression Logistique" = "logistic", 
                                    "Arbre de Décision" = "decision_tree", "k-NN" = "knn")),
            selectInput("classe_billet", "Classe du billet :", choices = c(1, 2, 3)),
            selectInput("sexe", "Sexe :", choices = c("male", "female")),
            numericInput("age", "Âge :", value = 30),
            numericInput("nb_famille_proche", "Nombre de frères/sœurs/conjoints :", value = 0),
            numericInput("nb_parents_enfants", "Nombre de parents/enfants :", value = 0),
            numericInput("tarif", "Tarif :", value = 50),
            actionButton("predict", "Prédire la survie")
          ),
          
          # Résultat de la prédiction et des probabilités
          box(
            title = "Résultat de la prédiction", width = 8,
            textOutput("result"),
            br(),
            h4("Probabilités (si disponibles)"),
            verbatimTextOutput("probabilities")
          )
        ),
        
        # Matrice de Confusion
        fluidRow(
          box(
            title = "Matrice de Confusion", status = "primary", solidHeader = TRUE, width = 12,
            plotOutput("confusion_matrix")
          )
        )
      )
    )
  )
)


#### >>> 3. Interface utilisateur ####
server <- function(input, output) {
  
  #### >> 3.1 Prédiction ####
  observeEvent(input$predict, {
    new_data <- data.frame(
      classe_billet = factor(input$classe_billet, levels = levels(train_data$classe_billet)),
      sexe = factor(input$sexe, levels = c("male", "female")),
      age = input$age,
      nb_famille_proche = input$nb_famille_proche,
      nb_parents_enfants = input$nb_parents_enfants,
      tarif = input$tarif
    )
    
    prediction <- switch(input$model,
                         "rf" = predict(rf_model, new_data, type = "class"),
                         "logistic" = ifelse(predict(logistic_model, new_data, type = "response") > 0.5, 1, 0),
                         "decision_tree" = predict(decision_tree_model, new_data, type = "class"),
                         "knn" = predict(knn_model, new_data))
    
    probabilities <- switch(input$model,
                            "rf" = predict(rf_model, new_data, type = "prob"),
                            "logistic" = predict(logistic_model, new_data, type = "response"),
                            "decision_tree" = predict(decision_tree_model, new_data, type = "prob"),
                            "knn" = predict(knn_model, new_data, type = "prob"))
    
    result <- ifelse(prediction == 1, "La personne a survécu.", "La personne est décédée.")
    output$result <- renderText(result)
    output$probabilities <- renderPrint(probabilities)
  })
  
  #### >> 3.2 Matrice de Confusion ####
  output$confusion_matrix <- renderPlot({
    req(input$model)
    
    predictions <- switch(input$model,
                          "rf" = predict(rf_model, test_data),
                          "logistic" = {
                            # Pour la régression logistique, on transforme les probabilités en classes binaires
                            prob <- predict(logistic_model, test_data, type = "response")
                            ifelse(prob > 0.5, 1, 0)
                          },
                          "decision_tree" = predict(decision_tree_model, test_data, type = "class"),
                          "knn" = predict(knn_model, test_data))
    
  
    predictions <- factor(predictions, levels = c(0, 1))
    test_data$survecu <- factor(test_data$survecu, levels = c(0, 1))
    
    # Calcul de la matrice de confusion
    confusion <- confusionMatrix(predictions, test_data$survecu)
    
    # Affichage de la matrice de confusion sous forme de graphique
    confusion_df <- as.data.frame(confusion$table)
    
    ggplot(confusion_df, aes(x = Reference, y = Prediction)) +
      geom_tile(aes(fill = Freq), color = "white") +
      geom_text(aes(label = Freq), vjust = 1) +
      scale_fill_gradient(low = "white", high = "steelblue") +
      labs(title = "Matrice de confusion", x = "Vrai", y = "Prédiction") +
      theme_minimal()
  })
  
  #### >> 3.3 Visualisation Analyse Exploratoire ####
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
        plotOutput("boxplot")
      )
    } else if (input$explore_type == "categorical") {
      plotOutput("categorical_plot")
    } else if (input$explore_type == "original_visualization") {  # Visualisation originale
      tagList(
        plotlyOutput("radial_plot"),
        plotlyOutput("correlation_matrix_plot"),
        plotOutput("survival_distribution_plot"),
        plotOutput("violin_plot")
      )
    }
  })
  
  #### > 3.3.1 Résumé ####
  output$summary_output <- renderPrint({
    summary(titanic_clean)
  })
  
  #### > 3.3.2 Histogramme ####
  output$histogram_plot <- renderPlot({
    req(input$hist_variable)
    ggplot(titanic_clean, aes_string(x = input$hist_variable, fill = "survecu")) +
      geom_histogram(bins = 30, alpha = 0.7, position = "identity", color = "black") +
      labs(x = input$hist_variable, y = "Fréquence", title = paste("Histogramme de", input$hist_variable)) +
      scale_fill_manual(values = c("red", "green"))
  })
  
  ##### > 3.3.3 Corrélation ####
  output$correlation_plot <- renderPlot({
    corr_matrix <- cor(titanic_clean %>% select(age, tarif, nb_famille_proche, nb_parents_enfants))
    corrplot(corr_matrix, method = "circle")
  })
  
  #### > 3.3.4 Boxplots ####
  output$boxplot <- renderPlot({
    ggplot(titanic_clean, aes_string(x = input$box_group, y = input$box_var, fill = input$box_group)) +
      geom_boxplot() +
      labs(title = paste("Boxplot :", input$box_var, "par", input$box_group))
  })
  
  
  #### > 3.3.5 Distribution par variables catégoriques ####
  output$categorical_plot <- renderPlot({
    req(input$cat_variable)
    ggplot(titanic_clean, aes_string(x = input$cat_variable, fill = "survecu")) +
      geom_bar(position = "dodge") +
      labs(title = paste("Distribution de", input$cat_variable, "par survie"))
  })
  
  #### > 3.3.6 Visualisation originale ####
  
  # Graphique Radial
  output$radial_plot <- renderPlotly({
    plot_data <- titanic_clean %>%
      group_by(classe_billet) %>%
      summarise(survived_rate = mean(survecu == "1"))
    
    plot_ly(data = plot_data, type = 'scatterpolar', r = ~survived_rate, theta = ~classe_billet, mode = 'lines+markers', fill = 'toself') %>%
      layout(polar = list(radialaxis = list(visible = T, range = c(0, 1))), title = "Taux de survie par classe")
  })
  
  # Distribution de la survie
  output$survival_distribution_plot <- renderPlot({
    ggplot(titanic_clean, aes(x = tarif, fill = survecu)) +
      geom_density(alpha = 0.6) +
      labs(title = "Distribution des tarifs par survie")
  })
  
  # Matrice de corrélation
  output$correlation_matrix_plot <- renderPlotly({
    corr_matrix <- cor(titanic_clean %>% select(age, tarif, nb_famille_proche, nb_parents_enfants))
    plot_ly(x = colnames(corr_matrix), y = colnames(corr_matrix), z = corr_matrix, type = "heatmap", colorscale = "Viridis") %>%
      layout(title = "Matrice de Corrélation")
  })
  
  # Graphique Violin pour la distribution de l'âge par survie
  output$violin_plot <- renderPlot({
    ggplot(titanic_clean, aes(x = survecu, y = age, fill = survecu)) +
      geom_violin() +
      labs(title = "Distribution de l'âge par survie")
  })
  
}


#### >>> 4. Lancement de l'application
shinyApp(ui = ui, server = server)
