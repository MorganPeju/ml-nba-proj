rm(list = ls())
################################################
#             Projet MTH6312 - NBA             #
#               par Morgan PEJU                #
################################################

############################################
###           Librairies et dataset      ###
############################################
# Librairies
install.packages("glmnet")
install.packages("boot")
install.packages("ggplot2")
install.packages("reshape2")
install.packages("caret")
install.packages("class")
install.packages("randomForest")
install.packages("dplyr")
install.packages("MASS")
install.packages("rattle")
library(rattle)
library(dplyr)
library(MASS)
library(caret)
library(class)
library(randomForest)
library(glmnet)
library(boot)
library(ggplot2)
require(reshape2)

# Chargement du dataset
NBA_Data <- read.csv("data_nba.csv", sep=";")
head(NBA_Data) # Visualisation partielle du dataframe

# Visualiser les types de données des attributs
sapply(NBA_Data, typeof)

# Résumé des données 
summary(NBA_Data)

# Repérer les valeurs manquantes
sapply(NBA_Data, function(x) sum(is.na(x)))
NBA_Data[is.na(NBA_Data$X3P.),]

# Remplacer les valeurs manquantes par des zéros
NBA_Data[is.na(NBA_Data)] <- 0
sapply(NBA_Data, function(x) sum(is.na(x)))

# Compter le nombre de lignes identiques (check for duplicates)
sum(duplicated(NBA_Data, fromLast = TRUE), na.rm = TRUE)

# Analyse de la distribution des données
melt.nba <- melt(NBA_Data)
ggplot(data = melt.nba, aes(x = value)) + 
  stat_density() + 
  facet_wrap(~variable, scales = "free")

# Préparation du dataset
NBA_Data$TARGET_5Yrs = factor(NBA_Data$TARGET_5Yrs)
NBA_Data$Name <- NULL

############################################
###       Sélection de variables         ###
############################################

# Calculer la matrice de corrélation 
cormat <- round(cor(NBA_Data[,1:19]),2)

# Fonction pour garder uniquement le triangle supérieur d'une matrice
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
upper_tri <- get_upper_tri(cormat)

# Fondre la matrice de corrélation
melted_cormat <- melt(upper_tri, na.rm = TRUE)

# Créer une heatmap de corrélation
ggheatmap <- ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

# Afficher heatmap de corrélation avec les valeurs 
ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 3) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))

# Trouver les prédicteurs hautement corrélés
hc = findCorrelation(cormat, cutoff=0.95)
hc = sort(hc)
# Réduire les données aux prédicteurs qui ne sont pas hautement corrélés
reduced_Data = NBA_Data[,-c(hc)]
head(reduced_Data)


############################################
###                Modèles               ###
############################################

# Définition du seed(matricule)
set.seed(2103232)

# Définition du nombre de folds pour la validation croisée
KFOLD = 10

# Partitionnement du dataset en set d'entrainement et de test
P <- 0.8 # Pourcentage des données d'entrainement
indxTrain <- createDataPartition(y = NBA_Data$TARGET_5Yrs,p = P,list = FALSE)
train <- NBA_Data[indxTrain,]
test <- NBA_Data[-indxTrain,]

train.X <- train
train.X$TARGET_5Yrs <- NULL
train.Y <- train$TARGET_5Yrs

test.X <- test
test.X$TARGET_5Yrs <- NULL
test.Y <- test$TARGET_5Yrs

# Normalisation des données
trainX <- train[,names(train) != c("TARGET_5Yrs")]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))

######     Regression Logistique      ###### 

# Définition du seed(matricule)
set.seed(2103232)

# Comparaison entre modèle complet et modèle par sélection de variables

### Modèle Complet ###
full.model <- glm(TARGET_5Yrs ~ ., data = train, family = binomial)
as.data.frame(coef(full.model))
# Predictions
lr.prob.full <- full.model %>% predict(test, type = "response")
lr.pred.full <- ifelse(lr.prob.full > 0.5, "0", "1")
# Calcul du taux d'erreur
lr.fullmodel.error_r <- mean(lr.pred.full == test$TARGET_5Yrs)
cat("Regression Logistique (COMPLET): Le taux d'erreur pour le modèle complet est de :", lr.fullmodel.error_r, ".")

### Modèle par sélection de variables ###
step.model <- full.model %>% stepAIC(trace = FALSE)
as.data.frame(coef(step.model))

# Predictions
lr.prob.step <- predict(step.model, test, type = "response")
lr.pred.step <- ifelse(lr.prob.step > 0.5, "0", "1")
# Calcul du taux d'erreur
lr.stepmodel.error_r <- mean(lr.pred.step == test$TARGET_5Yrs)
cat("Regression Logistique (STEPWISE): Le taux d'erreur pour le modèle avec sélection de variable est de :", lr.stepmodel.error_r, ".")

if (lr.stepmodel.error_r < lr.fullmodel.error_r) {
  df_lr <- as.data.frame(coef(step.model))
  
  selected_var <- rownames(df_lr)
  selected_var <- selected_var[selected_var != "(Intercept)"]
  
  lr_var <- which(selected_var %in% colnames(train))
  
  lr_var <- append(lr_var, which(colnames(train)=="TARGET_5Yrs"))
  
  lr_train = train[,c(lr_var)]
  lr_test = test[,c(lr_var)]
  
  lr_test.X <- lr_test
  lr_test.X$TARGET_5Yrs <- NULL
  lr_test.Y <- lr_test$TARGET_5Yrs
  # Définition du seed(matricule)
  set.seed(2103232)
  
  # Configuration de la méthode (cross-validation) avec K folds
  trControl <- trainControl(method = "cv", number = KFOLD)
  
  # Ajustement du modèle complet de régression logistique
  lr.fit.step <- train(TARGET_5Yrs ~ . ,
                       method = "glm",
                       trControl = trControl,
                       metric = "Accuracy",
                       data = lr_train)
  
  lr.fit.step
  # Récupération des résultats
  error_r.LRkfold <- 1 - lr.fit.step$results["Accuracy"][,1]
  cat("Regression Logistique (STEPWISE) K-Fold : \n Le taux derreur obtenu est : ",error_r.LRkfold,".\n")
  
  # Application du modèle sur les données de test
  lr.pred.step <- predict(lr.fit.step,lr_test.X)
  
  # Résultat: matrice de confusion
  lr.step.table <- table(lr.pred.step, lr_test.Y)
  lr.step.table
  # Calcul du taux d'erreur
  lr.test_error.step <- mean(lr.pred.step != lr_test.Y )
  cat("Regression Logistique (STEPWISE): Le taux d'erreur sur les données de test est : ", lr.test_error.step, ".\n")
  cat("La sensibilité est de: ",sensitivity(lr.step.table), "; et la spécificité vaut : ", specificity(lr.step.table), ".")
  
}else {
  # Définition du seed(matricule)
  set.seed(2103232)
  
  # Configuration de la méthode (cross-validation) avec K folds
  trControl <- trainControl(method = "cv", number = KFOLD)
  
  # Ajustement du modèle complet de régression logistique
  lr.fit <- train(TARGET_5Yrs ~ . ,
                  method = "glm",
                  trControl = trControl,
                  metric = "Accuracy",
                  data = train)
  
  lr.fit
  # Récupération des résultats
  error_r.LRkfold <- 1 - lr.fit$results["Accuracy"][,1]
  cat("Regression Logistique (COMPLET) K-Fold : \n Le taux d'erreur d'entrainement obtenu est : ",error_r.LRkfold,".\n")
  
  # Application du modèle sur les données de test
  lr.pred <- predict(lr.fit,test.X)
  
  # Résultat: matrice de confusion
  lr.table <- table(lr.pred, test.Y)
  print(lr.table)
  # Calcul du taux d'erreur
  lr.test_error <- mean(lr.pred != test.Y )
  cat("Regression Logistique (COMPLET) : Le taux d'erreur sur les données de test est : ", lr.test_error, ".\n")
  cat("La sensibilité est de: ",sensitivity(lr.table), "; et la spécificité vaut : ", specificity(lr.table), ".")
  
}

######  KNN: K plus proches voisins   ###### 

# Définition du seed(matricule)
set.seed(2103232)
# Définition de la valeur max du nombre de voisins
M <- 100
# Configuration de la méthode (cross-validation) avec K folds
trControl <- trainControl(method = "cv",number = KFOLD)
# Ajustement du modèle
knn.fit <- train(TARGET_5Yrs ~ . ,
                 method = "knn",
                 tuneGrid = expand.grid(k = 1:M),
                 trControl = trControl,
                 metric = "Accuracy",
                 data = train)

# Récupération des résultats
knn.fit_results <- knn.fit$results
results <- data.frame(k=knn.fit_results$k, error_r=1-knn.fit_results$Accuracy)
plot(results$k, results$error_r, type="b", col="black", main="KNN: Taux d'erreur en fonction du nombre de voisins", xlab = "Nombre de k voisins", ylab = "Taux d'erreur")

# On détermine le K optimal, pour lequel le taux d'erreur est minimal
min_results <- results[which.min(results$error_r),]
min_K <- min_results$k
min_error <- min_results$error_r
cat("La valeur optimale de k voisins est : ", min_K, "où le taux d'erreur obtenu est minimal soit :
", min_error,".")

# On trace le graphe du taux d'erreur en fonction de k
plot(x=results$k, y=results$error_r, type="b", xlab="k voisins", ylab="Taux d'erreur", ylim=c(0.28,0.39))
title(main="Le taux d'erreur en fonction de k voisins (10-Fold CV)")

# Définition du seed(matricule)
set.seed(2103232)

# Application du modèle sur les données de test
knn_minK <- knn(train.X, test.X, train.Y, k=min_K)

# Résultat: matrice de confusion
knn.table <- table(knn_minK, test.Y)
knn.table

# Calcul du taux d'erreur
knn.test_error <- mean(knn_minK != test.Y )
cat("Le taux d'erreur sur les données de test est : ", knn.test_error, " avec K (nombre de voisins)=", min_K,".\n")
cat("La sensibilité est de: ",sensitivity(knn.table), "; et la spécificité vaut : ", specificity(knn.table), ".")

######       Arbre de décision        ###### 

# Définition du seed(matricule)
set.seed(2103232)

# Configuration de la méthode (cross-validation) avec K folds
trControl <- trainControl(method = "cv",number = KFOLD)

# Ajustement du modèle
decisiontree.fit <- train(TARGET_5Yrs ~ .,
                          method = "rpart", 
                          trControl = trControl,
                          tuneLength = 20,
                          metric = "Accuracy",
                          data = train)

decisiontree.fit
# Récupération des résultats
decisiontree.fit_results <- decisiontree.fit$results

plot(decisiontree.fit, xlab="Paramètre de complexité (CP)",ylab="Précision", main="Arbre de décision: Précision en fonction du paramètre de complexité (CP) " )


decision_tree.results <- data.frame(cp=decisiontree.fit_results$cp, error_r=1-decisiontree.fit_results$Accuracy)
min_results_dt <- decision_tree.results[which.min(decision_tree.results$error_r),]
best_cp <- min_results_dt$cp
min_error_dt <- min_results_dt$error_r
cat("La valeur optimale de cp est : ", best_cp, "où le taux d'erreur obtenu est minimal soit :", min_error_dt,".")

# Définition du seed(matricule)
set.seed(2103232)

# Récupération du meilleur modèle
decisiontree.best_fit <- decisiontree.fit$finalModel

# Visualisation de l'arbre de décision
#fancyRpartPlot(decisiontree.best_fit) # Peut faire crasher RStudio en local mais fonctionne sur Google Colab

# Application du modèle sur les données de test
dt.best_cp <- predict(decisiontree.best_fit,test.X, type="class")

# Résultat: matrice de confusion
dt.table <- table(dt.best_cp, test.Y)
dt.table

# Calcul du taux d'erreur
dt.test_error <- mean(dt.best_cp != test.Y )
cat("Le taux d'erreur sur les données de test est : ", dt.test_error, " avec ", best_cp," comme paramètre de complexité.\n")
cat("La sensibilité est de: ",sensitivity(dt.table), "; et la spécificité vaut : ", specificity(dt.table), ".")


######          Random Forest         ###### 

# Définition du seed(matricule)
set.seed(2103232)
# Définition de la valeur max du nombre de prédicteurs choisis au hasard
MTRY <- 19

# Configuration de la méthode (cross-validation) avec K folds
trControl <- trainControl(method = "cv",number = KFOLD)

# Ajustement du modèle
randomForest.fit <- train(TARGET_5Yrs ~ .,
                          method = "rf",  
                          ntree = 500,
                          tuneGrid= expand.grid(.mtry=c(1:MTRY)),
                          trControl = trControl,
                          metric = "Accuracy",
                          data = train)

# Récupération des résultats
randomForest.fit_results <- randomForest.fit$results

plot(randomForest.fit, xlab="Nombre de prédicteurs choisi au hasard",ylab="Précision", main="Random Forest: Précision en fonction du nombre de prédicteurs " )

randomForest.fit
results <- data.frame(mtry=randomForest.fit_results$mtry, error_r=1-randomForest.fit_results$Accuracy)
min_results_rf <- results[which.min(results$error_r),]
best_mtry <- min_results_rf$mtry
min_error_rf <- min_results_rf$error_r
cat("La valeur optimale de mtry est : ", best_mtry, "où le taux d'erreur obtenu est minimal soit :
", min_error_rf,".")

# Définition du seed(matricule)
set.seed(2103232)

# Récupération du meilleur modèle
randomForest.best_fit <- randomForest.fit$finalModel
# Application du modèle sur les données de test
rf.best_mtry <- predict(randomForest.best_fit,test.X)

# Résultat: matrice de confusion
rf.table <- table(rf.best_mtry, test.Y)
rf.table
# Calcul du taux d'erreur
rf.test_error <- mean(rf.best_mtry != test.Y )
cat("Le taux d'erreur sur les données de test est : ", rf.test_error, " avec ", best_mtry," prédicteurs choisis au hasard.\n")
cat("La sensibilité est de: ",sensitivity(rf.table), "; et la spécificité vaut : ", specificity(rf.table), ".")

# Amélioration : Fixer mtry et faire varier le nombre d'arbres
modelstrees <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
ntrees <- seq(0, 1000, by = 100)[-1]

# Boucle sur le nombre d'arbres
for (ntree in ntrees) {
  # Définition du seed(matricule)
  set.seed(2103232)
  
  rf.trees <- train(TARGET_5Yrs ~ .,
                    data = train,
                    method = "rf",
                    metric = "Accuracy",
                    tuneGrid = tuneGrid,
                    trControl = trControl,
                    ntree = ntree)
  key <- toString(ntree)
  modelstrees[[key]] <- rf.trees
}
resultsTree <- resamples(modelstrees)

# Récupération des résultats des différents modèles
res = summary(resultsTree)
df_trees <- data.frame(ntrees)
df_trees$error_r <- 1 - res$statistics$Accuracy[,"Mean"]

# Visualisation du taux d'erreur en fonction du nombre d'arbres
plot(df_trees$ntree, df_trees$error_r, type="b", xlab="Nombre d'arbres", ylab="Taux d'erreur", main="Taux d'erreur en fonction du nombre d'arbres")

# Récupération du meilleur modèle (avec le nombre d'arbres optimal)
besttree_results_rf <- df_trees[which.min(df_trees$error_r),]
best_ntree <- besttree_results_rf$ntrees
best_ntree_error_r <- besttree_results_rf$error_r

# Définition du seed(matricule)
set.seed(2103232)

# Application du modèle sur les données de test
rf.bestmodel.fit <- modelstrees$`800`
rf.bestmodel <- predict(rf.bestmodel.fit,test.X)

# Résultat: matrice de confusion
rf.table <- table(rf.bestmodel, test.Y)
rf.table

# Calcul du taux d'erreur
rf.test_error <- mean(rf.bestmodel != test.Y )
cat("Le taux d'erreur sur les données de test est : ", rf.test_error, " avec ", best_mtry,"prédicteurs et ", best_ntree, "arbres.\n")
cat("La sensibilité est de: ",sensitivity(rf.table), "; et la spécificité vaut : ", specificity(rf.table), ".")

###############################################
###   PREDICTIONS SUR DE NOUVELLES DONNEES  ###
###############################################
GP <- c(82,79)
MIN <- c(28.1,33.7)
PTS <- c(10.2,14.7)
FGM <- c(3.6,5.2)
FGA <- c(8.4,11.3)
FG. <- c(42.8,46.0)
X3P.Made <- c(0.1,0.3)
X3PA <- c(0.4,1.0)
X3P. <- c(25.0,30.0)
FTM <- c(2.8,2.3)
FTA <- c(4.0,3.5)
FT. <- c(70.0,65.7)
OREB <- c(0.7,2.4)
DREB <- c(2.0,2.0)
REB <- c(2.7,4.4)
AST <- c(7.6,1.1)
STL <- c(1.6,0.7)
BLK <- c(0.2,0.3)
TOV <- c(2.0,2.1)

newdata = data.frame(GP,MIN,PTS,FGM,FGA,FG.,X3P.Made, X3PA, X3P.,FTM,FTA,FT.,OREB,DREB,REB,AST,STL,BLK,TOV)
newdata

# Regression Logistique
cat("Prédiction avec le modèle de Regression Logistique:")
predict(lr.fit, newdata)

# Arbre de décision
cat("\nPrédiction avec le modèle d'arbre de décision")
predict(decisiontree.best_fit, newdata,type="class")

# Random Forest
cat("\nPrédiction avec le modèle de Random Forest")
predict(rf.bestmodel.fit, newdata)

         ###############################################
##########   Modèles Avec Variables sélectionnées      #####################
         ###############################################

# Préparation du dataset
reduced_Data$TARGET_5Yrs = factor(reduced_Data$TARGET_5Yrs)
reduced_Data$Name <- NULL

# Partitionnement du dataset en set d'entrainement et de test
train <- reduced_Data[indxTrain,]
test <- reduced_Data[-indxTrain,]

train.X <- train
train.X$TARGET_5Yrs <- NULL
train.Y <- train$TARGET_5Yrs

test.X <- test
test.X$TARGET_5Yrs <- NULL
test.Y <- test$TARGET_5Yrs

# Normalisation des données
trainX <- train[,names(train) != c("TARGET_5Yrs")]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))

######     Regression Logistique      ###### 

# Définition du seed(matricule)
set.seed(2103232)

# Comparaison entre modèle complet et modèle par sélection de variables

### Modèle Complet ###
full.model <- glm(TARGET_5Yrs ~ ., data = train, family = binomial)
as.data.frame(coef(full.model))
# Predictions
lr.prob.full <- full.model %>% predict(test, type = "response")
lr.pred.full <- ifelse(lr.prob.full > 0.5, "0", "1")
# Calcul du taux d'erreur
lr.fullmodel.error_r <- mean(lr.pred.full == test$TARGET_5Yrs)
cat("Regression Logistique (COMPLET): Le taux d'erreur pour le modèle complet est de :", lr.fullmodel.error_r, ".")

### Modèle par sélection de variables ###
step.model <- full.model %>% stepAIC(trace = FALSE)
as.data.frame(coef(step.model))

# Predictions
lr.prob.step <- predict(step.model, test, type = "response")
lr.pred.step <- ifelse(lr.prob.step > 0.5, "0", "1")
# Calcul du taux d'erreur
lr.stepmodel.error_r <- mean(lr.pred.step == test$TARGET_5Yrs)
cat("Regression Logistique (STEPWISE): Le taux d'erreur pour le modèle avec sélection de variable est de :", lr.stepmodel.error_r, ".")

if (lr.stepmodel.error_r < lr.fullmodel.error_r) {
  df_lr <- as.data.frame(coef(step.model))
  
  selected_var <- rownames(df_lr)
  selected_var <- selected_var[selected_var != "(Intercept)"]
  
  lr_var <- which(selected_var %in% colnames(train))
  
  lr_var <- append(lr_var, which(colnames(train)=="TARGET_5Yrs"))
  
  lr_train = train[,c(lr_var)]
  lr_test = test[,c(lr_var)]
  
  lr_test.X <- lr_test
  lr_test.X$TARGET_5Yrs <- NULL
  lr_test.Y <- lr_test$TARGET_5Yrs
  # Définition du seed(matricule)
  set.seed(2103232)
  
  # Configuration de la méthode (cross-validation) avec K folds
  trControl <- trainControl(method = "cv", number = KFOLD)
  
  # Ajustement du modèle complet de régression logistique
  lr.fit.step <- train(TARGET_5Yrs ~ . ,
                       method = "glm",
                       trControl = trControl,
                       metric = "Accuracy",
                       data = lr_train)
  
  lr.fit.step
  # Récupération des résultats
  error_r.LRkfold <- 1 - lr.fit.step$results["Accuracy"][,1]
  cat("Regression Logistique (STEPWISE) K-Fold : \n Le taux derreur obtenu est : ",error_r.LRkfold,".\n")
  
  # Application du modèle sur les données de test
  lr.pred.step <- predict(lr.fit.step,lr_test.X)
  
  # Résultat: matrice de confusion
  lr.step.table <- table(lr.pred.step, lr_test.Y)
  lr.step.table
  # Calcul du taux d'erreur
  lr.test_error.step <- mean(lr.pred.step != lr_test.Y )
  cat("Regression Logistique (STEPWISE): Le taux d'erreur sur les données de test est : ", lr.test_error.step, ".\n")
  cat("La sensibilité est de: ",sensitivity(lr.step.table), "; et la spécificité vaut : ", specificity(lr.step.table), ".")
  
}else {
  # Définition du seed(matricule)
  set.seed(2103232)
  
  # Configuration de la méthode (cross-validation) avec K folds
  trControl <- trainControl(method = "cv", number = KFOLD)
  
  # Ajustement du modèle complet de régression logistique
  lr.fit <- train(TARGET_5Yrs ~ . ,
                  method = "glm",
                  trControl = trControl,
                  metric = "Accuracy",
                  data = train)
  
  lr.fit
  # Récupération des résultats
  error_r.LRkfold <- 1 - lr.fit$results["Accuracy"][,1]
  cat("Regression Logistique (COMPLET) K-Fold : \n Le taux derreur obtenu est : ",error_r.LRkfold,".\n")
  
  # Application du modèle sur les données de test
  lr.pred <- predict(lr.fit,test.X)
  
  # Résultat: matrice de confusion
  lr.table <- table(lr.pred, test.Y)
  lr.table
  # Calcul du taux d'erreur
  lr.test_error <- mean(lr.pred != test.Y )
  cat("Regression Logistique (COMPLET) : Le taux d'erreur sur les données de test est : ", lr.test_error, ".\n")
  cat("La sensibilité est de: ",sensitivity(lr.table), "; et la spécificité vaut : ", specificity(lr.table), ".")
  
}

######  KNN: K plus proches voisins   ###### 

# Définition du seed(matricule)
set.seed(2103232)
# Définition de la valeur max du nombre de voisins
M <- 100
# Configuration de la méthode (cross-validation) avec K folds
trControl <- trainControl(method = "cv",number = KFOLD)
# Ajustement du modèle
knn.fit <- train(TARGET_5Yrs ~ . ,
                 method = "knn",
                 tuneGrid = expand.grid(k = 1:M),
                 trControl = trControl,
                 metric = "Accuracy",
                 data = train)

# Récupération des résultats
knn.fit_results <- knn.fit$results
results <- data.frame(k=knn.fit_results$k, error_r=1-knn.fit_results$Accuracy)
plot(results$k, results$error_r, type="b", col="black", main="KNN: Taux d'erreur en fonction du nombre de voisins", xlab = "Nombre de k voisins", ylab = "Taux d'erreur")

# On détermine le K optimal, pour lequel le taux d'erreur est minimal
min_results <- results[which.min(results$error_r),]
min_K <- min_results$k
min_error <- min_results$error_r
cat("La valeur optimale de k voisins est : ", min_K, "où le taux d'erreur obtenu est minimal soit :
", min_error,".")

# On trace le graphe du taux d'erreur en fonction de k
plot(x=results$k, y=results$error_r, type="b", xlab="k voisins", ylab="Taux d'erreur", ylim=c(0.28,0.39))
title(main="Le taux d'erreur en fonction de k voisins (10-Fold CV)")

# Définition du seed(matricule)
set.seed(2103232)

# Application du modèle sur les données de test
knn_minK <- knn(train.X, test.X, train.Y, k=min_K)

# Résultat: matrice de confusion
knn.table <- table(knn_minK, test.Y)
knn.table

# Calcul du taux d'erreur
knn.test_error <- mean(knn_minK != test.Y )
cat("Le taux d'erreur sur les données de test est : ", knn.test_error, " avec K (nombre de voisins)=", min_K,".\n")
cat("La sensibilité est de: ",sensitivity(knn.table), "; et la spécificité vaut : ", specificity(knn.table), ".")

######       Arbre de décision        ###### 

# Définition du seed(matricule)
set.seed(2103232)

# Configuration de la méthode (cross-validation) avec K folds
trControl <- trainControl(method = "cv",number = KFOLD)

# Ajustement du modèle
decisiontree.fit <- train(TARGET_5Yrs ~ .,
                          method = "rpart", 
                          trControl = trControl,
                          tuneLength = 20,
                          metric = "Accuracy",
                          data = train)

decisiontree.fit
# Récupération des résultats
decisiontree.fit_results <- decisiontree.fit$results

plot(decisiontree.fit, xlab="Paramètre de complexité (CP)",ylab="Précision", main="Arbre de décision: Précision en fonction du paramètre de complexité (CP) " )

decision_tree.results <- data.frame(cp=decisiontree.fit_results$cp, error_r=1-decisiontree.fit_results$Accuracy)
min_results_dt <- decision_tree.results[which.min(decision_tree.results$error_r),]
best_cp <- min_results_dt$cp
min_error_dt <- min_results_dt$error_r
cat("La valeur optimale de cp est : ", best_cp, "où le taux d'erreur obtenu est minimal soit :", min_error_dt,".")

# Définition du seed(matricule)
set.seed(2103232)

# Récupération du meilleur modèle
decisiontree.best_fit <- decisiontree.fit$finalModel

# Visualisation de l'arbre de décision
fancyRpartPlot(decisiontree.best_fit)

# Application du modèle sur les données de test
dt.best_cp <- predict(decisiontree.best_fit,test.X, type="class")

# Résultat: matrice de confusion
dt.table <- table(dt.best_cp, test.Y)
dt.table

# Calcul du taux d'erreur
dt.test_error <- mean(dt.best_cp != test.Y )
cat("Le taux d'erreur sur les données de test est : ", dt.test_error, " avec ", best_cp," comme paramètre de complexité.\n")
cat("La sensibilité est de: ",sensitivity(dt.table), "; et la spécificité vaut : ", specificity(dt.table), ".")

######          Random Forest         ###### 

# Définition du seed(matricule)
set.seed(2103232)
# Définition de la valeur max du nombre de prédicteurs choisis au hasard
MTRY <- 19

# Configuration de la méthode (cross-validation) avec K folds
trControl <- trainControl(method = "cv",number = KFOLD)

# Ajustement du modèle
randomForest.fit <- train(TARGET_5Yrs ~ .,
                          method = "rf",  
                          ntree = 500,
                          tuneGrid= expand.grid(.mtry=c(1:MTRY)),
                          trControl = trControl,
                          metric = "Accuracy",
                          data = train)

# Récupération des résultats
randomForest.fit_results <- randomForest.fit$results

plot(randomForest.fit, xlab="Nombre de prédicteurs choisi au hasard",ylab="Précision", main="Random Forest: Précision en fonction du nombre de prédicteurs " )

randomForest.fit
results <- data.frame(mtry=randomForest.fit_results$mtry, error_r=1-randomForest.fit_results$Accuracy)
min_results_rf <- results[which.min(results$error_r),]
best_mtry <- min_results_rf$mtry
min_error_rf <- min_results_rf$error_r
cat("La valeur optimale de mtry est : ", best_mtry, "où le taux d'erreur obtenu est minimal soit :
", min_error_rf,".")

# Définition du seed(matricule)
set.seed(2103232)

# Récupération du meilleur modèle
randomForest.best_fit <- randomForest.fit$finalModel
# Application du modèle sur les données de test
rf.best_mtry <- predict(randomForest.best_fit,test.X)

# Résultat: matrice de confusion
rf.table <- table(rf.best_mtry, test.Y)
rf.table
# Calcul du taux d'erreur
rf.test_error <- mean(rf.best_mtry != test.Y )
cat("Le taux d'erreur sur les données de test est : ", rf.test_error, " avec ", best_mtry," prédicteurs choisis au hasard.\n")
cat("La sensibilité est de: ",sensitivity(rf.table), "; et la spécificité vaut : ", specificity(rf.table), ".")

# Amélioration : Fixer mtry et faire varier le nombre d'arbres
modelstrees <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
ntrees <- seq(0, 1000, by = 100)[-1]

# Boucle sur le nombre d'arbres
for (ntree in ntrees) {
  # Définition du seed(matricule)
  set.seed(2103232)
  
  rf.trees <- train(TARGET_5Yrs ~ .,
                    data = train,
                    method = "rf",
                    metric = "Accuracy",
                    tuneGrid = tuneGrid,
                    trControl = trControl,
                    ntree = ntree)
  key <- toString(ntree)
  modelstrees[[key]] <- rf.trees
}
resultsTree <- resamples(modelstrees)

# Récupération des résultats des différents modèles
res = summary(resultsTree)
df_trees <- data.frame(ntrees)
df_trees$error_r <- 1 - res$statistics$Accuracy[,"Mean"]

# Visualisation du taux d'erreur en fonction du nombre d'arbres
plot(df_trees$ntree, df_trees$error_r, type="b", xlab="Nombre d'arbres", ylab="Taux d'erreur", main="Taux d'erreur en fonction du nombre d'arbres")

# Récupération du meilleur modèle (avec le nombre d'arbres optimal)
besttree_results_rf <- df_trees[which.min(df_trees$error_r),]
best_ntree <- besttree_results_rf$ntrees
best_ntree_error_r <- besttree_results_rf$error_r

# Définition du seed(matricule)
set.seed(2103232)

# Application du modèle sur les données de test
rf.bestmodel.fit <- modelstrees$`900`
rf.bestmodel <- predict(rf.bestmodel.fit,test.X)

# Résultat: matrice de confusion
rf.table <- table(rf.bestmodel, test.Y)
rf.table

# Calcul du taux d'erreur
rf.test_error <- mean(rf.bestmodel != test.Y )
cat("Le taux d'erreur sur les données de test est : ", rf.test_error, " avec ", best_mtry,"prédicteurs et ", best_ntree, "arbres.\n")
cat("La sensibilité est de: ",sensitivity(rf.table), "; et la spécificité vaut : ", specificity(rf.table), ".")


