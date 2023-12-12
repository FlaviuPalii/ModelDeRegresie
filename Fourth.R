# Instalare pachete (dacă nu sunt instalate)
install.packages(c("tidyverse", "lattice", "rsample", "vip"))

# Incărcare pachete
library(tidyverse)
library(lattice)
library(rsample)
library(vip)
library(caret)
library(lmtest)


# Calea către fișierul CSV
data <- "ames.csv"

# Încărcarea datelor din fișierul CSV
ames <- read.csv(data)

# Verificarea încărcării datelor
glimpse(ames)

ames <- ames %>%
  select_if(~ !any(is.na(.)))  # Selectează doar coloanele fără date lipsă


# Splitarea setului de date
set.seed(123)
split <- initial_split(ames, prop = 0.7, strata = 'price')
ames_train <- training(split)
ames_test <- testing(split)

# Regresie liniară simplă și multiplă
model1 <- lm(price ~ Year.Built + Lot.Area, data = ames_train)
summary(model1)

model2 <- lm(price ~ Year.Built + Overall.Qual + Lot.Area, data = ames_train)
summary(model2)


# Evaluare metrici de eroare
rmse_model1 <- sqrt(mean(model1$residuals^2))
mse_model1 <- mean(model1$residuals^2)

rmse_model2 <- sqrt(mean(model2$residuals^2))
mse_model2 <- mean(model2$residuals^2)

# Validare încrucișată pentru modelele 1 și 2
set.seed(123)
cv_model1 <- train(
  form = price ~ Year.Built + Lot.Area,
  data = ames_train,
  method = 'lm',
  trControl = trainControl(method = 'cv', number = 10)
)

set.seed(123)
cv_model2 <- train(
  form = price ~ Year.Built + Lot.Area + Overall.Qual,
  data = ames_train,
  method = 'lm',
  trControl = trainControl(method = 'cv', number = 10)
)

# Evaluare performanță modele
summary(resamples(list(model1 = cv_model1, model2 = cv_model2)))

# Analiza condițiilor
# Liniaritatea pentru modelul 1
plot(model1)

plot(model1, which = c(1, 3))

# Normalitatea reziduurilor
qqPlot(model1$residuals)

# Test pentru independența reziduurilor
dwtest(model1)


# Liniaritatea pentru modelul 2
plot(model2)

plot(model2, which = c(1, 3))

# Normalitatea reziduurilor
qqPlot(model2$residuals)

# Test pentru independența reziduurilor
dwtest(model2)

# Interpretare caracteristici (pentru modelul 3)
vip(cv_model2, num_features = 10)



# Regresie liniară inițială cu Year si LotArea
single_level_factors <- sapply(ames_train, function(x) is.factor(x) && length(levels(x)) <= 1)
ames_train <- ames_train[, !single_level_factors]
# Modelul inițial cu Year.Built și Lot.Area
initial_model <- lm(price ~ Year.Built + Lot.Area, data = ames_train)
final_model <- step(initial_model, direction = "both", trace = FALSE)
summary(final_model)


# Adăugarea variabilei Overall.Qual la modelul existent
initial_model_with_qual <- lm(price ~ Overall.Qual + Lot.Area + Year.Built, data = ames_train)
final_model_with_qual <- step(initial_model_with_qual, direction = "both", trace = FALSE)
summary(final_model_with_qual)

# Evaluarea metricilor pentru modelul 1
rmse_model1 <- sqrt(mean(model1$residuals^2))
mse_model1 <- mean(model1$residuals^2)
rsquared_model1 <- summary(model1)$r.squared

cat("Metrics for Model 1:\n")
cat("RMSE:", rmse_model1, "\n")
cat("MSE:", mse_model1, "\n")
cat("R-squared:", rsquared_model1, "\n\n")

# Evaluarea metricilor pentru modelul 2
rmse_model2 <- sqrt(mean(model2$residuals^2))
mse_model2 <- mean(model2$residuals^2)
rsquared_model2 <- summary(model2)$r.squared

cat("Metrics for Model 2:\n")
cat("RMSE:", rmse_model2, "\n")
cat("MSE:", mse_model2, "\n")
cat("R-squared:", rsquared_model2, "\n\n")

# Evaluarea metricilor pentru final_model
rmse_final_model <- sqrt(mean(final_model$residuals^2))
mse_final_model <- mean(final_model$residuals^2)
rsquared_final_model <- summary(final_model)$r.squared

cat("Metrics for Final Model (Stepwise):\n")
cat("RMSE:", rmse_final_model, "\n")
cat("MSE:", mse_final_model, "\n")
cat("R-squared:", rsquared_final_model, "\n\n")

# Evaluarea metricilor pentru final_model_with_qual
rmse_final_model_with_qual <- sqrt(mean(final_model_with_qual$residuals^2))
mse_final_model_with_qual <- mean(final_model_with_qual$residuals^2)
rsquared_final_model_with_qual <- summary(final_model_with_qual)$r.squared

cat("Metrics for Final Model (Stepwise with Overall.Qual):\n")
cat("RMSE:", rmse_final_model_with_qual, "\n")
cat("MSE:", mse_final_model_with_qual, "\n")
cat("R-squared:", rsquared_final_model_with_qual, "\n\n")

# Facem predicțiile
predictions_final_model <- predict(final_model, newdata = ames_test)
predictions_final_model_with_qual <- predict(final_model_with_qual, newdata = ames_test)

# Creăm diagrame de dispersie pentru compararea valorilor reale cu predicțiile
plot(ames_test$price, predictions_final_model, main = "Predicții vs. Valori reale pentru Modelul Final (Stepwise)",
     xlab = "Valori reale", ylab = "Predicții", col = "blue")
abline(0, 1, col = "red")  # Linia de referință pentru valorile reale egale cu predicțiile
legend("topleft", legend = "Valori reale = Predicții", col = "red", lty = 1, cex = 0.8)

plot(ames_test$price, predictions_final_model_with_qual, main = "Predicții vs. Valori reale pentru Modelul Final (Stepwise cu Overall.Qual)",
     xlab = "Valori reale", ylab = "Predicții", col = "blue")
abline(0, 1, col = "red")  # Linia de referință pentru valorile reale egale cu predicțiile
legend("topleft", legend = "Valori reale = Predicții", col = "red", lty = 1, cex = 0.8)


