# Validation Set

install.packages("rsample")
library(rsample)
mData=read.csv("./nba.csv")

mData = na.omit(mData)
duplicated(mData)
nrow(mData[duplicated(mData$Player),])
mData <- mData[!duplicated(mData$Player),]
mData1 <- mData[,-1]
mData1 <- mData1[,-2]
mData1 <- mData1[,-4]

set.seed(123)

data_split <- initial_split(mData1, prob = 0.80, strata = Salary)
data_train <- training(data_split)
data_test  <-  testing(data_split)

regres_train <- lm(Salary~. - FTr - TS., 
                   data_train)
regres_train1 <- lm(Salary~ . , 
                    data_train)
c(AIC(regres_train),AIC(regres_train1))

pred_0 <- predict(regres_train,newdata = data_test)
MSE0 <- mean((data_test$Salary-pred_0)^2)
pred_1 <- predict(regres_train1,newdata = data_test)
MSE1 <- mean((data_test$Salary-pred_1)^2)
c(MSE0,MSE1)

# Leave-One-Out Cross-Validation
install.packages("glmnet")
install.packages("boot")
library(glmnet)
library (boot)
set.seed(123)

glm.fit1=glm(Salary~. - FTr - TS., mData1, family = gaussian())
coef(glm.fit1)

cv.err = cv.glm(mData1 ,glm.fit1)
cv.err$delta

glm.fit2=glm(Salary~. ,mData1,family = gaussian())

cv.err2 =cv.glm(mData1,glm.fit2)
cv.err2$delta

# K-Fold Cross-Validation

set.seed(123)
cv.err =cv.glm(mData1,glm.fit1,K=10)
cv.err$delta

# Ridge
library(rsample) 
library(glmnet)   
library(dplyr)    
library(ggplot2)

set.seed(123)
ames_split <- initial_split(mData1, prop = .7, strata = "Salary")
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

ames_train_x <- model.matrix(Salary ~ ., ames_train)[, -1]
ames_train_y <- log(ames_train$Salary)

ames_test_x <- model.matrix(Salary ~ ., ames_test)[, -1]
ames_test_y <- log(ames_test$Salary)

dim(ames_train_x)

ames_ridge <- glmnet(
  x = ames_train_x,
  y = ames_train_y,
  alpha = 0
)

plot(ames_ridge, xvar = "lambda")

ames_ridge$lambda %>% head()

coef(ames_ridge)[c("AST.", "MP"), 100]
coef(ames_ridge)[c("AST.", "MP"), 1] 

# Tuning λ
ames_ridge_cv <- cv.glmnet(
  x = ames_train_x,
  y = ames_train_y,
  alpha = 0
)

# plot results
plot(ames_ridge_cv)

min(ames_ridge_cv$cvm)
ames_ridge_cv$lambda.min 

log(ames_ridge_cv$lambda.min)

ames_ridge_cv$cvm[ames_ridge_cv$lambda == ames_ridge_cv$lambda.1se]  # 1 st.error of min MSE

ames_ridge_cv$lambda.1se  # lambda for this MSE

log(ames_ridge_cv$lambda.1se)

plot(ames_ridge, xvar = "lambda")
abline(v = log(ames_ridge_cv$lambda.1se), col = "red", lty = "dashed")

# Ventajas y Desventajas

coef(ames_ridge_cv, s = "lambda.1se") %>%
  broom::tidy() %>%
  filter(row != "(Intercept)") %>%
  top_n(25, wt = abs(value)) %>%
  ggplot(aes(value, reorder(row, value))) +
  geom_point() +
  ggtitle("Top 25 influential variables") +
  xlab("Coefficient") +
  ylab(NULL)

# Lasso

ames_lasso <- glmnet(
  x = ames_train_x,
  y = ames_train_y,
  alpha = 1
)

plot(ames_lasso, xvar = "lambda")

# Tuning - CV

ames_lasso_cv <- cv.glmnet(
  x = ames_train_x,
  y = ames_train_y,
  alpha = 1
)

plot(ames_lasso_cv)

min(ames_lasso_cv$cvm)  
ames_lasso_cv$lambda.min
ames_lasso_cv$cvm[ames_lasso_cv$lambda == ames_lasso_cv$lambda.1se]
ames_lasso_cv$lambda.1se

plot(ames_lasso, xvar = "lambda")
abline(v = log(ames_lasso_cv$lambda.min), col = "red", lty = "dashed")
abline(v = log(ames_lasso_cv$lambda.1se), col = "red", lty = "dashed")

# Ventajas y Desventajas

coef(ames_lasso_cv, s = "lambda.1se") %>%
  tidy() %>%
  filter(row != "(Intercept)") %>%
  ggplot(aes(value, reorder(row, value), color = value > 0)) +
  geom_point(show.legend = FALSE) +
  ggtitle("Influential variables") +
  xlab("Coefficient") +
  ylab(NULL)

min(ames_ridge_cv$cvm)
min(ames_lasso_cv$cvm)

# Elastic Net (Red elástica)

lasso    <- glmnet(ames_train_x, ames_train_y, alpha = 1.0) 
elastic1 <- glmnet(ames_train_x, ames_train_y, alpha = 0.25) 
elastic2 <- glmnet(ames_train_x, ames_train_y, alpha = 0.75) 
ridge    <- glmnet(ames_train_x, ames_train_y, alpha = 0.0)

par(mfrow = c(2, 2), mar = c(6, 4, 6, 2) + 0.1)
plot(lasso, xvar = "lambda", main = "Lasso (Alpha = 1)\n\n\n")
plot(elastic1, xvar = "lambda", main = "Elastic Net (Alpha = .25)\n\n\n")
plot(elastic2, xvar = "lambda", main = "Elastic Net (Alpha = .75)\n\n\n")
plot(ridge, xvar = "lambda", main = "Ridge (Alpha = 0)\n\n\n")

# Tuning

fold_id <- sample(1:10, size = length(ames_train_y), replace=TRUE)

tuning_grid <- tibble::tibble(
  alpha      = seq(0, 1, by = .1),
  mse_min    = NA,
  mse_1se    = NA,
  lambda_min = NA,
  lambda_1se = NA
)
tuning_grid

for(i in seq_along(tuning_grid$alpha)) {
  
  # fit CV model for each alpha value
  fit <- cv.glmnet(ames_train_x, ames_train_y, alpha = tuning_grid$alpha[i], foldid = fold_id)
  
  # extract MSE and lambda values
  tuning_grid$mse_min[i]    <- fit$cvm[fit$lambda == fit$lambda.min]
  tuning_grid$mse_1se[i]    <- fit$cvm[fit$lambda == fit$lambda.1se]
  tuning_grid$lambda_min[i] <- fit$lambda.min
  tuning_grid$lambda_1se[i] <- fit$lambda.1se
}

tuning_grid

tuning_grid %>%
  mutate(se = mse_1se - mse_min) %>%
  ggplot(aes(alpha, mse_min)) +
  geom_line(size = 2) +
  geom_ribbon(aes(ymax = mse_min + se, ymin = mse_min - se), alpha = .25) +
  ggtitle("MSE ± one standard error")
