rm(list = ls())
library(Matrix)
library(mvnfast)
library(invgamma)
library(truncnorm)
library(doParallel)
library(foreach)
library(coda)
library(tidyverse)
source("../algo_group_fusion_horseshoe.R")
RMSE <- function(x, y) return(sqrt(mean((x-y)^2)))
mode <- function(x) density(x)$x[which.max(density(x)$y)]
cores <- 3

# Paramètres à faire varier
n <- c(500) 
G <- 10 
p <- c(150*G, 300*G)
se2 <- c(2)
rho <- c(0.95)

nb_rep <- 5
fold <- 1:10
rep <- 1:nb_rep

pars_tmp <- expand.grid(n = n, G = G, p = p, se2 = se2, rho = rho, rep = rep)

pars <- expand.grid(n = n, G = G, p = p, se2 = se2, rho = rho, rep = rep, fold = fold)
dim(pars)

save(pars, file = "pars.Rdata")

id.cv <- list()
for(k in rep) id.cv[[k]] <- sample(max(fold), n, replace = TRUE)
save(id.cv, file = "id.cv.Rdata")



# Simulation --------------------------------------------------------------

system("mkdir data")
k=1
for(k in 1:nrow(pars_tmp)){
  print(k)
  g <- function(t,n) {sin(4*t/n -2) + 2*exp(-30*(4*t/n -2)^2)} # fct of Faulkner
  #_____ valeurs simulées
  sim <- list()
  n <- sim$n <- pars_tmp$n[k]
  p <- sim$p <- pars_tmp$p[k]
  G <- sim$G <- G  
  group_id <- sim$group_id <- kronecker(1:G, rep(1, p/G))
  sim$mu <-  5
  beta <- rep(0, p)
  beta[group_id == 1] <- g(1:table(group_id)[1], table(group_id)[1])
  beta[group_id == 3] <- c(rep(0, floor(table(group_id)[3]*0.1)), rep(1, table(group_id)[3] - floor(table(group_id)[3]*0.1)))
  beta[group_id == 5] <- c(rep(0, floor(table(group_id)[5]*0.8)), rep(1, table(group_id)[5] - floor(table(group_id)[5]*0.8)))
  # beta[group_id == 7] <- rep(0.7, table(group_id)[7])
  sim$beta <- beta 
  
  plot(sim$beta, t='l')
  abline(v=c(0, cumsum(table(group_id))), lty = 3)
  
  sim$rho <- pars_tmp$rho[k]
  sim$se2 <- pars_tmp$se2[k]
  
  
  
  Gamma <- diag(p/G);  for(i in 1:(p/G)) for(j in 1:(p/G)) Gamma[i, j] <- sim$rho ^ abs(i-j) # correlation of covariable
  Gamma <- kronecker(diag(G), Gamma)
  X <- rmvn(n, rep(0, p), Gamma)
  Z <- matrix(0, n, 5)                      # Design matrix for 1 random effect
  A <- diag(5)                              # correlation structure of the random effect
  Y <- as.vector(4 + X %*% beta + Z %*% c(rmvn(1, rep(0, ncol(A)), 3 * A)) + rmvn(n, 0, sqrt(sim$se2)))
  save(Y, X, Z, A, sim, file = paste0("data/data_sim_n=", pars_tmp$n[k], "_G=", pars_tmp$G[k], "_p=",
                                      pars_tmp$p[k], "_se2=", pars_tmp$se2[k], "_rho=", pars_tmp$rho[k],
                                      "_rep=", pars_tmp$rep[k], ".Rdata"))
}






############################## Fit ########################################
doParallel::registerDoParallel(cores = min(cores, nrow(pars)))
system("mkdir results")


# Fused_HS_3_levels -------------------------------------------------------
system("mkdir results/Fused_HS_3_levels")

k=1
list_chain <- foreach::foreach(k = 1:nrow(pars), .verbose = FALSE) %dopar% {
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))

  n <- nrow(Y); p <- ncol(X)
  nb_group <- pars$G[k]
  degree = 1

  print(k)
  settings        <- list()
  settings$niter  <- 10000
  settings$burnin <- 5000
  settings$thin   <- 10
  length_group    <- rep(p/nb_group, nb_group)
  settings$group  <- rep(1:nb_group, times = length_group)
  settings$d      <- degree
  settings$D      <- lapply(table(settings$group), function(p) diff(diag(p), differences = settings$d ))

  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X[which(id.cv[[pars$rep[k]]] != fold), ]


  chain <- group_fused_HS_MCMC(y=y_train, X=X_train, settings=settings, var_sel = TRUE, b2_ = TRUE)

  lppd <- sum(log(apply(chain$prob_Y_theta, 2, mean)))
  Pwaic <- sum(apply(log(chain$prob_Y_theta), 2, var))
  waic <- -2 * lppd + 2 * Pwaic
  chain$lppd <- lppd
  chain$Pwaic <- Pwaic
  chain$waic <- waic
  save(chain, settings, file = paste0("results/Fused_HS_3_levels/chain_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                                      pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                                      "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata") )
  return()
}

# Fused_HS_2_levels -------------------------------------------------------
system("mkdir results/Fused_HS_2_levels")

k=1
list_chain <- foreach::foreach(k = 1:nrow(pars), .verbose = FALSE) %dopar% {
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))
  
  n <- nrow(Y); p <- ncol(X)
  nb_group <- pars$G[k]
  degree = 1
  
  print(k)
  settings        <- list()
  settings$niter  <- 10000
  settings$burnin <- 5000
  settings$thin   <- 10
  length_group    <- rep(p/nb_group, nb_group)
  settings$group  <- rep(1:nb_group, times = length_group)
  settings$d      <- degree
  settings$D      <- lapply(table(settings$group), function(p) diff(diag(p), differences = settings$d ))
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X[which(id.cv[[pars$rep[k]]] != fold), ]
  
  
  chain <- group_fused_HS_MCMC(y=y_train, X=X_train, settings=settings, var_sel = TRUE, b2_ = FALSE)
  
  lppd <- sum(log(apply(chain$prob_Y_theta, 2, mean)))
  Pwaic <- sum(apply(log(chain$prob_Y_theta), 2, var))
  waic <- -2 * lppd + 2 * Pwaic
  chain$lppd <- lppd
  chain$Pwaic <- Pwaic
  chain$waic <- waic
  save(chain, settings, file = paste0("results/Fused_HS_2_levels/chain_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                                      pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                                      "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata") )
  return()
}


# Fusion_HS_3_levels -------------------------------------------------------
system("mkdir results/Fusion_HS_3_levels")

k=1
list_chain <- foreach::foreach(k = 1:nrow(pars), .verbose = FALSE) %dopar% {
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))
  
  n <- nrow(Y); p <- ncol(X)
  nb_group <- pars$G[k]
  degree = 1
  
  print(k)
  settings        <- list()
  settings$niter  <- 10000
  settings$burnin <- 5000
  settings$thin   <- 10
  length_group    <- rep(p/nb_group, nb_group)
  settings$group  <- rep(1:nb_group, times = length_group)
  settings$d      <- degree
  settings$D      <- lapply(table(settings$group), function(p) diff(diag(p), differences = settings$d ))
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X[which(id.cv[[pars$rep[k]]] != fold), ]
  
  
  chain <- group_fused_HS_MCMC(y=y_train, X=X_train, settings=settings, var_sel = FALSE, b2_ = TRUE)
  
  lppd <- sum(log(apply(chain$prob_Y_theta, 2, mean)))
  Pwaic <- sum(apply(log(chain$prob_Y_theta), 2, var))
  waic <- -2 * lppd + 2 * Pwaic
  chain$lppd <- lppd
  chain$Pwaic <- Pwaic
  chain$waic <- waic
  save(chain, settings, file = paste0("results/Fusion_HS_3_levels/chain_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                                      pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                                      "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata") )
  return()
}

# Fusion_HS_2_levels -------------------------------------------------------
system("mkdir results/Fusion_HS_2_levels")

k=1
list_chain <- foreach::foreach(k = 1:nrow(pars), .verbose = FALSE) %dopar% {
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))
  
  n <- nrow(Y); p <- ncol(X)
  nb_group <- pars$G[k]
  degree = 1
  
  print(k)
  settings        <- list()
  settings$niter  <- 10000
  settings$burnin <- 5000
  settings$thin   <- 10
  length_group    <- rep(p/nb_group, nb_group)
  settings$group  <- rep(1:nb_group, times = length_group)
  settings$d      <- degree
  settings$D      <- lapply(table(settings$group), function(p) diff(diag(p), differences = settings$d ))
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X[which(id.cv[[pars$rep[k]]] != fold), ]
  
  
  chain <- group_fused_HS_MCMC(y=y_train, X=X_train, settings=settings, var_sel = FALSE, b2_ = FALSE)
  
  lppd <- sum(log(apply(chain$prob_Y_theta, 2, mean)))
  Pwaic <- sum(apply(log(chain$prob_Y_theta), 2, var))
  waic <- -2 * lppd + 2 * Pwaic
  chain$lppd <- lppd
  chain$Pwaic <- Pwaic
  chain$waic <- waic
  save(chain, settings, file = paste0("results/Fusion_HS_2_levels/chain_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                                      pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                                      "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata") )
  return()
}



# SPLS --------------------------------------------------------------------
system("mkdir results/SPLS")
library(spls)

k=1
for(k in 1:nrow(pars)){
  print(k)
  load("id.cv.Rdata")
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))

  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X[which(id.cv[[pars$rep[k]]] == fold), ]

  cv <- cv.spls( X_train, y_train, eta = seq(0, .999, 0.1), K = c(1:20),
                 fold = 5)
  fit <- spls(X_train, y_train, eta = cv$eta.opt, K = cv$K.opt)
  beta_hat <- fit$betahat
  mu_hat <- fit$mu

  plot(beta_hat, t='l'); abline(v=cumsum(table(rep(1:10, each = pars$p[k]/10))))
  lines(sim$beta, col=2)

  y_pred <- predict.spls(fit, X_test)
  rmse <- RMSE(y_pred, y_test); rmse
  print(rmse)

  save(fit, mu_hat, beta_hat, cv, y_pred, y_test, X_test, rmse, file = paste0("results/SPLS/fit_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                                                                              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                                                                              "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata") )
  # return()
}



# cMCP  ------------------------------------------------------------

system("mkdir results/cMCP")
doParallel::registerDoParallel(cores = cores)


alpha <- 1
lambda <- seq(0.05, 2, 0.05)

library(grpreg)

k=1
for(k in 1:nrow(pars)){

  print(k)
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))

  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X[which(id.cv[[pars$rep[k]]] == fold), ]

  id.cv.2 <- sample(1:5, length(y_train), replace = TRUE)
  pars_rmse <- expand.grid(alpha=alpha, lambda=lambda)

  i <- 1; j <- 1
  # for(i in 1:nrow(rmse_p)){
  rmse_cv <- foreach::foreach(i = 1:nrow(pars_rmse), .verbose = FALSE, .combine="rbind") %dopar% {
    # print(i)
    rmse_tmp <- rep(NA, 5)

    for(j in 1:5){
      X_train_tmp <-  X_train[which(id.cv.2 != j), ]
      y_train_tmp <-  y_train[which(id.cv.2 != j)]
      X_test_tmp <-  X_train[which(id.cv.2 == j), ]
      y_test_tmp <-  y_train[which(id.cv.2 == j)]

      fit_tmp <- grpreg(X_train_tmp, y_train_tmp,
                        group=rep(1:10, each = pars$p[k]/10),
                        penalty="cMCP",
                        lambda = pars_rmse$lambda[i],
                        alpha = pars_rmse$alpha[i])

      y_hat_tmp <- predict(fit_tmp, X_test_tmp)
      rmse_tmp[j] <- RMSE(y_hat_tmp, y_test_tmp)
    }
    # summary(rmse_tmp)
    # pars_rmse$rmse_p[i] <- mean(rmse_tmp)
    return(data.frame(alpha=pars_rmse$alpha[i], lambda=pars_rmse$lambda[i], rmse_p = mean(rmse_tmp)))
  }
  id_tmp <- which.min(rmse_cv$rmse_p)
  rmse_cv[id_tmp, ]
  lambda_opt <- rmse_cv$lambda[id_tmp]; print(paste("lambda_opt = ", lambda_opt))
  alpha_opt <- 1 #rmse_cv$alpha[id_tmp]; print(paste("alpha_opt = ", alpha_opt))

  fit <- grpreg(X_train, y_train, group=rep(1:10, each = pars$p[k]/10), penalty="cMCP", alpha = alpha_opt, lambda = lambda_opt)

  mu_hat <- coef(fit)[1]; mu_hat
  beta_hat <- coef(fit)[-1]
  # plot(beta_hat, t='l'); abline(v=cumsum(table(rep(1:10, each = pars$p[k]/10)))); lines(sim$beta, col=2)

  y_hat <- predict(fit, X=as.matrix(X_test))
  rmse_p <- RMSE(y_hat, y_test); rmse_p

  # y_hat <- mu_hat + as.matrix(X_test) %*% (beta_hat)
  # rmse_p <- RMSE(y_hat, y_test); rmse_p

  print(rmse_p)
  save(fit, alpha_opt, alpha, lambda_opt, lambda, mu_hat, beta_hat, y_hat, y_test, X_test, rmse_p, rmse_cv,
       file = paste0("results/cMCP/fit_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                     pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                     "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}





# cMCP + ridge  ------------------------------------------------------------
system("mkdir results/cMCP_Ridge")
doParallel::registerDoParallel(cores = cores)


alpha <- c(1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
lambda <- seq(0.05, 1, 0.05)

library(grpreg)

k=1
for(k in 1:nrow(pars)){

  print(k)
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))

  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X[which(id.cv[[pars$rep[k]]] == fold), ]

  id.cv.2 <- sample(1:5, length(y_train), replace = TRUE)
  pars_rmse <- expand.grid(alpha=alpha, lambda=lambda)

  i <- 1; j <- 1
  # for(i in 1:nrow(rmse_p)){
  rmse_cv <- foreach::foreach(i = 1:nrow(pars_rmse), .verbose = FALSE, .combine="rbind") %dopar% {
    # print(i)
    rmse_tmp <- rep(NA, 5)

    for(j in 1:5){
      X_train_tmp <-  X_train[which(id.cv.2 != j), ]
      y_train_tmp <-  y_train[which(id.cv.2 != j)]
      X_test_tmp <-  X_train[which(id.cv.2 == j), ]
      y_test_tmp <-  y_train[which(id.cv.2 == j)]

      fit_tmp <- grpreg(X_train_tmp, y_train_tmp,
                        group=rep(1:10, each = pars$p[k]/10),
                        penalty="cMCP",
                        lambda = pars_rmse$lambda[i],
                        alpha = pars_rmse$alpha[i])

      y_hat_tmp <- predict(fit_tmp, X_test_tmp)
      rmse_tmp[j] <- RMSE(y_hat_tmp, y_test_tmp)
    }
    # summary(rmse_tmp)
    # pars_rmse$rmse_p[i] <- mean(rmse_tmp)
    return(data.frame(alpha=pars_rmse$alpha[i], lambda=pars_rmse$lambda[i], rmse_p = mean(rmse_tmp)))
  }
  id_tmp <- which.min(rmse_cv$rmse_p)
  rmse_cv[id_tmp, ]
  lambda_opt <- rmse_cv$lambda[id_tmp]; print(paste("lambda_opt = ", lambda_opt))
  alpha_opt <- rmse_cv$alpha[id_tmp]; print(paste("alpha_opt = ", alpha_opt))

  fit <- grpreg(X_train, y_train, group=rep(1:10, each = pars$p[k]/10), penalty="cMCP", alpha = alpha_opt, lambda = lambda_opt)

  mu_hat <- coef(fit)[1]; mu_hat
  beta_hat <- coef(fit)[-1]
  # plot(beta_hat, t='l'); abline(v=cumsum(table(rep(1:10, each = pars$p[k]/10)))); lines(sim$beta, col=2)

  y_hat <- predict(fit, X=as.matrix(X_test))
  rmse_p <- RMSE(y_hat, y_test); rmse_p

  # y_hat <- mu_hat + as.matrix(X_test) %*% (beta_hat)
  # rmse_p <- RMSE(y_hat, y_test); rmse_p

  print(rmse_p)
  save(fit, alpha_opt, alpha, lambda_opt, lambda, mu_hat, beta_hat, y_hat, y_test, X_test, rmse_p, rmse_cv,
       file = paste0("results/cMCP_Ridge/fit_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                     pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                     "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}




# Elastic-Net  ------------------------------------------------------------
system("mkdir results/EN")
doParallel::registerDoParallel(cores = cores)

library(glmnet)

alpha <- c(seq(0, 0.3, 0.01), seq(0.4, 1, 0.1))#, seq(0.5, 1, 0.1)) #c(1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
lambda <- seq(0.1, 5, 0.2)


k=1
for(k in 1:nrow(pars)){
  
  print(k)
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X[which(id.cv[[pars$rep[k]]] == fold), ]
  
  id.cv.2 <- sample(1:5, length(y_train), replace = TRUE)
  pars_rmse <- expand.grid(alpha=alpha, lambda=lambda)
  
  i <- 1; j <- 1
  # for(i in 1:nrow(rmse_p)){
  rmse_cv <- foreach::foreach(i = 1:nrow(pars_rmse), .verbose = FALSE, .combine="rbind") %dopar% {
    # print(i)
    rmse_tmp <- rep(NA, 5)
    
    for(j in 1:5){
      X_train_tmp <-  X_train[which(id.cv.2 != j), ]
      y_train_tmp <-  y_train[which(id.cv.2 != j)]
      X_test_tmp <-  X_train[which(id.cv.2 == j), ]
      y_test_tmp <-  y_train[which(id.cv.2 == j)]
      
      fit_tmp <- glmnet(X_train_tmp, y_train_tmp, 
                        lambda = pars_rmse$lambda[i],
                        alpha = pars_rmse$alpha[i])
      
      y_hat_tmp <- predict(fit_tmp, X_test_tmp)
      rmse_tmp[j] <- RMSE(y_hat_tmp, y_test_tmp)
    }
    # summary(rmse_tmp)
    # pars_rmse$rmse_p[i] <- mean(rmse_tmp)
    return(data.frame(alpha=pars_rmse$alpha[i], lambda=pars_rmse$lambda[i], rmse_p = mean(rmse_tmp)))
  }
  id_tmp <- which.min(rmse_cv$rmse_p)
  rmse_cv[id_tmp, ]
  lambda_opt <- rmse_cv$lambda[id_tmp]; print(paste("lambda_opt = ", lambda_opt))
  alpha_opt <- rmse_cv$alpha[id_tmp]; print(paste("alpha_opt = ", alpha_opt))
  
  # library(ggplot2)
  # ggplot(data = rmse_cv, aes(x=alpha, y=lambda, fill=rmse_p)) + 
  #   geom_tile(color = "white")+
  #   scale_fill_gradient(name = "RMSEp",
  #                       low = "white",
  #                       high = "black")
  # scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
  #                      midpoint = mean(range(rmse_cv$rmse_p)), limit = range(rmse_cv$rmse_p), space = "Lab", 
  #                      name="RMSEp")
  
  
  fit <- glmnet(X_train, y_train, alpha = alpha_opt, lambda = lambda_opt)
  
  mu_hat <- coef(fit)[1]; mu_hat
  beta_hat <- coef(fit)[-1]
  plot(beta_hat, t='l'); abline(v=cumsum(table(rep(1:10, each = pars$p[k]/10)))); lines(sim$beta, col=2)
  
  y_hat <- predict(fit, newx=as.matrix(X_test))
  rmse_p <- RMSE(y_hat, y_test); rmse_p
  
  # y_hat <- mu_hat + as.matrix(X_test) %*% (beta_hat)
  # rmse_p <- RMSE(y_hat, y_test); rmse_p
  
  print(rmse_p)
  save(fit, alpha_opt, alpha, lambda_opt, lambda, mu_hat, beta_hat, y_hat, y_test, X_test, rmse_p, rmse_cv, 
       file = paste0("results/EN/fit_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                     pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                     "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}




############################# Concat #################################

# SPLSR ---------------------------------------------------------------------

files <- system("ls results/SPLS/" , intern = TRUE)
present <- rep(NA, 100)
for(k in 1:100){
  present[k] <- any(files == paste0("fit_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                                    pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                                    "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}
which(!present)
sum(present)

k=1
library(spls)
rmse_SPLS <- res_SPLS <- NULL
for(k in 1:nrow(pars)){
  print(k)
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))
  load(paste0("results/SPLS/fit_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  
  beta_hat <- fit$betahat
  # plot(beta_hat, t='l', col = 4)
  # rmse
  
  rmse_SPLS <- rbind(rmse_SPLS, data.frame( meth = 'SPLSR',
                                            p = pars$p[k],
                                            rep  = pars$rep[k],
                                            fold = pars$fold[k],
                                            rmse = rmse))
  
  res_SPLS <- rbind(res_SPLS, 
                    data.frame(
                      meth = "SPLSR",
                      coef = "none",
                      diff = "none",
                      n = pars$n[k], G = pars$G[k], p=pars$p[k], rep = pars$rep[k], se2 = pars$se2[k], rho = pars$rho[k], fold = pars$fold[k],
                      x  = 1:sim$p,
                      Group = rep(1:10, each=pars$p[k]/pars$G[k]),
                      rmse_P = rmse,
                      beta = beta_hat,
                      IC_inf = NA, IC_sup = NA, eiq = NA,
                      beta_true = sim$beta)
  )
  
}



# cMCP ---------------------------------------------------------------------

files <- system("ls results/cMCP/" , intern = TRUE)
present <- rep(NA, 100)
for(k in 1:100){
  present[k] <- any(files == paste0("fit_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                                    pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                                    "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}
which(!present)
sum(present)

k=3
pars[k, ]
library(grpreg)
rmse_cMCP <- res_cMCP <- NULL
for(k in 1:nrow(pars)){
  print(k)
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))
  load(paste0("results/cMCP/fit_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  
  
  rmse_cMCP <- rbind(rmse_cMCP, data.frame( meth = 'cMCP',
                                            p = pars$p[k],
                                            rep  = pars$rep[k],
                                            fold = pars$fold[k],
                                            rmse = rmse_p))
  
  res_cMCP <- rbind(res_cMCP, 
                    data.frame(
                      meth = "cMCP",
                      coef = "none",
                      diff = "none",
                      n = pars$n[k], G = pars$G[k], p=pars$p[k], rep = pars$rep[k], se2 = pars$se2[k], rho = pars$rho[k], fold = pars$fold[k],
                      x  = 1:sim$p,
                      Group = rep(1:10, each=pars$p[k]/pars$G[k]),
                      rmse_P = rmse_p,
                      beta = beta_hat,
                      IC_inf = NA, IC_sup = NA, eiq = NA,
                      beta_true = sim$beta)
  )
  
}



# cMCP_Ridge ---------------------------------------------------------------------

files <- system("ls results/cMCP_Ridge/" , intern = TRUE)
present <- rep(NA, 100)
for(k in 1:100){
  present[k] <- any(files == paste0("fit_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                                    pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                                    "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}
which(!present)
sum(present)

k=1
library(grpreg)
rmse_cMCP_Ridge <- res_cMCP_Ridge <- NULL
for(k in 1:nrow(pars)){
  print(k)
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))
  load(paste0("results/cMCP_Ridge/fit_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  
 
  rmse_cMCP_Ridge <- rbind(rmse_cMCP_Ridge, data.frame( meth = 'cMCP_Ridge',
                                                        p = pars$p[k],
                                                        rep  = pars$rep[k],
                                                        fold = pars$fold[k],
                                                        rmse = rmse_p))
  
  res_cMCP_Ridge <- rbind(res_cMCP_Ridge, 
                          data.frame(
                            meth = "cMCP_Ridge",
                            coef = "none",
                            diff = "none",
                            n = pars$n[k], G = pars$G[k], p=pars$p[k], rep = pars$rep[k], se2 = pars$se2[k], rho = pars$rho[k], fold = pars$fold[k],
                            x  = 1:sim$p,
                            Group = rep(1:10, each=pars$p[k]/pars$G[k]),
                            rmse_P = rmse_p,
                            beta = beta_hat,
                            IC_inf = NA, IC_sup = NA, eiq = NA,
                            beta_true = sim$beta)
  )
  
}

# EN ---------------------------------------------------------------------

files <- system("ls results/EN/" , intern = TRUE)
present <- rep(NA, 100)
for(k in 1:100){
  present[k] <- any(files == paste0("fit_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                                    pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                                    "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}
which(!present)
sum(present)

k=1
library(grpreg)
rmse_EN <- res_EN <- NULL
for(k in 1:nrow(pars)){
  print(k)
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))
  load(paste0("results/EN/fit_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  
  
  rmse_EN <- rbind(rmse_EN, data.frame( meth = 'EN',
                                        p = pars$p[k],
                                        rep  = pars$rep[k],
                                        fold = pars$fold[k],
                                        rmse = rmse_p))
  
  res_EN <- rbind(res_EN, 
                  data.frame(
                    meth = "ENR",
                    coef = "none",
                    diff = "none",
                    n = pars$n[k], G = pars$G[k], p=pars$p[k], rep = pars$rep[k], se2 = pars$se2[k], rho = pars$rho[k], fold = pars$fold[k],
                    x  = 1:sim$p,
                    Group = rep(1:10, each=pars$p[k]/pars$G[k]),
                    rmse_P = rmse_p,
                    beta = beta_hat,
                    IC_inf = NA, IC_sup = NA, eiq = NA,
                    beta_true = sim$beta)
  )
  
}


# Fusion_HS_2_levels ---------------------------------------------------------------

files <- system("ls results/Fusion_HS_2_levels/" , intern = TRUE)
present <- rep(NA, 100)
for(k in 1:100){
  present[k] <- any(files == paste0("chain_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                                    pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                                    "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}
which(!present)
sum(present)

k=1
res_Fusion_HS_2_levels <- foreach(k = which(present), .combine = rbind, .verbose = FALSE) %dopar% {
  
  print(k)
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))
  load(paste0("results/Fusion_HS_2_levels/chain_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X[which(id.cv[[pars$rep[k]]] == fold), ]
  
  beta_hat <- colMeans(chain$beta)
  mu_hat <- mean(chain$mu) 
  y_pred <- mu_hat + X_test %*% beta_hat
  rmse <- RMSE(y_pred, y_test)
  
  return( data.frame(
    meth = "Fusion_HS_2_levels",
    coef = "none",
    diff = "HS_2_levels",
    n = pars$n[k], G = pars$G[k], p=pars$p[k], rep = pars$rep[k], se2 = pars$se2[k], rho = pars$rho[k], fold = pars$fold[k],
    x  = 1:sim$p, 
    Group = rep(1:10, each=pars$p[k]/pars$G[k]),
    rmse_P = rmse,
    beta = colMeans(chain$beta),
    IC_inf = apply(chain$beta, 2, quantile, 0.025),
    IC_sup = apply(chain$beta, 2, quantile, 0.975),
    eiq = apply(chain$beta, 2, quantile, 0.975)-
      apply(chain$beta, 2, quantile, 0.025),
    beta_true = sim$beta))
}





# Fusion_HS_3_levels ---------------------------------------------------------------
files <- system("ls results/Fusion_HS_3_levels/" , intern = TRUE)
present <- rep(NA, 100)
for(k in 1:100){
  present[k] <- any(files == paste0("chain_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                                    pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                                    "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}
which(!present)
sum(present)

res_Fusion_HS_3_levels <- foreach(k = which(present), .combine = rbind, .verbose = FALSE) %dopar% {
  
  print(k)
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))
  load(paste0("results/Fusion_HS_3_levels/chain_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X[which(id.cv[[pars$rep[k]]] == fold), ]
  
  beta_hat <- colMeans(chain$beta)
  mu_hat <- mean(chain$mu) 
  y_pred <- mu_hat + X_test %*% beta_hat
  rmse <- RMSE(y_pred, y_test)
  
  return( data.frame(
    meth = "Fusion_HS_3_levels",
    coef = "none",
    diff = "HS_3_levels",
    n = pars$n[k], G = pars$G[k], p=pars$p[k], rep = pars$rep[k], se2 = pars$se2[k], rho = pars$rho[k], fold = pars$fold[k],
    x  = 1:sim$p,
    Group = rep(1:10, each=pars$p[k]/pars$G[k]),
    rmse_P = rmse,
    beta = colMeans(chain$beta),
    IC_inf = apply(chain$beta, 2, quantile, 0.025),
    IC_sup = apply(chain$beta, 2, quantile, 0.975),
    eiq = apply(chain$beta, 2, quantile, 0.975)-
      apply(chain$beta, 2, quantile, 0.025),
    beta_true = sim$beta))
}



# Fused_HS_2_levels ---------------------------------------------------------------

k=1
res_Fused_HS_2_levels <- foreach(k = 1:nrow(pars), .combine = rbind, .verbose = FALSE) %dopar% {
  
  print(k)
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))
  load(paste0("results/Fused_HS_2_levels/chain_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X[which(id.cv[[pars$rep[k]]] == fold), ]
  
  beta_hat <- colMeans(chain$beta)
  mu_hat <- mean(chain$mu) 
  y_pred <- mu_hat + X_test %*% beta_hat
  rmse <- RMSE(y_pred, y_test)
  
  return( data.frame(
    meth = "Fused_HS_2_levels",
    coef = "local_ind",
    diff = "HS_2_levels",
    n = pars$n[k], G = pars$G[k], p=pars$p[k], rep = pars$rep[k], se2 = pars$se2[k], rho = pars$rho[k], fold = pars$fold[k],
    x  = 1:sim$p,
    Group = rep(1:10, each=pars$p[k]/pars$G[k]),
    rmse_P = rmse,
    beta = colMeans(chain$beta),
    IC_inf = apply(chain$beta, 2, quantile, 0.025),
    IC_sup = apply(chain$beta, 2, quantile, 0.975),
    eiq = apply(chain$beta, 2, quantile, 0.975)-
      apply(chain$beta, 2, quantile, 0.025),
    beta_true = sim$beta))
}



# Fused_HS_3_levels ---------------------------------------------------------------
files <- system("ls results/Fused_HS_3_levels/" , intern = TRUE)
present <- rep(NA, 100)
for(k in 1:100){
  present[k] <- any(files == paste0("chain_n=", pars$n[k], "_G=", pars$G[k], "_p=",
                                    pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
                                    "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}
which(!present)
sum(present)

res_Fused_HS_3_levels <- foreach(k = which(present), .combine = rbind, .verbose = FALSE) %dopar% {
  
  print(k)
  load(paste0("data/data_sim_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], ".Rdata"))
  load(paste0("results/Fused_HS_3_levels/chain_n=", pars$n[k], "_G=", pars$G[k], "_p=",
              pars$p[k], "_se2=", pars$se2[k], "_rho=", pars$rho[k],
              "_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X[which(id.cv[[pars$rep[k]]] == fold), ]
  
  beta_hat <- colMeans(chain$beta)
  mu_hat <- mean(chain$mu) 
  y_pred <- mu_hat + X_test %*% beta_hat
  rmse <- RMSE(y_pred, y_test)
  
  return( data.frame(
    meth = "Fused_HS_3_levels",
    coef = "local_ind",
    diff = "HS_2_levels",
    n = pars$n[k], G = pars$G[k], p=pars$p[k], rep = pars$rep[k], se2 = pars$se2[k], rho = pars$rho[k], fold = pars$fold[k],
    x  = 1:sim$p,
    Group = rep(1:10, each=pars$p[k]/pars$G[k]),
    rmse_P = rmse,
    beta = colMeans(chain$beta),
    IC_inf = apply(chain$beta, 2, quantile, 0.025),
    IC_sup = apply(chain$beta, 2, quantile, 0.975),
    eiq = apply(chain$beta, 2, quantile, 0.975)-
      apply(chain$beta, 2, quantile, 0.025),
    beta_true = sim$beta))
}


res_beta <- rbind(res_Fused_HS_3_levels, res_Fusion_HS_3_levels,
                  res_Fused_HS_2_levels, res_Fusion_HS_2_levels,
                  res_SPLS, res_EN, res_cMCP, res_cMCP_Ridge)

save(res_beta, res_SPLS, 
     res_EN, res_cMCP, res_cMCP_Ridge,
     res_Fusion_HS_3_levels, res_Fused_HS_3_levels, 
     res_Fusion_HS_2_levels, res_Fused_HS_2_levels,  
     file = "results.Rdata")


# ggplot ------------------------------------------------------------------

load("results.Rdata")
levels(res_beta$meth)
# Rename
levels(res_beta$meth) <- c("group fused HS 3L", 
                           "group fusion HS 3L", 
                           "group fused HS 2L", 
                           "group fusion HS 2L", 
                           "SPLSR", "ENR", 
                           "cMCP", "cMCP_Ridge")
levels(res_beta$meth)




colnames(res_beta)
res_beta2 <- aggregate(list(beta=res_beta[, "beta"], 
                            IC_inf = res_beta[, "IC_inf"],
                            IC_sup = res_beta[, "IC_sup"]),
                       by = list(meth = res_beta$meth, 
                                 beta_true = res_beta$beta_true, 
                                 x = res_beta$x, p = res_beta$p), 
                       FUN = mean)
head(res_beta2)

res_beta_tmp <- res_beta
IC_inf <- aggregate(list(IC_inf = res_beta_tmp[, "beta"]),
                    by = list(meth = res_beta_tmp$meth, 
                              beta_true = res_beta_tmp$beta_true, 
                              x = res_beta_tmp$x, p = res_beta_tmp$p), 
                    FUN = quantile, prob=0.025)
IC_sup <- aggregate(list(IC_sup = res_beta_tmp[, "beta"]),
                    by = list(meth = res_beta_tmp$meth, 
                              beta_true = res_beta_tmp$beta_true, 
                              x = res_beta_tmp$x, p = res_beta_tmp$p), 
                    FUN = quantile, prob=0.975)

res_beta2[, "IC_inf"] <- IC_inf$IC_inf
res_beta2[, "IC_sup"] <- IC_sup$IC_sup
res_beta2$eiq <- res_beta2$IC_sup - res_beta2$IC_inf


ggplot(data = res_beta2 %>% filter(p==1500, x<453, meth %in% c("cMCP")), aes(x =x, y=beta, group= meth)) + 
  geom_rect(data=color.df1, mapping=aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="blue", alpha=0.1, inherit.aes = FALSE)+
  geom_rect(data=color.df2, mapping=aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="red", alpha=0.1, inherit.aes = FALSE)+
  geom_rect(data=color.df3, mapping=aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="green", alpha=0.1, inherit.aes = FALSE)+
  geom_line(aes(x= x, y = beta_true), colour="red") + 
  geom_line(colour = "black") +
  facet_wrap(~meth) + theme(legend.position = "none") + 
  # ylim(-1.2, 2) +  xlab("t") + 
  geom_ribbon(aes(ymin=IC_inf, ymax=IC_sup), alpha=0.4)



lim_inf <- cumsum(rep(150, 10))
xmin <- c(1, lim_inf[-10])[seq(2, 10, 2)]
xmax <- lim_inf[seq(2, 10, 2)]
color.df=data.frame(xmin=xmin, xmax=xmax, ymin=-Inf, ymax=Inf)

for(i in 1:nrow(res_beta2)){
  res_beta2$IC_inf[i] <- max(res_beta2$IC_inf[i], -1.2)
  res_beta2$IC_sup[i] <- min(res_beta2$IC_sup[i], 2)
}


plot_1500 <- grid.arrange(
  ggplot(data = res_beta2 %>% filter(p==1500, meth %in% c("group fused HS 3L", "group fusion HS 3L")), aes(x =x, y=beta, group= meth)) + 
    geom_rect(data=color.df, mapping=aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="blue", alpha=0.1, inherit.aes = FALSE)+
    geom_line(aes(x= x, y = beta_true), colour="red") + 
    geom_line(colour = "black") +
    facet_wrap(~meth) + theme(legend.position = "none") + 
    ylim(-1.2, 2) +  xlab("t") + 
    geom_ribbon(aes(ymin=IC_inf, ymax=IC_sup), alpha=0.4),
  # 
  ggplot(data = res_beta2 %>% filter(p==1500, meth %in% c("group fused HS 2L", "group fusion HS 2L")), aes(x =x, y=beta, group= meth)) +
    geom_rect(data=color.df, mapping=aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="blue", alpha=0.1, inherit.aes = FALSE)+
    geom_ribbon(aes(ymin=IC_inf, ymax=IC_sup), alpha=0.4) +
    geom_line(aes(x= x, y = beta_true), colour="red") + 
    geom_line(colour = "black") +  xlab("t") + 
    facet_wrap(~meth) + theme(legend.position = "none") + 
    ylim(-1.2, 2),
  #
  ggplot(data = res_beta2 %>% filter(p==1500, meth %in% c("SPLSR", "ENR")), aes(x =x, y=beta, group= meth)) + 
    geom_rect(data=color.df, mapping=aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="blue", alpha=0.1, inherit.aes = FALSE)+
    geom_line(aes(x= x, y = beta_true), colour="red") + 
    geom_line(colour = "black") +
    facet_wrap(~meth) + theme(legend.position = "none") + 
    ylim(-1.2, 2) +  xlab("t") + 
    geom_ribbon(aes(ymin=IC_inf, ymax=IC_sup), alpha=0.4),
  #
  ggplot(data = res_beta2 %>% filter(p==1500, meth %in% c("cMCP", "cMCP_Ridge")), aes(x =x, y=beta, group= meth)) + 
    geom_rect(data=color.df, mapping=aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="blue", alpha=0.1, inherit.aes = FALSE)+
    geom_line(aes(x= x, y = beta_true), colour="red") + 
    geom_line(colour = "black") +
    facet_wrap(~meth) + theme(legend.position = "none") + 
    ylim(-1.2, 2) +
    xlab("t") + 
    geom_ribbon(aes(ymin=IC_inf, ymax=IC_sup), alpha=0.4),
  ncol = 1
)
ggsave("Figures/estimation_p=1500.pdf", plot = plot_1500, width = 8, height = 11)
ggsave("Figures/estimation_p=1500.png", plot = plot_1500, device = "png", scale = 1.5, width = 11, height = 12, units = "cm")



lim_inf <- cumsum(rep(300, 10))
xmin <- c(1, lim_inf[-10])[seq(2, 10, 2)]
xmax <- lim_inf[seq(2, 10, 2)]
color.df=data.frame(xmin=xmin, xmax=xmax, ymin=-Inf, ymax=Inf)


plot_3000 <- grid.arrange(
  ggplot(data = res_beta2 %>% filter(p==3000, meth %in% c("group fused HS 2L", "group fused HS 3L")), aes(x =x, y=beta, group= meth)) +
    geom_rect(data=color.df, mapping=aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="blue", alpha=0.1, inherit.aes = FALSE)+
    geom_line(aes(x= x, y = beta_true), colour="red") + 
    geom_line(colour = "black") +
    facet_wrap(~meth) + theme(legend.position = "none") + 
    ylim(-1.2, 2) + xlab("t") + 
    geom_ribbon(aes(ymin=IC_inf, ymax=IC_sup), alpha=0.4),
 #
  ggplot(data = res_beta2 %>% filter(p==3000, meth %in% c("SPLSR", "ENR")), aes(x =x, y=beta, group= meth)) + 
    geom_rect(data=color.df, mapping=aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="blue", alpha=0.1, inherit.aes = FALSE)+
    geom_line(aes(x= x, y = beta_true), colour="red") + 
    geom_line(colour = "black") +
    facet_wrap(~meth) + theme(legend.position = "none") + 
    ylim(-1.2, 2) + xlab("t") + 
    geom_ribbon(aes(ymin=IC_inf, ymax=IC_sup), alpha=0.4),
  #
  ggplot(data = res_beta2 %>% filter(p==3000, meth %in% c("cMCP", "cMCP_Ridge")), aes(x =x, y=beta, group= meth)) + 
    geom_rect(data=color.df, mapping=aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="blue", alpha=0.1, inherit.aes = FALSE)+
    geom_line(aes(x= x, y = beta_true), colour="red") + 
    geom_line(colour = "black") +
    facet_wrap(~meth) + theme(legend.position = "none") + 
    ylim(-1.2, 2) +
    xlab("t") + 
    geom_ribbon(aes(ymin=IC_inf, ymax=IC_sup), alpha=0.4),
  ncol = 1
)
ggsave("Figures/estimation_p=3000.pdf", plot_3000, device = "pdf", scale = 1, width = 8, height = 10)
ggsave("Figures/estimation_p=3000.png", plot_3000, device = "png", scale = 1.5, width = 10, height = 10, units = "cm")


colnames(res_beta)



colnames(res_beta)
rmse_df <- aggregate(list(rmse_P = res_beta[, c("rmse_P")]), by = list(meth = res_beta$meth, p = res_beta$p, rep = res_beta$rep), FUN = mean, na.rm = TRUE)
rmse_df2 <- aggregate(list(rmse_P = rmse_df[, "rmse_P"]), by = list(meth = rmse_df$meth, p = rmse_df$p), FUN = mean, na.rm = TRUE)
rmse_df2













# TABLE --------------------------------------------------------------------
load("results.Rdata")
levels(res_beta$meth)
# Rename
levels(res_beta$meth) <- c("group fused HS 3L", 
                           "group fusion HS 3L", 
                           "group fused HS 2L", 
                           "group fusion HS 2L", 
                           "SPLSR", "ENR", 
                           "cMCP", "cMCP_Ridge")
levels(res_beta$meth)



# for Bayesian approaches, a variable is selected if zero does not belong to regression coefficient credible interval
res_beta_bayesian <- res_beta %>% filter(! meth %in% c("PLSR", "SPLSR", "SCGLR", "EN", "cMCP", "cMCP_Ridge")) 
res_beta_bayesian$beta_sel = 1*(res_beta_bayesian$IC_inf * res_beta_bayesian$IC_sup >= 0)
res_beta_bayesian$beta = res_beta_bayesian$beta * res_beta_bayesian$beta_sel

# For frequentist approaches, a variable is selected if its regression coefficient is different to zero
res_beta_freq <- res_beta %>% filter(meth %in% c("SPLSR", "EN", "cMCP", "cMCP_Ridge")) 
res_beta_freq$beta_sel = 1*(res_beta_freq$beta != 0)

res_beta <- rbind(res_beta_bayesian, res_beta_freq)



res_beta$FP <- 1*with(res_beta, beta_sel == TRUE & beta_true == 0)
res_beta$FN <- 1*with(res_beta, beta_sel == FALSE & beta_true != 0)
res_beta$TP <- 1*with(res_beta, beta_sel == TRUE & beta_true !=0)
res_beta$TN <- 1*with(res_beta, beta_sel == FALSE & beta_true == 0)

# mean aggregation of the results over the different repetitions and folds:
res_beta_mean <- aggregate(res_beta[, c("rmse_P", "beta", "IC_inf", "IC_sup", "eiq", "beta_sel", "FP", "FN", "TN", "TP")],
                           by = list(meth = res_beta$meth,
                                     beta_true = res_beta$beta_true,
                                     x = res_beta$x, p = res_beta$p),
                           FUN = mean, na.rm=TRUE)

# Confident intervals 
res_beta_freq <- res_beta 
IC_inf <- aggregate(list(IC_inf = res_beta_freq[, "beta"]),
                    by = list(meth = res_beta_freq$meth, 
                              beta_true = res_beta_freq$beta_true, 
                              x = res_beta_freq$x, p = res_beta_freq$p), 
                    FUN = quantile, prob=0.025)
IC_sup <- aggregate(list(IC_sup = res_beta_freq[, "beta"]),
                    by = list(meth = res_beta_freq$meth, 
                              beta_true = res_beta_freq$beta_true, 
                              x = res_beta_freq$x, p = res_beta_freq$p), 
                    FUN = quantile, prob=0.975)

res_beta_mean[, "IC_inf"] <- IC_inf$IC_inf
res_beta_mean[, "IC_sup"] <- IC_sup$IC_sup
res_beta_mean$eiq <- res_beta_mean$IC_sup - res_beta_mean$IC_inf




# p = 1500 --------------------------------------------------------------------

res_beta_tmp_mean <- res_beta_mean %>% filter(p==1500)

res_rmse_P = by(res_beta_tmp_mean$rmse_P, res_beta_tmp_mean$meth, mean, na.rm = TRUE)
res_rmse_beta = by(res_beta_tmp_mean, res_beta_tmp_mean$meth, function(x) {RMSE(x$beta,x$beta_true)} )
res_beta_tmp_v <- res_beta_tmp_mean %>% filter(beta_true != 0, x < (p/10 +3))
res_rmse_beta_v = by(res_beta_tmp_v, res_beta_tmp_v$meth, function(x) {RMSE(x$beta, x$beta_true)} )
res_beta_tmp_c <- res_beta_tmp_mean %>% filter(beta_true != 0, x > (p/10 +3))
res_rmse_beta_c = by(res_beta_tmp_c, res_beta_tmp_c$meth, function(x) {RMSE(x$beta, x$beta_true)} )
res_beta_tmp_1 <- res_beta_tmp_mean %>% filter(beta_true != 0)
res_rmse_beta_1 = by(res_beta_tmp_1, res_beta_tmp_1$meth, function(x) {RMSE(x$beta, x$beta_true)} )
res_beta_tmp_0 <- res_beta_tmp_mean %>% filter(beta_true == 0)
res_rmse_beta_0 = by(res_beta_tmp_0, res_beta_tmp_0$meth, function(x) {RMSE(x$beta, x$beta_true)} )

#  False negatives, false positives, true negatives, true positives and MCC
FP <- res_FP_beta <- by(res_beta_tmp_mean, res_beta_tmp_mean$meth, function(x) {sum(x$FP)} )
FN <- res_FN_beta <- by(res_beta_tmp_mean, res_beta_tmp_mean$meth, function(x) {sum(x$FN)} )
TN <- res_TN_beta <- by(res_beta_tmp_mean, res_beta_tmp_mean$meth, function(x) {sum(x$TN)} )
TP <- res_TP_beta <- by(res_beta_tmp_mean, res_beta_tmp_mean$meth, function(x) {sum(x$TP)} )
MCC <- (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))




tab_mean <- data.frame(t(rbind(RMSE = round(as.numeric(res_rmse_P), 2),
                               # RMSE_beta = round(as.numeric(res_rmse_beta), 4),
                               RMSE_1 = round(as.numeric(res_rmse_beta_1), 3), 
                               # RMSE_V = round(as.numeric(res_rmse_beta_v), 4),
                               # RMSE_C = round(as.numeric(res_rmse_beta_c), 4),
                               RMSE_0 = round(res_rmse_beta_0, 3), 
                               CIW = round(res_eiq_median, 2), 
                               FP=round(res_FP_beta/(1500-315)*100, 1),
                               FN=round(res_FN_beta/315*100, 1), 
                               MCC = round(MCC, 2))))

tab_mean_1500







# p = 3000 --------------------------------------------------------------------

res_beta_tmp_mean <- res_beta_mean %>% filter(p==3000)

res_rmse_P = by(res_beta_tmp_mean$rmse_P, res_beta_tmp_mean$meth, mean, na.rm = TRUE)
res_rmse_beta = by(res_beta_tmp_mean, res_beta_tmp_mean$meth, function(x) {RMSE(x$beta,x$beta_true)} )
res_beta_tmp_v <- res_beta_tmp_mean %>% filter(beta_true != 0, x < (p/10 +3))
res_rmse_beta_v = by(res_beta_tmp_v, res_beta_tmp_v$meth, function(x) {RMSE(x$beta, x$beta_true)} )
res_beta_tmp_c <- res_beta_tmp_mean %>% filter(beta_true != 0, x > (p/10 +3))
res_rmse_beta_c = by(res_beta_tmp_c, res_beta_tmp_c$meth, function(x) {RMSE(x$beta, x$beta_true)} )
res_beta_tmp_1 <- res_beta_tmp_mean %>% filter(beta_true != 0)
res_rmse_beta_1 = by(res_beta_tmp_1, res_beta_tmp_1$meth, function(x) {RMSE(x$beta, x$beta_true)} )
res_beta_tmp_0 <- res_beta_tmp_mean %>% filter(beta_true == 0)
res_rmse_beta_0 = by(res_beta_tmp_0, res_beta_tmp_0$meth, function(x) {RMSE(x$beta, x$beta_true)} )

#  False negatives, false positives, true negatives, true positives and MCC
FP <- res_FP_beta <- by(res_beta_tmp_mean, res_beta_tmp_mean$meth, function(x) {sum(x$FP)} )
FN <- res_FN_beta <- by(res_beta_tmp_mean, res_beta_tmp_mean$meth, function(x) {sum(x$FN)} )
TN <- res_TN_beta <- by(res_beta_tmp_mean, res_beta_tmp_mean$meth, function(x) {sum(x$TN)} )
TP <- res_TP_beta <- by(res_beta_tmp_mean, res_beta_tmp_mean$meth, function(x) {sum(x$TP)} )
MCC <- (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))



tab_mean <- data.frame(t(rbind(RMSE = round(as.numeric(res_rmse_P), 2),
                               # RMSE_b = round(as.numeric(res_rmse_beta), 4),
                               RMSE_1 = round(as.numeric(res_rmse_beta_1), 3), 
                               # RMSE_V = round(as.numeric(res_rmse_beta_v), 4), 
                               # RMSE_C = round(as.numeric(res_rmse_beta_c), 4), 
                               RMSE_0 = round(res_rmse_beta_0, 3), 
                               CIW = round(res_eiq_median, 2), 
                               FP=round(res_FP_beta/(3000-630)*100, 2),
                               FN=round(res_FN_beta/630*100, 2),  
                               MCC = round(MCC, 2))))


tab_mean_3000




# GROUP 1500 -------------------------------------------------------------------
# Calculation of group false negatives, false positives, true negatives, true positives and MCC for p=1500
tmp <- res_beta %>% filter(p==1500) %>% filter(! meth %in% c("PLSR", "SPLSR", "SCGLR", "EN", "cMCP", "cMCP_Ridge"))
tmp$selected <- 1 * (sign(tmp$IC_inf) * sign(tmp$IC_sup) > 0)

colnames(tmp)
colnames(freq_meth_1500)

freq_meth_1500$fold = 1
freq_meth_1500$selected <- freq_meth_1500$beta_sel
tmp <- rbind(tmp[, c("meth", "Group", "rep", "fold", "selected")], freq_meth_1500[, c("meth", "Group", "rep", "fold", "selected")])

group_selection <- aggregate(
  list(selected = tmp$selected), 
  by = list(meth = tmp$meth, Group = tmp$Group, rep = tmp$rep, fold=tmp$fold), 
  FUN = function(x) 1*(sum(x) > 0))

tmp_non_zero_group <- group_selection %>% filter(Group %in% c(1, 3, 5))
VP <- aggregate(
  list(VP = tmp_non_zero_group$selected), 
  by = list(meth = tmp_non_zero_group$meth, rep = tmp_non_zero_group$rep, fold=tmp_non_zero_group$fold), 
  FUN = sum)
VP


tmp_zero_group <- group_selection %>% filter(! Group %in% c(1, 3, 5))
FP <- aggregate(
  list(FP = tmp_zero_group$selected), 
  by = list(meth = tmp_zero_group$meth, rep = tmp_zero_group$rep, fold=tmp_zero_group$fold), 
  FUN = sum)
FP

group_selection <- VP
group_selection$FN <- 3 - group_selection$VP 
group_selection$FP <- FP$FP
group_selection$TP <- group_selection$VP
group_selection$TN <- 10-group_selection$FP
group_selection$MCC_G <- (group_selection$TP*group_selection$TN - group_selection$FP*group_selection$FN)/sqrt((group_selection$TP+group_selection$FP)*(group_selection$TP+group_selection$FN)*(group_selection$TN+group_selection$FP)*(group_selection$TN+group_selection$FN))

group_selection_mean <- aggregate(
  list(FP = group_selection$FP, FN = group_selection$FN, VP = group_selection$VP, MCC_G = group_selection$MCC_G), 
  by = list(meth = group_selection$meth), 
  FUN = mean)
group_selection_mean$FP_G <- group_selection_mean$FP/7 *100
group_selection_mean$FN_G <- group_selection_mean$FN/3*100
group_selection_mean$VP_G <- group_selection_mean$VP/3*100
group_selection_mean
rownames(group_selection_mean) <- group_selection_mean$meth


group_selection_mean_1500 <- round(group_selection_mean[, c("FP_G", "FN_G", "MCC_G")], 2)
group_selection_mean_1500[, c("FP_G", "FN_G")] <- round(group_selection_mean_1500[, c("FP_G", "FN_G")])
group_selection_mean_1500



# GROUP 3000-------------------------------------------------------------------
# Calculation of group false negatives, false positives, true negatives, true positives and MCC for p=3000
tmp <- res_beta %>% filter(p==3000) %>% filter(! meth %in% c("PLSR", "SPLSR", "SCGLR", "EN", "cMCP", "cMCP_Ridge"))
tmp$selected <- 1 * (sign(tmp$IC_inf) * sign(tmp$IC_sup) > 0)

colnames(tmp)
colnames(freq_meth_3000)

freq_meth_3000$fold = 1
freq_meth_3000$selected <- freq_meth_3000$beta_sel
tmp <- rbind(tmp[, c("meth", "Group", "rep", "fold", "selected")], freq_meth_3000[, c("meth", "Group", "rep", "fold", "selected")])

group_selection <- aggregate(
  list(selected = tmp$selected), 
  by = list(meth = tmp$meth, Group = tmp$Group, rep = tmp$rep, fold=tmp$fold), 
  FUN = function(x) 1*(sum(x) > 0))

tmp_non_zero_group <- group_selection %>% filter(Group %in% c(1, 3, 5))
VP <- aggregate(
  list(VP = tmp_non_zero_group$selected), 
  by = list(meth = tmp_non_zero_group$meth, rep = tmp_non_zero_group$rep, fold=tmp_non_zero_group$fold), 
  FUN = sum)
VP


tmp_zero_group <- group_selection %>% filter(! Group %in% c(1, 3, 5))
FP <- aggregate(
  list(FP = tmp_zero_group$selected), 
  by = list(meth = tmp_zero_group$meth, rep = tmp_zero_group$rep, fold=tmp_zero_group$fold), 
  FUN = sum)
FP


group_selection <- VP
group_selection$FN <- 3 - group_selection$VP 
group_selection$FP <- FP$FP
group_selection$TP <- group_selection$VP
group_selection$TN <- 10-group_selection$FP
group_selection$MCC_G <- (group_selection$TP*group_selection$TN - group_selection$FP*group_selection$FN)/sqrt((group_selection$TP+group_selection$FP)*(group_selection$TP+group_selection$FN)*(group_selection$TN+group_selection$FP)*(group_selection$TN+group_selection$FN))

group_selection_mean <- aggregate(
  list(FP = group_selection$FP, FN = group_selection$FN, VP = group_selection$VP, MCC_G = group_selection$MCC_G), 
  by = list(meth = group_selection$meth), 
  FUN = mean)
group_selection_mean$FP_G <- group_selection_mean$FP/7 *100
group_selection_mean$FN_G <- group_selection_mean$FN/3*100
group_selection_mean$VP_G <- group_selection_mean$VP/3*100
group_selection_mean
rownames(group_selection_mean) <- group_selection_mean$meth


group_selection_mean_3000 <- round(group_selection_mean[, c("FP_G", "FN_G", "MCC_G")], 2)
group_selection_mean_3000[, c("FP_G", "FN_G")] <- round(group_selection_mean_3000[, c("FP_G", "FN_G")])
group_selection_mean_3000



# table -------------------------------------------------------------------
tab_mean_1500

tab_mean_1500 <- cbind(tab_mean_1500, group_selection_mean_1500)
tab_mean_1500$FN <- round(tab_mean_1500$FN)
library(xtable)
print(xtable(tab_mean_1500, digits = 3))


tab_mean_3000 <- cbind(tab_mean_3000, group_selection_mean_3000)
round(tab_mean_3000, 3)

library(xtable)
print(xtable(tab_mean_3000, digits = 3))















