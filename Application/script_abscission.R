rm(list = ls())
library(Matrix)
library(mvnfast)
library(invgamma)
library(truncnorm)
library(doParallel)
library(foreach)
library(coda)
source("../algo_group_fusion_horseshoe.R")
RMSE <- function(x, y) return(sqrt(mean((x-y)^2)))
cores <- 3


load("abscission.Rdata")

id_g <- names(X_list); id_g
X <- do.call(cbind, X_list)
X_scale <- scale(X)

rep <- 5
folds <- 5
n <- length(DFD)

pars <- expand.grid(rep = 1:rep, fold = 1:folds)

id.cv <- list()
for(k in 1:rep) id.cv[[k]] <- sample(folds, n, replace = TRUE)
# save(id.cv, file = "id_cv.Rdata")
# load("id_cv.Rdata")

system("mkdir results")


# Group Fused HS 3L ----------------------------------------------------------

system("mkdir results/fused_HS_3L")

fit_fused_HS <- group_fused_HS(
  y=DFD, X=X_scale, selection = TRUE, degree = 1, nb_group = length(id_g),
  model = 'gaussian', niter = 15000, burnin = 5000,
  thin = 20, rep = rep, cores = cores, CV = folds,
  id.cv = id.cv, path = "results/fused_HS_3L")

save(fit_fused_HS, file = "fit_fused_HS_3L.Rdata")
plot.effect(fit_fused_HS, t='l', col = 4)



# Group Fusion HS 3L ------------------------------------------------------
system("mkdir results/fusion_HS_3L")

fit_fusion_HS <- group_fused_HS(
  y=DFD, X=X_scale, selection = FALSE, degree = 1, nb_group = length(id_g),
  model = 'gaussian', niter = 15000, burnin = 5000,
  thin = 20, rep = rep, cores = cores, CV = folds,
  id.cv = id.cv, path = "results/fusion_HS_3L")

save(fit_fusion_HS, file = "fit_fusion_HS_3L.Rdata")
plot.effect(fit_fusion_HS, t='l', col = 4)

# Group Fused HS 2L ----------------------------------------------------------

system("mkdir results/fused_HS_2L")

fit_fused_HS <- group_fused_HS(
  y=DFD, X=X_scale, selection = TRUE, degree = 1, nb_group = length(id_g),
  b2_ = FALSE,
  model = 'gaussian', niter = 15000, burnin = 5000,
  thin = 20, rep = rep, cores = cores, CV = folds,
  id.cv = id.cv, path = "results/fused_HS_2L")

save(fit_fused_HS, file = "fit_fused_HS_2L.Rdata")
plot.effect(fit_fused_HS, t='l', col = 4)



# Group Fusion HS 2L ------------------------------------------------------
system("mkdir results/fusion_HS_2L")

fit_fusion_HS <- group_fused_HS(
  y=DFD, X=X_scale, selection = FALSE, degree = 1, nb_group = length(id_g),
  b2_ = FALSE,
  model = 'gaussian', niter = 15000, burnin = 5000,
  thin = 20, rep = rep, cores = cores, CV = folds,
  id.cv = id.cv, path = "results/fusion_HS_2L")

save(fit_fusion_HS, file = "fit_fusion_HS_2L.Rdata")
plot.effect(fit_fusion_HS, t='l', col = 4)


# SPLS --------------------------------------------------------------------
system("mkdir results/SPLS")

library(spls)

k=1
list_chain <- foreach::foreach(k = 1:nrow(pars), .verbose = FALSE) %dopar% {
  # list_chain <- foreach::foreach(k = which(!present), .verbose = FALSE) %dopar% {
  print(k)

  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X_scale[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X_scale[which(id.cv[[pars$rep[k]]] == fold), ]

  cv <- cv.spls( X_train, y_train, eta = seq(0, .999, 0.02), K = c(1:5), fold = 2 )
  # print(cv$K.opt)
  fit <- spls(X_train, y_train, eta = cv$eta.opt, K = cv$K.opt )
  beta_hat <- fit$betahat
  plot(beta_hat, t='l')

  y_pred <- predict.spls(fit, X_test)
  rmse <- RMSE(y_pred, y_test); print(rmse)

  save(fit, beta_hat, cv, y_pred, y_test, rmse, file = paste0("results/SPLS/fit_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata") )
  return()
}


# cMCP  ------------------------------------------------------------
system("mkdir results/cMCP")

cores <- 3
doParallel::registerDoParallel(cores = cores)


alpha <- 1
lambda <- seq(0.15, 1, 0.05)

library(grpreg)

k=1
for(k in 1:nrow(pars)){
  
  print(k)
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X_scale[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X_scale[which(id.cv[[pars$rep[k]]] == fold), ]
  
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
                        group=rep(1:9, each = 121),
                        penalty="cMCP",
                        lambda = pars_rmse$lambda[i],
                        alpha = pars_rmse$alpha[i])
      
      y_hat_tmp <- predict(fit_tmp, as.matrix(X_test_tmp))
      rmse_tmp[j] <- RMSE(y_hat_tmp, y_test_tmp)
    }
    # summary(rmse_tmp)
    # pars_rmse$rmse_p[i] <- mean(rmse_tmp)
    return(data.frame(alpha=pars_rmse$alpha[i], lambda=pars_rmse$lambda[i], rmse_p = mean(rmse_tmp)))
  }
  head(rmse_cv)
  plot(rmse_cv$lambda, rmse_cv$rmse_p)
  id_tmp <- which.min(rmse_cv$rmse_p)
  rmse_cv[id_tmp, ]
  lambda_opt <- rmse_cv$lambda[id_tmp]; print(paste("lambda_opt = ", lambda_opt))
  alpha_opt <- 1 #rmse_cv$alpha[id_tmp]; print(paste("alpha_opt = ", alpha_opt))
  
  fit <- grpreg(X_train, y_train, group=rep(1:9, each = 121), penalty="cMCP", alpha = alpha_opt, lambda = lambda_opt)
  
  mu_hat <- coef(fit)[1]; mu_hat
  beta_hat <- coef(fit)[-1]
  # plot(beta_hat, t='l'); abline(v=cumsum(rep(121, 9))); 
  
  y_hat <- predict(fit, X=as.matrix(X_test))
  rmse_p <- RMSE(y_hat, y_test); rmse_p
  
  # y_hat <- mu_hat + as.matrix(X_test) %*% (beta_hat)
  # rmse_p <- RMSE(y_hat, y_test); rmse_p
  
  print(rmse_p)
  save(fit, alpha_opt, alpha, lambda_opt, lambda, mu_hat, beta_hat, y_hat, y_test, X_test, rmse_p, rmse_cv,
       file = paste0("results/cMCP/fit_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  rm(fit, alpha_opt, lambda_opt, mu_hat, beta_hat, y_hat, y_test, X_test, rmse_p, rmse_cv)
}





# cMCP + ridge  ------------------------------------------------------------
system("mkdir results/cMCP_Ridge")

cores <- 3
doParallel::registerDoParallel(cores = cores)


alpha <- c(0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
lambda <- seq(0.15, 1.5, 0.05)

library(grpreg)

k=1
for(k in 1:nrow(pars)){
  
  print(k)
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X_scale[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X_scale[which(id.cv[[pars$rep[k]]] == fold), ]
  
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
                        group=rep(1:9, each = 121),
                        penalty="cMCP",
                        lambda = pars_rmse$lambda[i],
                        alpha = pars_rmse$alpha[i])
      
      y_hat_tmp <- predict(fit_tmp, as.matrix(X_test_tmp))
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
  
  fit <- grpreg(X_train, y_train, group=rep(1:9, each = 121), penalty="cMCP", alpha = alpha_opt, lambda = lambda_opt)
  
  mu_hat <- coef(fit)[1]; mu_hat
  beta_hat <- coef(fit)[-1]
  # plot(beta_hat, t='l'); abline(v=cumsum(table(rep(1:10, each = pars$p[k]/10)))); lines(sim$beta, col=2)
  
  y_hat <- predict(fit, X=as.matrix(X_test))
  rmse_p <- RMSE(y_hat, y_test); rmse_p
  
  # y_hat <- mu_hat + as.matrix(X_test) %*% (beta_hat)
  # rmse_p <- RMSE(y_hat, y_test); rmse_p
  
  print(rmse_p)
  save(fit, alpha_opt, alpha, lambda_opt, lambda, mu_hat, beta_hat, y_hat, y_test, X_test, rmse_p, rmse_cv,
       file = paste0("results/cMCP_Ridge/fit_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  rm(fit, alpha_opt, lambda_opt, mu_hat, beta_hat, y_hat, y_test, X_test, rmse_p, rmse_cv)
}




# Elastic-Net  ------------------------------------------------------------
system("mkdir results/EN")

cores <- 3
doParallel::registerDoParallel(cores = cores)

library(glmnet)

alpha <- c(seq(0, 0.3, 0.01), seq(0.4, 1, 0.1))#, seq(0.5, 1, 0.1)) #c(1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
lambda <- seq(0.1, 5, 0.2)


k=1
for(k in 1:nrow(pars)){
  
  print(k)
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X_scale[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X_scale[which(id.cv[[pars$rep[k]]] == fold), ]
  
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
      
      y_hat_tmp <- predict(fit_tmp, as.matrix(X_test_tmp))
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
  
  # toto <- which(rmse_cv$alpha=="0.02")
  # plot(rmse_cv$lambda[toto], rmse_cv$rmse_p[toto])
  # 
  # toto <- which(rmse_cv$lambda=="2.3")
  # plot(rmse_cv$alpha[toto], rmse_cv$rmse_p[toto])
  
  fit <- glmnet(X_train, y_train, alpha = alpha_opt, lambda = lambda_opt)
  
  mu_hat <- coef(fit)[1]; mu_hat
  beta_hat <- coef(fit)[-1]
  # plot(beta_hat, t='l'); abline(v=cumsum(table(rep(1:10, each = pars$p[k]/10)))); lines(sim$beta, col=2)
  
  y_hat <- predict(fit, newx=as.matrix(X_test))
  rmse_p <- RMSE(y_hat, y_test); rmse_p
  
  # y_hat <- mu_hat + as.matrix(X_test) %*% (beta_hat)
  # rmse_p <- RMSE(y_hat, y_test); rmse_p
  
  print(rmse_p)
  save(fit, alpha_opt, alpha, lambda_opt, lambda, mu_hat, beta_hat, y_hat, y_test, X_test, rmse_p, rmse_cv, 
       file = paste0("results/EN/fit_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  rm(fit, alpha_opt, lambda_opt, mu_hat, beta_hat, y_hat, y_test, X_test, rmse_p, rmse_cv)
}





# ANALYSE -----------------------------------------------------------------

rmse_cMCP <- res_cMCP <- NULL

k=1
for(k in 1:nrow(pars)){
  print(k)
  load(paste0("results/cMCP/fit_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata") )
  
  rmse_cMCP<- rbind(rmse_cMCP, data.frame( meth = 'cMCP',
                                           rep  = pars$rep[k],
                                           fold = pars$fold[k],
                                           rmse = rmse_p))
  
  res_cMCP <- rbind(res_cMCP, 
                    data.frame(
                      meth = "cMCP",
                      coef = "none",
                      diff = "none",
                      rep = pars$rep[k], fold = pars$fold[k],
                      x  = 1:ncol(X), Group = rep(id_g, each = 121),
                      t = seq(-180, 180, 3),
                      rmse = rmse_p, 
                      beta = beta_hat,
                      IC_inf = NA, IC_sup = NA, eiq = NA)
  )
  rm(fit, alpha_opt, alpha, lambda_opt, lambda, mu_hat, beta_hat, y_hat, y_test, X_test, rmse_p, rmse_cv)
}



library(grpreg)
rmse_cMCP_Ridge <- res_cMCP_Ridge <- NULL

k=1
for(k in 1:nrow(pars)){
  print(k)
  load(paste0("results_AN/cMCP_Ridge/fit_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata") )

  rmse_cMCP_Ridge<- rbind(rmse_cMCP_Ridge, data.frame( meth = 'cMCP_Ridge',
                                                       rep  = pars$rep[k],
                                                       fold = pars$fold[k],
                                                       rmse = rmse_p))
  
  res_cMCP_Ridge <- rbind(res_cMCP_Ridge, 
                          data.frame(
                            meth = "cMCP_Ridge",
                            coef = "none",
                            diff = "none",
                            rep = pars$rep[k], fold = pars$fold[k],
                            x  = 1:ncol(X),  Group = rep(id_g, each = 121),
                            t = seq(-180, 180, 3),
                            rmse = rmse_p, 
                            beta = beta_hat,
                            IC_inf = NA, IC_sup = NA, eiq = NA)
  )
  rm(fit, alpha_opt, alpha, lambda_opt, lambda, mu_hat, beta_hat, y_hat, y_test, X_test, rmse_p, rmse_cv)
}


# EN ------------------------------------------------------------------
library(grpreg)
rmse_EN <- res_EN <- NULL

k=10
for(k in 1:nrow(pars)){
  print(k)
  load(paste0("results_AN/EN/fit_rep=", pars$rep[k], "_fold_", pars$fold[k], ".Rdata") )
  
  rmse_EN<- rbind(rmse_EN, data.frame( meth = 'ENR',
                                       rep  = pars$rep[k],
                                       fold = pars$fold[k],
                                       rmse = rmse_p))
  
  res_EN <- rbind(res_EN, 
                  data.frame(
                    meth = "ENR",
                    coef = "none",
                    diff = "none",
                    rep = pars$rep[k], fold = pars$fold[k],
                    x  = 1:ncol(X), Group = rep(id_g, each = 121),
                    t = seq(-180, 180, 3),
                    rmse = rmse_p, 
                    beta = beta_hat,
                    IC_inf = NA, IC_sup = NA, eiq = NA)
  )
  rm(fit, alpha_opt, alpha, lambda_opt, lambda, mu_hat, beta_hat, y_hat, y_test, X_test, rmse_p, rmse_cv)
}

# Fusion_HS_2_levels ---------------------------------------------------------------

doParallel::registerDoParallel(cores = 3)

files <- system("ls results/Fusion_HS_2L/" , intern = TRUE)
present <- rep(NA, nrow(pars))
for(k in 1:nrow(pars)){
  present[k] <- any(files == paste0("chain_rep_", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}
which(!present)
sum(present)


k=1
res_Fusion_HS_2_levels <- foreach(k = which(present), .combine = rbind, .verbose = FALSE) %dopar% {
  
  print(k)
  load(paste0("results/Fusion_HS_2L/chain_rep_", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X_scale[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X_scale[which(id.cv[[pars$rep[k]]] == fold), ]
  
  beta_df <- data.frame(beta = colMeans(chain$beta),
                        IC_inf = apply(chain$beta, 2, quantile, 0.025),
                        IC_sup = apply(chain$beta, 2, quantile, 0.975),
                        group = rep(id_g, each = 121))
  beta_df$selected <- 1* (sign(beta_df$IC_inf) * sign(beta_df$IC_sup) > 0)
  group_selection <- aggregate(list(selected = beta_df$selected), by = list(group = beta_df$group), FUN = function(x) 1*(sum(x) > 0))
  group_selection
  beta_df$beta_hat <- beta_df$beta; beta_df$beta_hat[beta_df$group %in% group_selection$group[group_selection$selected == 0]] <- 0
  beta_df$beta_hat2 <- beta_df$beta; beta_df$beta_hat2[beta_df$selected == 0] <- 0
  
  mu_hat <- mean(chain$mu) ; round(mu_hat, 2)
  y_pred <- mu_hat + as.matrix(X_test) %*% beta_df$beta
  rmse <- RMSE(y_pred, y_test); print(rmse1)
  
  
  return( data.frame(
    meth = "group fusion HS 2L",
    coef = "none",
    diff = "HS_2_levels",
    rep = pars$rep[k], fold = pars$fold[k],
    x  = 1:ncol(X), Group = rep(id_g, each = 121),
    t = seq(-180, 180, 3),
    rmse = rmse, 
    beta = colMeans(chain$beta),
    IC_inf = apply(chain$beta, 2, quantile, 0.025),
    IC_sup = apply(chain$beta, 2, quantile, 0.975),
    eiq = apply(chain$beta, 2, quantile, 0.975)-
      apply(chain$beta, 2, quantile, 0.025)
  ))
}







# Fusion_HS_3L ---------------------------------------------------------------
files <- system("ls results/Fusion_HS_3L/" , intern = TRUE)
present <- rep(NA, nrow(pars))
for(k in 1: nrow(pars)){
  present[k] <- any(files == paste0("chain_rep_", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}
which(!present)
sum(present)

k=1
res_Fusion_HS_3_levels <- foreach(k = which(present), .combine = rbind, .verbose = FALSE) %dopar% {
  
  print(k)
  load(paste0("results_AN/Fusion_HS_3L/chain_rep_", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X_scale[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X_scale[which(id.cv[[pars$rep[k]]] == fold), ]
  
  beta_df <- data.frame(beta = colMeans(chain$beta),
                        IC_inf = apply(chain$beta, 2, quantile, 0.025),
                        IC_sup = apply(chain$beta, 2, quantile, 0.975),
                        group = rep(id_g, each = 121))
  beta_df$selected <- 1* (sign(beta_df$IC_inf) * sign(beta_df$IC_sup) > 0)
  group_selection <- aggregate(list(selected = beta_df$selected), by = list(group = beta_df$group), FUN = function(x) 1*(sum(x) > 0))
  group_selection
  beta_df$beta_hat <- beta_df$beta; beta_df$beta_hat[beta_df$group %in% group_selection$group[group_selection$selected == 0]] <- 0
  beta_df$beta_hat2 <- beta_df$beta; beta_df$beta_hat2[beta_df$selected == 0] <- 0
  
  
  mu_hat <- mean(chain$mu) ; round(mu_hat, 2)
  y_pred <- mu_hat + as.matrix(X_test) %*% beta_df$beta
  rmse <- RMSE(y_pred, y_test); print(rmse1)
  
  
  return( data.frame(
    meth = "group fusion HS 3L",
    coef = "none",
    diff = "HS_2_levels",
    rep = pars$rep[k], fold = pars$fold[k],
    x  = 1:ncol(X), Group = rep(id_g, each = 121),
    rmse = rmse, 
    t = seq(-180, 180, 3),
    beta = colMeans(chain$beta), 
    IC_inf = apply(chain$beta, 2, quantile, 0.025),
    IC_sup = apply(chain$beta, 2, quantile, 0.975),
    eiq = apply(chain$beta, 2, quantile, 0.975)-
      apply(chain$beta, 2, quantile, 0.025)))
}




# Fused_HS_2_levels ---------------------------------------------------------------

files <- system("ls results/Fused_HS_2L/" , intern = TRUE)
present <- rep(NA,  nrow(pars))
for(k in 1: nrow(pars)){
  present[k] <- any(files == paste0("chain_rep_", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}
which(!present)
sum(present)


k=1
res_Fused_HS_2_levels <- foreach(k = which(present), .combine = rbind, .verbose = FALSE) %dopar% {
  
  print(k)
  load(paste0("results/Fused_HS_2L/chain_rep_", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X_scale[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X_scale[which(id.cv[[pars$rep[k]]] == fold), ]
  
  beta_df <- data.frame(beta = colMeans(chain$beta),
                        IC_inf = apply(chain$beta, 2, quantile, 0.025),
                        IC_sup = apply(chain$beta, 2, quantile, 0.975),
                        group = rep(id_g, each = 121))
  beta_df$selected <- 1* (sign(beta_df$IC_inf) * sign(beta_df$IC_sup) > 0)
  group_selection <- aggregate(list(selected = beta_df$selected), by = list(group = beta_df$group), FUN = function(x) 1*(sum(x) > 0))
  group_selection
  beta_df$beta_hat <- beta_df$beta; beta_df$beta_hat[beta_df$group %in% group_selection$group[group_selection$selected == 0]] <- 0
  beta_df$beta_hat2 <- beta_df$beta; beta_df$beta_hat2[beta_df$selected == 0] <- 0
  
  # plot(beta_df$beta, t='l'); abline(v=cumsum(rep(121, 9))); abline(0, 0)
  # lines(beta_df$IC_inf, lty = 3)
  # lines(beta_df$IC_sup, lty = 3)
  # lines(beta_df$beta_hat, col = 2)
  # lines(beta_df$beta_hat2, col = 4)
  
  mu_hat <- mean(chain$mu) ; round(mu_hat, 2)
  y_pred <- mu_hat + as.matrix(X_test) %*% beta_df$beta
  rmse <- RMSE(y_pred, y_test); print(rmse1)
  
  return( data.frame(
    meth = "group fused HS 2L",
    coef = "none",
    diff = "HS_2_levels",
    rep = pars$rep[k], fold = pars$fold[k],
    x  = 1:ncol(X), Group = rep(id_g, each = 121),
    t = seq(-180, 180, 3),
    rmse = rmse, 
    beta = colMeans(chain$beta),
    IC_inf = apply(chain$beta, 2, quantile, 0.025),
    IC_sup = apply(chain$beta, 2, quantile, 0.975),
    eiq = apply(chain$beta, 2, quantile, 0.975)-
      apply(chain$beta, 2, quantile, 0.025)
  ))
}






# Fused_HS_3_levels ---------------------------------------------------------------
files <- system("ls results/fused_HS_3L/", intern = TRUE)
present <- rep(NA, nrow(pars))
for(k in 1: nrow(pars)){
  present[k] <- any(files == paste0("chain_rep_", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}
which(!present)
sum(present)

pars[present, ]


k=1
res_Fused_HS_3_levels <- foreach(k = which(present), .combine = rbind, .verbose = FALSE) %dopar% {
  
  print(k)
  load(paste0("results/fused_HS_3L/chain_rep_", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
  
  fold <- pars$fold[k]
  y_train <- Y[which(id.cv[[pars$rep[k]]] != fold)]
  X_train <- X_scale[which(id.cv[[pars$rep[k]]] != fold), ]
  y_test <- Y[which(id.cv[[pars$rep[k]]] == fold)]
  X_test <- X_scale[which(id.cv[[pars$rep[k]]] == fold), ]
  
  beta_df <- data.frame(beta = colMeans(chain$beta),
                        IC_inf = apply(chain$beta, 2, quantile, 0.025),
                        IC_sup = apply(chain$beta, 2, quantile, 0.975),
                        group = rep(id_g, each = 121))
  beta_df$selected <- 1* (sign(beta_df$IC_inf) * sign(beta_df$IC_sup) > 0)
  group_selection <- aggregate(list(selected = beta_df$selected), by = list(group = beta_df$group), FUN = function(x) 1*(sum(x) > 0))
  group_selection
  beta_df$beta_hat <- beta_df$beta; beta_df$beta_hat[beta_df$group %in% group_selection$group[group_selection$selected == 0]] <- 0
  beta_df$beta_hat2 <- beta_df$beta; beta_df$beta_hat2[beta_df$selected == 0] <- 0

  mu_hat <- mean(chain$mu) ; round(mu_hat, 2)
  y_pred <- mu_hat + as.matrix(X_test) %*% beta_df$beta
  rmse <- RMSE(y_pred, y_test); print(rmse1)
  
  return( data.frame(
    meth =  "group fused HS 3L",
    coef = "none",
    diff = "HS_2_levels",
    rep = pars$rep[k], fold = pars$fold[k],
    x  = 1:ncol(X), Group = rep(id_g, each = 121),
    t = seq(-180, 180, 3),
    rmse = rmse,
    beta = colMeans(chain$beta),
    IC_inf = apply(chain$beta, 2, quantile, 0.025),
    IC_sup = apply(chain$beta, 2, quantile, 0.975),
    eiq = apply(chain$beta, 2, quantile, 0.975)-
      apply(chain$beta, 2, quantile, 0.025)))
}




res_beta <- rbind(res_cMCP, res_cMCP_Ridge, res_EN, 
                  res_SPLS,
                  res_Fusion_HS_3_levels,
                  res_Fused_HS_3_levels, 
                  res_Fusion_HS_2_levels,
                  res_Fused_HS_2_levels
)

save(res_beta, res_cMCP, res_cMCP_Ridge, res_EN, res_SPLS, 
     res_Fusion_HS_3_levels, res_Fused_HS_3_levels, 
     res_Fusion_HS_2_levels, res_Fused_HS_2_levels, 
     file = "results_AN.Rdata")




# Load --------------------------------------------------------------------
load("results_AN.Rdata")


# ggplot ------------------------------------------------------------------
res_beta2 <- aggregate(list(beta=res_beta[, "beta"], 
                            IC_inf = res_beta[, "IC_inf"],
                            IC_sup = res_beta[, "IC_sup"]),
                       by = list(meth = res_beta$meth, 
                                 x = res_beta$x, t=res_beta$t, 
                                 Group=res_beta$Group), 
                       FUN = mean)
head(res_beta2)

res_beta_tmp <- res_beta 
IC_inf <- aggregate(list(IC_inf = res_beta_tmp[, "beta"]),
                    by = list(meth = res_beta_tmp$meth, 
                              x = res_beta_tmp$x, t=res_beta_tmp$t, 
                              Group=res_beta_tmp$Group), 
                    FUN = quantile, prob=0.025)
IC_sup <- aggregate(list(IC_sup = res_beta_tmp[, "beta"]),
                    by = list(meth = res_beta_tmp$meth, 
                              x = res_beta_tmp$x, t=res_beta_tmp$t, 
                              Group=res_beta_tmp$Group), 
                    FUN = quantile, prob=0.975)

res_beta2[, "IC_inf"] <- IC_inf$IC_inf
res_beta2[, "IC_sup"] <- IC_sup$IC_sup


tmp_order <- c(
  "group fused HS 3L",
  "group fused HS 2L",
  "group fusion HS 3L", 
  "group fusion HS 2L", 
  "cMCP", "cMCP_Ridge",
  "ENR", "SPLSR"
)

res_beta2$meth <- factor(res_beta2$meth, levels = tmp_order)
res_beta$meth <- factor(res_beta$meth, levels = tmp_order)
unique(res_beta$meth)

borne_sup <- 1.5
borne_inf <- -1.5
for(i in 1:nrow(res_beta2)){
  res_beta2$beta[i] <- min(max(res_beta2$beta[i], borne_inf), borne_sup)
  res_beta2$IC_inf[i] <- max(res_beta2$IC_inf[i], borne_inf)
  res_beta2$IC_sup[i] <- min(res_beta2$IC_sup[i], borne_sup)
}

myplot <- ggplot(data = res_beta2, 
                 aes(x = t, y=beta, group = Group, color = Group)) +
  geom_line(size = 0.5) +  ylim(borne_inf, borne_sup) +
  facet_wrap(~meth, ncol = 2) +
  geom_line(aes(x= t, y = IC_inf), linetype = "dotted", alpha = 0.8) + 
  geom_line(aes(x= t, y = IC_sup), linetype = "dotted", alpha = 0.8) +
  scale_x_continuous(breaks = seq(-180, 180, 60), minor_breaks = seq(-180, 180, 30))  # + theme_bw() +

myplot
ggsave("Figures/estimation_beta.pdf", plot = myplot, scale = 1.5, width = 11, height = 12, units = "cm")
ggsave("Figures/estimation_beta.png", plot = myplot, device = "png", scale = 1.5, width = 11, height = 12, units = "cm")







for(i in 1:nrow(res_beta2)){
  res_beta2$IC_inf[i] <- max(res_beta2$IC_inf[i], -1.2)
  res_beta2$IC_sup[i] <- min(res_beta2$IC_sup[i], 1.1)
}

gridExtra::grid.arrange(
  ggplot(data = res_beta2 %>% filter(meth == "group fused HS 3L", 
                                     Group %in% c("Tmin")), 
         aes(x =t, y=beta)) +
    geom_ribbon(aes(ymin=IC_inf, ymax=IC_sup), alpha=0.4) +
    geom_line(size = 0.7, color="green4") +  ylim(-1.2, 1.1) +
    theme(legend.position="none") +
    facet_wrap(~Group) +
    scale_x_continuous(breaks = seq(-180, 180, 60), minor_breaks = seq(-180, 180, 30)),  # + theme_bw() +
  #
  ggplot(data = res_beta2 %>% filter(meth == "group fused HS 3L", 
                                     Group %in% c("DRD")), 
         aes(x =t, y=beta)) +
    geom_ribbon(aes(ymin=IC_inf, ymax=IC_sup), alpha=0.4) +
    theme(legend.position="none") +
    geom_line(size = 0.7, color="green4") +  ylim(-1.2, 1.1) +
    facet_wrap(~Group) +
    scale_x_continuous(breaks = seq(-180, 180, 60), minor_breaks = seq(-180, 180, 30)),  # + theme_bw() +
  #
  ggplot(data = res_beta2 %>% filter(meth == "group fused HS 3L", 
                                     Group %in% c("SR")), 
         aes(x =t, y=beta)) +
    geom_line(size = 0.7, color="green4") +  ylim(-1.2, 1.1) +
    geom_ribbon(aes(ymin=IC_inf, ymax=IC_sup), alpha=0.4) +
    theme(legend.position="none") +
    facet_wrap(~Group) +
    scale_x_continuous(breaks = seq(-180, 180, 60), minor_breaks = seq(-180, 180, 30)),  # + theme_bw() +
  #
  ggplot(data = res_beta2 %>% filter(meth == "group fused HS 3L", 
                                     Group %in% c("SD")), 
         aes(x =t, y=beta)) +
    geom_line(size = 0.7, color="red") +  ylim(-1.2, 1.1) +
    geom_ribbon(aes(ymin=IC_inf, ymax=IC_sup), alpha=0.4) +
    theme(legend.position="none") +
    facet_wrap(~Group) +
    ylab("") +
    # theme(axis.title.x = element_blank()) +
    scale_x_continuous(breaks = seq(-180, 180, 60), minor_breaks = seq(-180, 180, 30)),  # + theme_bw() +
  nrow=1
)


ggplot(data = res_beta2 %>% filter(meth == "group fused HS 3L", 
                                   Group %in% c("Tmin", "DRD", "SR", "SD")), 
       aes(x=t, y=beta)) +
  geom_line(size = 0.7, aes(x=t, y=beta, colour=Group)) + #, color=c("green4", "green4", "green4", "red")) + 
  ylim(-1.2, 1.1) +
  geom_ribbon(aes(ymin=IC_inf, ymax=IC_sup), alpha=0.35) +
  theme(legend.position="none") +
  facet_wrap(~Group, nrow=1) +
  ylab("") +
  scale_color_manual(breaks = c("Tmin", "DRD", "SR", "SD"),
                     values=c("red", "green4", "green4", "green4"))+
  scale_x_continuous(breaks = seq(-180, 180, 60), minor_breaks = seq(-180, 180, 30))
ggsave("Figures/estimation_beta_2.pdf", scale = 1.5, width = 16, height = 7, units = "cm")




# Table -------------------------------------------------------------------

rmse_df <- aggregate(list(rmse_p = res_beta[, c("rmse")]), 
                     by = list(meth = res_beta$meth), 
                     FUN = mean, na.rm = TRUE)


rmse_df$rmse_p <- round(rmse_df$rmse_p, 2)
rmse_df <- rmse_df[order(rmse_df$rmse_p), ]
rmse_df


library(xtable)
print(xtable((rmse_df[, ])), include.rownames = FALSE)



tmp <- res_beta
tmp$selected <- 1* (sign(tmp$IC_inf) * sign(tmp$IC_sup) > 0)
group_selection <- aggregate(
  list(selected = tmp$selected), 
  by = list(meth = tmp$meth, Group = tmp$Group, rep = tmp$rep, fold=tmp$fold), 
  FUN = function(x) 1*(sum(x) > 0))

group_selection <- aggregate(
  list(selected = group_selection$selected), 
  by = list(meth = group_selection$meth, Group = group_selection$Group), 
  FUN = mean)
group_selection$selected <- round(group_selection$selected , 2)
View(group_selection)


tmp <- res_beta2
tmp$selected <- 1* (sign(tmp$IC_inf) * sign(tmp$IC_sup) > 0)
group_selection2 <- aggregate(
  list(selected = tmp$selected), 
  by = list(meth = tmp$meth, Group = tmp$Group), 
  FUN = function(x) 1*(sum(x) > 0))
group_selection2$selected <- round(group_selection2$selected , 2)
View(group_selection2)

treshold <- 0.1
group_selection %>% filter(meth == "PLSR", selected > treshold)
group_selection %>% filter(meth == "SPLSR", selected > treshold)
group_selection %>% filter(meth == "GRPREG", selected > treshold)
group_selection %>% filter(meth == "Fusion_HS_2_levels_ext", selected > treshold)
group_selection %>% filter(meth == "Fusion_HS_2_levels", selected > treshold)
group_selection %>% filter(meth == "Fusion_HS_3_levels", selected > treshold)
group_selection %>% filter(meth == "Fused_HS_2_levels_ext", selected > treshold)
group_selection %>% filter(meth == "Fused_HS_2_levels", selected > treshold)
group_selection %>% filter(meth == "Fused_HS_3_levels", selected > treshold)


toto <- rbind(t(group_selection %>% filter(meth == "Fused_HS_3_levels") %>% select(selected)),
              t(group_selection %>% filter(meth == "Fused_HS_2_levels") %>% select(selected)),
              t(group_selection %>% filter(meth == "Fused_HS_2_levels_ext") %>% select(selected)),
              t(group_selection %>% filter(meth == "Fusion_HS_3_levels") %>% select(selected)),
              t(group_selection %>% filter(meth == "Fusion_HS_2_levels") %>% select(selected)),
              t(group_selection %>% filter(meth == "Fusion_HS_2_levels_ext") %>% select(selected)))

rownames(toto) <- c("Fused 3L", "Fused 2L", "Fused 2L*", "Fusion 3L", "Fusion 2L", "Fusion 2L*")
colnames(toto) <- t(group_selection %>% filter(meth == "Fused_HS_3_levels"))[2, ]
toto

xtable(toto)

group_selection <- data.frame(aggregate(list(selected = res_beta2$beta), by = list(meth = res_beta2$meth, group = res_beta2$Group),FUN = function(x) sum(abs(x))))
group_selection$selected <- round(group_selection$selected, 2)

group_selection %>% filter(meth == "PLSR")
group_selection %>% filter(meth == "SPLSR")
group_selection %>% filter(meth == "GRPREG")
group_selection %>% filter(meth == "Fusion_HS_2_levels_ext")
group_selection %>% filter(meth == "Fusion_HS_2_levels")
group_selection %>% filter(meth == "Fusion_HS_3_levels")
group_selection %>% filter(meth == "Fused_HS_2_levels_ext")
group_selection %>% filter(meth == "Fused_HS_2_levels")
group_selection %>% filter(meth == "Fused_HS_3_levels")

unique(group_selection$group)






# Convergence -------------------------------------------------------------



# Fused_HS_3_levels ---------------------------------------------------------------
files <- system("ls results/Fused_HS_3L/" , intern = TRUE)
present <- rep(NA, nrow(pars))
for(k in 1: nrow(pars)){
  present[k] <- any(files == paste0("chain_rep_", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}
which(!present)
sum(present)

present2 <- which(present)

# HS
mcmc_list_chain_hs <- list()
k=1
for(k in 1:length(present2)){
  print(k)
  load(paste0("results/Fused_HS_3L/chain_rep_", pars$rep[present2[k]], "_fold_", pars$fold[present2[k]], ".Rdata"))
  
  colnames(chain$beta) <- paste0("b", 1:ncol(chain$beta))
  tmp <- do.call(cbind, chain[c("beta", "mu", "se2")])
  mcmc_list_chain_hs[[k]] <- coda::mcmc(tmp)#, thin = thinin, start = burnin+1, end = niter)
  rm(chain)
}
mcmc_list_hs <- coda::mcmc.list(mcmc_list_chain_hs); rm(mcmc_list_chain_hs)
gelman.diag_hs <- coda::gelman.diag(mcmc_list_hs)
gelman.diag_hs
plot(gelman.diag_hs$psrf[paste0("b", 1:1089), 1], ylab = "", ylim=c(1, 1.2), main = "fHS"); abline(1.1, 0, lty=2, col = 2, lwd=2)
gelman.diag_hs[[1]][c("mu", "se2"), ]
summary(gelman.diag_hs[[1]][, 1])






# Fusion_HS_3_levels ---------------------------------------------------------------
files <- system("ls results/Fusion_HS_3L/" , intern = TRUE)
present <- rep(NA, 100)
for(k in 1:100){
  present[k] <- any(files == paste0("chain_rep_", pars$rep[k], "_fold_", pars$fold[k], ".Rdata"))
}
which(!present)
sum(present)
present2 <- which(present)

# HS
mcmc_list_chain_hs <- list()
k=1
for(k in 1:length(present2)){
  print(k)
  load(paste0("results/Fusion_HS_3L/chain_rep_", pars$rep[present2[k]], "_fold_", pars$fold[present2[k]], ".Rdata"))
  
  colnames(chain$beta) <- paste0("b", 1:ncol(chain$beta))
  tmp <- do.call(cbind, chain[c("beta", "mu", "se2")])
  mcmc_list_chain_hs[[k]] <- coda::mcmc(tmp)#, thin = thinin, start = burnin+1, end = niter)
  rm(chain)
}
mcmc_list_hs <- coda::mcmc.list(mcmc_list_chain_hs); rm(mcmc_list_chain_hs)
gelman.diag_hs <- coda::gelman.diag(mcmc_list_hs)
gelman.diag_hs
plot(gelman.diag_hs$psrf[paste0("b", 1:1089), 1], ylab = "", ylim=c(1, 1.2), main = "fHS"); abline(1.1, 0, lty=2, col = 2, lwd=2)
gelman.diag_hs[[1]][c("mu", "se2"), ]
summary(gelman.diag_hs[[1]][, 1])
summary(gelman.diag_hs$psrf[paste0("b", 1:1089), 1])





