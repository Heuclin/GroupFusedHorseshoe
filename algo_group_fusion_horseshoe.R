require(Matrix)
require(mvnfast)
require(invgamma)
require(truncnorm)
require(doParallel)
require(foreach)
require(coda)




#' MCMC sampler (for one chain) for group fusion / fused Horseshoe prior selection
#'
#'
#' @param y: vector ofthe response variable
#' @param X: the matrix of the indexing variable, if many groups are present, X is the collection of all matrix associated to each group
#' @param Z: optional design matrix for a random effect
#' @param A: optional covariance matrix for a random effect
#' @param group:  vector of length ncol(X) indicating the number group of each variable in X
#' @param d: the degree of the difference penalty
#' @param D: the matrix of the finite difference operator
#' @param model: c('gaussian', 'probit'), indicating is the responce is gaussian or probit regression
#' @param settings: a list containing the MCMC settings:
#' \itemize{
#'    \item niter: number of iterations of the MCMC chain
#'    \item burnin: number of initial iterations of the MCMC chain which should be discarded
#'    \item thin: save only every thin-th iteration
#' }
#' @param diff3levels: boolean indicating if three levels (global, group specific and local) of variance parameters should be used on coefficient differences instead of two (global and local), default is TRUE
#' @param var_sel: boolean indicating if selection of variable is needed (fused) or not (fusion), default choice is TRUE
#'
#' @return return a list of the MCMC sample of each parameter
#'
#' @author Benjamin Heuclin \email{benjamin.heuclin@@gmail.com}, Frédéric Mortier, Catherine Trottier Marie Denis
#' @seealso \code{\link{sparse_group_fusion_horseshoe}, \link{fusion_horseshoe_MCMC}}
#' @references \url{}
#' @export
#'
#' @example R/exemple_2.R
group_fused_HS_MCMC <- function(y, X, Z=NULL, A=NULL, group,
                                d=1, D, model = 'gaussian',
                                settings, diff3levels=TRUE, var_sel = TRUE
){
  epsilon = 0
  
  if(model != 'gaussian' & model != 'probit') stop("model must be 'gaussian' or 'probit'")
  if(model == 'gaussian') print("Sparse Group Fused Horseshoe prior for gaussian responses")
  if(model == 'probit') print("Sparse Group Fused Horseshoe prior for binary responses")
  if(!is.null(A) & is.null(Z)) stop("you may specify Z")
  if(!is.null(Z) & is.null(A)) stop("you may specify A")
  Y <- y
  n_iter        <- settings$niter
  burnin        <- settings$burnin
  thinin        <- settings$thin
  n             <- length(Y)
  G             <- max(group)
  group_length  <- table(group)
  p             <- ncol(X)
  idx_g         <- list()
  for(g in 1:G) idx_g[[g]] <- c(0, cumsum((group_length-d)))[g] + 1:(group_length[g]-d)
  if(G==1) diff3levels <- FALSE
  
  
  chain         <- list()
  chain$mu      <- rep(   NA, floor(n_iter - burnin)/thinin)
  chain$beta    <- matrix(NA, floor(n_iter - burnin)/thinin, p)
  chain$upsilon   <- matrix(NA, floor(n_iter - burnin)/thinin, p)
  chain$omega   <- matrix(NA, floor(n_iter - burnin)/thinin, sum(group_length - d))
  chain$lambda2 <- matrix(NA, floor(n_iter - burnin)/thinin, G)
  chain$tau2 <- rep(NA, floor(n_iter - burnin)/thinin)
  if(!is.null(A)) chain$U   <- matrix(NA, floor(n_iter - burnin)/thinin, ncol(Z))
  if(!is.null(A)) chain$su2 <- rep(   NA, floor(n_iter - burnin)/thinin)
  if(model == 'probit') {chain$Y <- matrix(NA, floor(n_iter - burnin)/thinin, n)
  colnames(chain$Y) <- paste0('Y', 1:ncol(chain$Y))}
  chain$se2     <- rep(   NA, floor(n_iter - burnin)/thinin)
  
  
  mu              <- mean(Y)
  beta            <-  c(mvnfast::rmvn(p, 0, 1)) # c(rep(1, 3), rep(0, p-3))
  omega   <- phi  <- rep(1, sum(group_length - d))
  upsilon <- eta  <- rep(0, sum(group_length))  # c(1e5, 1e5, 1e5, rep(1, p-3)) #
  lambda2 <- psi  <- rep(1, G)
  tau2            <- 1e-3 # abs(c(rmvn(1, 0, 0.00001)))  #
  ksi             <- 1
  if(!is.null(A)){
    Ainv          <- solve(A)
    su2 <- psi    <- 1
    U             <- rep(0, ncol(Z))
    ZU            <- Z %*% U
  }else{
    ZU            <- rep(0, n)
  }
  se2 <- a        <- abs(c(mvnfast::rmvn(1, 0, 1))) #
  
  
  BtSB            <- rep(NA, G)
  BtSB_a          <- rep(NA, G)
  BtSB_b          <- rep(NA, G)
  iter<- ii       <- 1
  q               <- ncol(Z)
  
  Upsilon_g_list <- DOD_list <- list()
  
  # MCMC
  print("0%")
  while(iter <= n_iter){
    # if(iter %% 100 == 0) print(iter)
    if(iter %% (n_iter/10) == 0) print(paste((iter %/% (n_iter/10))*10, "%"))
    
    # print(iter)
    # if(iter == 292) browser()
    
    # Update Y for probit regression
    if(model == 'probit'){
      Xb = mu + X %*% beta + ZU
      for(i in 1:n){
        if(y[i]==1) Y[i] <- truncnorm::rtruncnorm(1, mean = Xb[i], sd = 1, a = 0)
        else Y[i] <- truncnorm::rtruncnorm(1, mean = Xb[i], sd = 1, b = 0)
      }
    }
    
    # mu
    mu <- rnorm(1, 1/n * sum(Y - X %*% beta - ZU), se2/n)
    
    
    for(g in 1:G){
      # print(g)
      # selection within group
      if (!var_sel){
        Upsilon_g <- 0*diag(group_length[g])
        diag(Upsilon_g)[1:d] <- 1
        Upsilon_g_list[[g]] <- Upsilon_g
      }else{
        for(i in 1 : group_length[g]){
          upsilon[which(group == g)[i]] <- rinvgamma(n = 1, shape = 1, rate = 1/eta[which(group == g)[i]] + beta[which(group == g)[i]]^2/(2*se2))+1e-5
          eta[which(group == g)[i]] <- rinvgamma(n = 1, shape = 1, rate = 1 + 1/upsilon[which(group == g)[i]])
        }
        Upsilon_g_list[[g]] <- Upsilon_g <- diag(1/upsilon[which(group == g)])
      }
      
      # fusion within group
      for(i in 1 : (group_length[g] - d)){
        omega[idx_g[[g]][i]] <- rinvgamma(n= 1, shape = 1, rate = 1/ phi[idx_g[[g]][i]] + (D[[g]][i, ] %*% beta[which(group == g)])^2 / (se2*tau2 * lambda2[g]))+1e-5
        phi[idx_g[[g]][i]] <- rinvgamma(n = 1, shape = 1, rate = 1+1/omega[idx_g[[g]][i]])
      }
      Omega_g <- diag(1/omega[idx_g[[g]]])
      
      
      DOD_list[[g]] <- DOD <- t(D[[g]]) %*% Omega_g %*% D[[g]]
      
      # beta_g
      prec <- crossprod(X[, which(group == g)])/se2   +
        1/(se2) * (Upsilon_g + (1/tau2) * (1/lambda2[g]) * DOD ) +
        epsilon*diag(group_length[g])
      tmp_bg <- 1/se2 * t(X[, which(group == g)]) %*% (Y - mu -  X[, which(group != g)] %*% beta[which(group != g)] - ZU)
      
      s <- svd(prec)
      id.svd <- 1:group_length[g]
      Dinv <- diag(1/s$d[id.svd])
      beta[which(group == g)] <- crossprod(t(s$u[, id.svd] %*% diag(1/sqrt(s$d[id.svd]))),
                                           c(rmvn(1, diag(1/sqrt(s$d[id.svd])) %*% t(s$u[, id.svd]) %*% tmp_bg, diag(length(id.svd)), isChol = TRUE)))
      
      
      if(diff3levels){
        lambda2[g] <- rinvgamma(n = 1, shape = (group_length[g]+1)/2, rate = 1/ psi[g] +
                                  t(beta[which(group == g)]) %*% (DOD + epsilon*diag(group_length[g])) %*% beta[which(group == g)] / (2*se2*tau2))
        if(lambda2[g] < 1e-5) lambda2[g] <- 1e-5
        psi[g] <- rinvgamma(n = 1, shape = 1, rate = 1+1/lambda2[g])
      }else{
        lambda2[g] <- 1
      }
      
      BtSB[g]   <- t(beta[which(group == g)])     %*%     (Upsilon_g + (1/tau2) * (1/lambda2[g]) * DOD)     %*%     beta[which(group == g)]
      BtSB_a[g] <- t(beta[which(group == g)])     %*%     Upsilon_g     %*%     beta[which(group == g)]
      BtSB_b[g] <- t(beta[which(group == g)])     %*%     ( 1/lambda2[g] * (DOD + epsilon*diag(group_length[g])))     %*%     beta[which(group == g)]
      
    }
    
    # tau2 (global variance paramter on coefficient differences)
    rate_tau2 = 1/ksi + sum(BtSB_b ) / (2 * se2)
    tau2 <- rinvgamma(n = 1, shape = (p+1)/2, rate = rate_tau2)
    if(tau2 < 1e-5) tau2 <- 1e-5
    ksi <-  rinvgamma(n = 1, shape = 1, rate = 1+1/tau2)
    
    
    for(g in 1:G){
      BtSB[g]   <- t(beta[which(group == g)])     %*%     (Upsilon_g_list[[g]] + (1/tau2) * (1/lambda2[g]) * DOD_list[[g]])     %*%     beta[which(group == g)]
    }
    
    
    # random effect
    if(!is.null(A)){
      Sigma_u <- solve(Ainv / su2 + crossprod(Z)/se2)
      U <- as.vector(rmvn(1, Sigma_u %*% crossprod(Z, Y - mu - X %*% beta)/se2, Sigma_u))
      su2 <-  rinvgamma(n = 1, shape = 1/2 + q/2, rate = 1/psi + crossprod(U, Ainv %*% U)/2)
      psi <-  rinvgamma(n = 1, shape = 1, rate = 1+1/su2)
      ZU <- Z %*% U
    }
    
    rate_se2 <- 1/a + sum(BtSB/(2)) + crossprod(Y - mu - X %*% beta - ZU)/2
    se2 <- rinvgamma(n= 1, shape = (1+p+n)/2, rate = rate_se2)
    a <- rinvgamma(n = 1, shape = 1, rate = 1+1/se2)
    
    #__________________________________________
    if(iter > burnin & iter %% thinin == 0){
      chain$mu[ii] <- mu
      chain$beta[ii, ] <- beta
      chain$upsilon[ii, ] <- upsilon
      chain$omega[ii, ] <- omega
      chain$lambda2[ii, ] <- lambda2
      chain$tau2[ii] <- tau2
      if(!is.null(A)) chain$U[ii, ] <- U
      if(!is.null(A)) chain$su2[ii] <- su2
      if(model == 'probit') chain$Y[ii, ] <- Y
      chain$se2[ii] <- se2
      ii <- ii + 1
    }
    iter <- iter + 1
  }
  prob_Y_theta <- matrix(NA, length(chain$se2), n); colnames(prob_Y_theta) <- paste0("prob_Y_theta", 1:n)
  if(is.null(A)){
    for(nit in 1:length(chain$se2)){
      mu <- chain$mu[nit]
      beta = chain$beta[nit, ]
      se2 = chain$se2[nit]
      # y.hat <- x %*% beta + Z %*% kronecker(diag(nb_niv), diag(sdu)) %*% chain_h$u[nit, ]
      prob_Y_theta[nit, ] <- unlist(lapply(1:n, function(i) mvnfast::dmvn(Y[i], mu +X[i, ] %*% beta, se2)))
    }}else{
      for(nit in 1:length(chain$se2)){
        mu <- chain$mu[nit]
        beta = chain$beta[nit, ]
        U = chain$U[nit, ]
        se2 = chain$se2[nit]
        # y.hat <- x %*% beta + Z %*% kronecker(diag(nb_niv), diag(sdu)) %*% chain_h$u[nit, ]
        prob_Y_theta[nit, ] <- unlist(lapply(1:n, function(i) mvnfast::dmvn(Y[i], mu +X[i, ] %*% beta + Z[i, ] %*% U, se2)))
      }
    }
  chain$loglikelihood <- rowSums(log(prob_Y_theta))
  chain$prob_Y_theta <- prob_Y_theta
  return(chain)
}









#' (BETA VERSION) Main function to run the sparse group selection for indexing variables within group using group fusion / fused horseshoe prior.
#'
#'
#' \code{group_fused_HS} compute the bayesian hierarchical model described in the paper ... to ...
#' It allows to apply repetion and cross validation to calculate RMSE (gaussian response) or F1 score (binary response).
#'
#' @param y vector of the response variable
#' @param X the matrix of the indexing variable, if many groups are present, X is the collection of all matrix associated to each group
#' @param selection Boolean indicating if selection of variable is needed (fused) or not (fusion), default choice is TRUE
#' @param degree number indicating the degre of the difference penalty, must be >= 1
#' @param nb_group number of group, default choice is one
#' @param length_group optional, vector of the length of each group, default choice is NULL indicating that all groups have the same length (ncol(X)/nb_group)
#' @param model c('gaussian', 'probit'), indicating is the responce is gaussian or probit regression
#' @param niter number of iteration of the MCMC chain, default is 10000
#' @param burnin number of initial iterations of the MCMC chain which should be discarded, defaults to 5000
#' @param thin save only every thin-th iteration, default is 10
#' @param CV numeric, indicating the number of folds for cross-validation, defaults to NULL indicating no cross-validation
#' @param id.cv optional, list of length 'rep' of vector of length 'n'. Each vector indicates the group indicator of the individuals for the cross-validation, defaults is NULL indicating the function affects ramdomly a fold to each individual for each repetition
#' @param rep number of repetition, default is one
#' @param cores number of cores for parallelisation, default is 1
#' @param gelman.plot boolean indicating if the gelman plot should be plotted in the case where K is upper than one, default is FALSE
#' @param traceplot boolean indicating if the traceplot should be plotted in the case where K is upper than one, default is FALSE
#' @param Z optional design matrix for a random effect
#' @param A opational covariance matrix for a random effect
#' @param save boolean indicating if the MCMC chains should me saved, default is true
#' @param path optional (existing) path to save the MCMC chains if save = TRUE, if NULL, path = Fused_HS if selection = TRUE, Fusion_HS ortherwise
#' @param ... additional arguments for the \link{fused_horseshoe_MCMC} function
#' @return return a list containing:
#' \itemize{
#'   \item settings: the list of settings of the MCMC chains,
#'   \item list_chain: the list of the MCMC chains generated by \code{\link{fused_horseshoe_MCMC}}
#'   \item waic_cv_folds: vector of the waic of each fold if CV != NULL
#'   \item waic: the mean of the waic over the folds
#'   \item rmse_cv_folds: the vector of the RMSE of each fold if CV != NULL and model = 'gaussian'
#'   \item rmse_cv: the mean of the rmse_cv_folds
#'   \item F1_score_cv_folds: the vector of the F1_score of each fold if CV = != NULL and model = 'probit'
#'   \item F1_score_cv: the mean of the F1_score_cv_folds
#'   \item mcmc_list: mcmc formating list of list_chain to achieve the elman-rubin diagnostic if K is upper than one
#'   \item gelman.diag: the gelman-rubin diagnostic if K is upper than one
#'   \item estimations: the estimations (mean over the chains of the mean within each chain) of the parameters
#' }
#'
#'
#' @author Benjamin Heuclin \email{benjamin.heuclin@@gmail.com}, Frédéric Mortier, Catherine Trottier Marie Denis
#' @seealso \code{\link[coda]{gelman.diag}, \link[coda]{gelman.plot}, \link[coda]{plot.mcmc}, \link[MLmetrics]{F1_Score}, \link{fused_horseshoe_MCMC}}
#' @keywords Horseshoe prior, fusion penalty, fused penalty, MCMC
#' @references \url{}
#' @export
#'
#' @example R/exemple_1.R
#'
group_fused_HS <- function(y, X, selection = TRUE, degree = 1, nb_group = 1, length_group = NULL,
                           model = 'gaussian', niter = 10000, burnin = 5000, thin = 10,
                           CV = NULL, id.cv = NULL, rep = 1, cores = 1, gelman.plot=FALSE, traceplot=FALSE,
                           Z = NULL, A = NULL, save = TRUE, path = NULL, ...)
{
  if(cores == -1) cores <- detectCores()-1
  print(paste("Nb cores = ", cores))
  if(is.null(path)){
    if(selection) path <- "Fused_HS/" else path <- "Fusion_HS/"
    system(paste0("mkdir ", path))
  }
  
  output          <- list()
  settings        <- list()
  settings$niter  <- niter
  settings$burnin <- burnin
  settings$thin   <- thin
  
  p <- ncol(X)
  if(p%%nb_group !=0 & is.null(length_group)) warning("The number of variables is not a multiple of the number of groups.")
  if(is.null(length_group)) length_group <- rep(p/nb_group, nb_group) else nb_group <- length(length_group)
  group <- settings$group  <- rep(1:nb_group, times = length_group)
  d <- settings$d      <- degree
  D <- settings$D      <- lapply(table(settings$group), function(p) diff(diag(p), differences = settings$d ))
  settings$model  <- model
  settings$CV <- CV
  if(!is.null(CV)){
    if(is.null(id.cv)) {
      id.cv <- list()
      for(k in 1:rep) id.cv[[k]] <- sample(1:CV, length(y), replace = TRUE)
    }
    settings$id.cv <- id.cv
  }
  output$settings <- settings
  
  # MCMC _____________________________________________________________________________
  if(is.null(CV)) CV <- 1
  pars <- expand.grid(rep = 1:rep, fold = 1:CV)
  doParallel::registerDoParallel(cores = min(cores, nrow(pars)))
  
  list_chain <- foreach::foreach(k = 1:nrow(pars), .verbose = FALSE) %dopar% {
    fold <- pars$fold[k]
    if(CV){
      y_train <- y[which(id.cv[[pars$rep[k]]] != fold)]
      X_train <- X[which(id.cv[[pars$rep[k]]] != fold), ]
      if(!is.null(Z)) Z_train <- Z[which(id.cv[[pars$rep[k]]] != fold), ]  else Z_train <- NULL
    }else{
      y_train <- y
      X_train <- X
      if(!is.null(Z)) Z_train <- Z else Z_train <- NULL
    }
    
    chain <- group_fused_HS_MCMC(y=y_train, X=X_train, Z=Z_train,
                                 group=group,
                                 A=A, d=d, D=D, model = model, settings=settings,
                                 var_sel = selection, ...)
    
    lppd <- sum(log(apply(chain$prob_Y_theta, 2, mean)))
    Pwaic <- sum(apply(log(chain$prob_Y_theta), 2, var))
    waic <- -2 * lppd + 2 * Pwaic
    chain$lppd <- lppd
    chain$Pwaic <- Pwaic
    chain$waic <- waic
    save(chain, settings, file = paste0(path, "/chain_rep_", pars$rep[k], "_fold_", pars$fold[k], ".Rdata") )
    if(save) return() else return(chain)
  }
  
  if(!save) {output$list_chain <- list_chain ; rm(list_chain)}
  
  
  
  # Gelman diag _____________________________________________________________________________
  output$gelman.diag <- NULL
  if(nrow(pars)>1){
    mcmc_list_chain <- list()
    for(k in 1:nrow(pars)){
      if(save) load(paste0(path, "/chain_rep_", pars$rep[k], "_fold_", pars$fold[k], ".Rdata")) else chain <- output$list_chain[[k]]
      if(is.null(A)){
        if(model!='probit' & !CV) tmp <- do.call(cbind, chain[c("mu", "beta", "se2")])
        if(model!='probit' & CV) tmp <- do.call(cbind, chain[c("mu", "beta", "se2")])
        if(model=='probit' & !CV) tmp <- do.call(cbind, chain[c("mu", "beta")]) 
        if(model=='probit' & CV) tmp <- do.call(cbind, chain[c("mu", "beta")]) 
      }else{
        if(model!='probit' & !CV) tmp <- do.call(cbind, chain[c("mu", "beta", "U", "su2", "se2")])
        if(model!='probit' & CV) tmp <- do.call(cbind, chain[c("mu", "beta", "U", "su2", "se2")])
        if(model=='probit' & !CV) tmp <- do.call(cbind, chain[c("mu", "beta", "U", "su2")]) # "Y",
        if(model=='probit' & !CV) tmp <- do.call(cbind, chain[c("mu", "beta", "U", "su2")]) # "Y",
      }
      mcmc_list_chain[[k]] <- coda::mcmc(tmp)#, thin = thinin, start = burnin+1, end = niter)
    }
    output$mcmc_list <- coda::mcmc.list(mcmc_list_chain); rm(mcmc_list_chain)
    output$gelman.diag <- coda::gelman.diag(output$mcmc_list)
    print("Summary of the Gelman diagnostic:")
    print(summary(output$gelman.diag$psrf))
    print("mpsrf:")
    print(output$gelman.diag$mpsrf)
    if(gelman.plot) coda::gelman.plot(output$mcmc_list)
    if(traceplot) plot(output$mcmc_list)
  }
  
  
  
  
  # estimation_mean _____________________________________________________________________________
  estimation <- list()
  estimation$mu_df  <- estimation$se2_df <- NULL
  estimation$beta_df <- estimation$IC_inf_beta_df <- estimation$IC_sup_beta_df <- NULL
  estimation$omega_df <- NULL; estimation$lambda2_df <- NULL; estimation$tau2_df <- NULL
  if(selection){estimation$upsilon_df <- NULL}
  if(!is.null(A)){estimation$su2_df <- estimation$U_df <- NULL }
  if(model=='probit') estimation$Y_df <- NULL
  estimation$lppd_df <- estimation$Pwaic_df <- estimation$waic_df <- NULL
  d=degree
  
  for(k in 1:nrow(pars)){
    if(save) load(paste0(path, "/chain_rep_", pars$rep[k], "_fold_", pars$fold[k], ".Rdata")) else chain <- output$list_chain[[k]]
    
    estimation$mu_df <- rbind(estimation$mu_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], mu = mean(chain$mu)))
    estimation$beta_df <- rbind(estimation$beta_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], id = 1:p, beta = colMeans(chain$beta)))
    estimation$IC_inf_beta_df <- rbind(estimation$IC_inf_beta_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], id = 1:p, IC_inf_beta = apply(chain$beta,2, quantile, 0.025)))
    estimation$IC_sup_beta_df <- rbind(estimation$IC_sup_beta_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], id = 1:p, IC_sup_beta = apply(chain$beta,2, quantile, 0.975)))
    
    estimation$se2_df <- rbind(estimation$se2_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], se2=mean(chain$se2)))
    estimation$omega_df <- rbind(estimation$omega_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], id = 1:(p-nb_group*d), omega = colMeans(chain$omega)))
    estimation$lambda2_df <- rbind(estimation$lambda2_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], id = 1:nb_group, lambda2 = colMeans(chain$lambda2)))
    estimation$tau2_df <- rbind(estimation$tau2_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], tau2 = mean(chain$tau2)))
    if(selection){
      estimation$upsilon_df <- rbind(estimation$upsilon_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], id = 1:p, upsilon = colMeans(chain$upsilon)))
    }
    if(!is.null(A)){
      estimation$su2_df <- rbind(estimation$su2_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], su2 = mean(chain$su2)))
      estimation$U_df <-  rbind(estimation$U_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], id = 1:length(colMeans(chain$U)), U = colMeans(chain$U)))
    }
    if(model=='probit') estimation$Y_df <- rbind(estimation$Y_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], id = 1:length(Y = colMeans(chain$Y)),  Y = colMeans(chain$Y)))
    estimation$lppd_df <- rbind(estimation$lppd_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], lppd = chain$lppd))
    estimation$Pwaic_df <- rbind(estimation$Pwaic_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], Pwaic = chain$Pwaic))
    estimation$waic_df <- rbind(estimation$waic_df, data.frame(rep = pars$rep[k], fold = pars$fold[k], waic = chain$waic))
  }
  
  output$estimations <- estimation
  
  estimation_mean <- list()
  estimation_mean$mu <- mean(estimation$mu_df$mu)
  estimation_mean$beta <- aggregate(estimation$beta_df$beta, by = list(id = estimation$beta_df$id), FUN = mean)$x
  estimation_mean$IC_inf_beta <- aggregate(estimation$IC_inf_beta_df$IC_inf_beta, by = list(id = estimation$IC_inf_beta_df$id), FUN = mean)$x
  estimation_mean$IC_sup_beta <- aggregate(estimation$IC_sup_beta_df$IC_sup_beta, by = list(id = estimation$IC_sup_beta_df$id), FUN = mean)$x
  estimation_mean$omega <- aggregate(estimation$omega_df$omega, by = list(id = estimation$omega_df$id), FUN = mean)$x
  estimation_mean$lambda2 <- aggregate(estimation$lambda2_df$lambda2, by = list(id = estimation$lambda2_df$id), FUN = mean)$x
  estimation_mean$tau2 <- mean(estimation$tau2_df$tau2)
  if(selection){
    estimation_mean$upsilon <- aggregate(estimation$upsilon_df$upsilon, by = list(id = estimation$upsilon_df$id), FUN = mean)$x
  }
  if(!is.null(A)){
    estimation_mean$su2 <- mean(estimation$su2_df$su2)
    estimation_mean$U <- aggregate(estimation$U_df$U, by = list(id = estimation$U_df$id), FUN = mean)$x
  }
  if(model=='probit') estimation_mean$Y <- aggregate(estimation$Y_df$Y, by = list(id = estimation$Y_df$id), FUN = mean)$x
  estimation_mean$se2      <- mean(estimation$se2_df$se2)
  estimation_mean$lppd     <- mean(estimation$lppd_df$lppd)
  estimation_mean$Pwaic    <- mean(estimation$Pwaic_df$Pwaic)
  estimation_mean$waic     <- mean(estimation$waic_df$waic)
  output$estimations_mean  <- estimation_mean
  output$waic              <- estimation_mean$waic
  
  
  
  
  
  #_________________________________________________________________________________________
  
  
  if(CV>1){
    rmse_cv_folds <- F1_score_cv_folds <- matrix(NA, rep, CV)
    rownames(rmse_cv_folds) <- rownames(F1_score_cv_folds) <- paste0("rep_", 1:rep)
    colnames(rmse_cv_folds) <- colnames(F1_score_cv_folds) <- paste0("fold_", 1:CV)
    
    for(i in 1:rep){
      for(k in 1:CV){
        mu <- estimation$mu_df$mu[estimation$mu_df$rep == i & estimation$mu_df$fold == k]
        beta <- estimation$beta_df$beta[estimation$beta_df$rep == i & estimation$beta_df$fold == k]
        if(!is.null(A)) U <- estimation$U_df$U[estimation$U_df$rep == i & estimation$U_df$fold == k]
        
        if( is.null(A)) y_pred <- mu + X[which(id.cv[[i]] == k), ]%*% beta else y_pred <- mu + X[which(id.cv[[i]] == k), ]%*% beta + Z[which(id.cv[[i]] == k), ]%*% U
        if(model == "gaussian"){
          rmse_cv_folds[i, k] <- sqrt(mean((y_pred - y[which(id.cv[[i]] == k)])^2))
        }else{
          y_pred_0_1 <- 0*(y_pred<0) + 1*(y_pred>=0)
          TP <- sum(y_pred_0_1[which(y[which(id.cv[[i]] == k)] == 1)])
          FP <- sum(y_pred_0_1[which(y[which(id.cv[[i]] == k)] == 0)])
          FN <- sum(y[which(id.cv[[k]] == k)]) - TP
          precision = TP / (TP + FP)
          recall = TP / (TP + FN)
          F1_score_cv_folds[i, k] <- 2*precision*recall/(precision + recall)
          # F1_score_cv_folds[k] <- MLmetrics::F1_Score(y[which(id.cv[[k]] == k)], y_pred_0_1)
        }
      }
      if(model == "gaussian"){
        output$rmse_cv_folds <- rmse_cv_folds
        output$rmse_cv <- mean(rmse_cv_folds)
      }else{
        output$F1_score_cv_folds <- F1_score_cv_folds
        output$F1_score_cv <- mean(F1_score_cv_folds)
      }
    }
  }
  
  
  
  return(output)
  
}








#' Plot the estimated effects with credible intervals
#'
#' @param fit object from \code{sparse_group_fusion_horseshoe} function
#' @param ... Arguments to be passed to \code{plot} function, such as graphical parameters
#'
#' @return
#' @export
#'
#' @examples
plot.effect <- function(fit, type="l", ...){
  plot(fit$estimations_mean$beta, type=type, ...)
  lines(fit$estimations_mean$IC_inf_beta, lty = 3)
  lines(fit$estimations_mean$IC_sup_beta, lty = 3)
  abline(v=c(0, cumsum(table(fit$settings$group))+0.5))
  abline(0, 0, lty = 2)
}





















