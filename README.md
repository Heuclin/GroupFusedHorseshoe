

# The Bayesian group fused horseshoe prior

B. Heuclin, J. Gibaud, F. Mortier, C. Trottier, S. Tisné, M. Denis

20/05/2021

![](logo.png)

We propose the Bayesian group fused horseshoe prior implemented in a MCMC sampler (in R language) for the double selection of groups of variables and variables while taking into account the explicative variables indexation. 
This work is related to the paper: 
"Bayesian sparse group selection with indexed regressors within groups: the group fused horseshoe prior", B. Heuclin, J. Gibaud, F. Mortier, C. Trottier, S. Tisné and M. Denis (submitted).


* The `algo_group_fusion_horseshoe.R` file contains the R code of the MCMC sampler algorithm. 
* The `Application` folder contains the fruit abscission dataset `abscission.Rdata` and the R script `script_abscission.R` using to generate the application results of the paper.
* The `Simulation` folder constains the R script `script_simulation.R` using to generate the simulated results of the paper. 

### Description of the fruit abscission dataset `abscission.Rdata`:
It contains:

  * `DFD`, a 1173-vector which is the days from pollination to fruit drop of bunches (the response variable) and
  * `X_list`, a list containing 9 1173x121-matrices of measurements associated to the 9 environmental variables (Tmax, Tmin, RH, VPD, FTSW, DRD, SD, R and SR)

### Description of the R functions:
`algo_group_fusion_horseshoe.R` file contains the `group_fused_HS_MCMC` function which allows to run the MCMC sampler (for one chain) for the group fusion / fused horseshoe approach. 
The arguments of this function are:

* `y`: vector ofthe response variable
* `X`: the matrix of the indexing variable, if many groups are present, X is the collection of all matrix associated to each group
* `Z`: optional design matrix for a random effect
* `A`: optional covariance matrix for a random effect
* `group`:  vector of length ncol(X) indicating the number group of each variable in X
* `d`: the degree of the difference penalty
* `D`: the matrix of the finite difference operator
* `model`: c('gaussian', 'probit'), indicating is the responce is gaussian or probit regression
* `settings`: a list containing the MCMC settings:
    + `niter`: number of iterations of the MCMC chain
    + `burnin`: number of initial iterations of the MCMC chain which should be discarded
    + `thin`: save only every thin-th iteration

* `diff3levels`: boolean indicating if three levels (global, group specific and local) of variance parameters should be used on coefficient differences instead of two (global and local), default is TRUE
* `var_sel`: boolean indicating if selection of variable is needed (fused) or not (fusion), default choice is TRUE


It return a list of the MCMC samples for each parameter.




### Exemple on the fruit abscission dataset of oil palm trees

Load libraries
```{r}
rm(list = ls())
library(Matrix)
library(mvnfast)
library(invgamma)
library(truncnorm)
library(doParallel)
library(foreach)
library(coda)
```


```{r}
source("algo_group_fusion_horseshoe.R")
RMSE <- function(x, y) return(sqrt(mean((x-y)^2)))
```


Load the fruit abscission dataset  
```{r}
load("Application/abscission.Rdata")

length(DFD)
dim(X_list[[1]])

Y = DFD
id_g <- names(X_list); id_g
X <- do.call(cbind, X_list)
X_scale <- scale(X)
```




Fit the Bayesian group fused horseshoe approach
```{r}
system("mkdir results/fused_HS")

n <- nrow(Y); p <- ncol(X)
nb_group <- length(id_g)
degree = 1

length_group    <- rep(p/nb_group, nb_group)
group  <- rep(1:nb_group, times = length_group)
d      <- degree
D      <- lapply(table(group), function(p) diff(diag(p), differences = d ))

settings        <- list()
settings$niter  <- 10000
settings$burnin <- 5000
settings$thin   <- 10

chain <- group_fused_HS_MCMC(
  y=Y, X=X_scale, 
  group = group, d=d, D=D, 
  settings=settings, 
  var_sel = TRUE, diff3levels = TRUE
)
  
```

Plot estimated coefficient profiles
```{r}
plot(apply(chain$beta, 2, mean), t='l')
abline(v=c(0, cumsum(table(group))+0.5))
abline(0, 0, lty = 2)
lines(apply(chain$beta, 2, quantile, 0.025), lty=3)
lines(apply(chain$beta, 2, quantile, 0.975), lty=3)
```



