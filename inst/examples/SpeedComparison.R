# Munich rent data from catdata package
data("rent", package = "catdata")

# The considered predictors are the same as in 
# Gertheiss and Tutz (Ann. Appl. Stat., 2010).
# Response is monthly rent per square meter in Euro

# Urban district in Munich
rent$area <- as.factor(rent$area)

# Create formula with 'rentm' as response variable,
# 'area' with a Generalized Fused Lasso penalty,
formu <- rentm ~ p(area, pen = "gflasso")


# Quick comparison with previous package in terms of speed
rbenchmark::benchmark(smurf::glmsmurf(formula = formu, family = gaussian, data = rent, 
                                      pen.weights = "glm.stand", lambda = 5e-3,
                                      control = glmsmurf.control(epsilon = 1e-8, print = F , ncores = 1, po.ncores = 1)),
                      ssmurf::glmsmurf(formula = formu, family = gaussian, data = rent, 
                                       pen.weights = "glm.stand", lambda = 5e-3,
                                       control = glmsmurf.control(epsilon = 1e-8, print = F , ncores = 1, po.ncores = 1)), 
                      replications = 20)


# The following comparison takes a bit longer
l = c(exp(seq(log(64), log(1e-4), length.out = 50)))

# Speed comparison to the previous package
rbenchmark::benchmark(smurf::glmsmurf(formula = formu, family = gaussian, data = rent, 
                                      pen.weights = "glm.stand", lambda = "cv1se.dev",
                                      control = glmsmurf.control(epsilon = 1e-8, print = F , k = 10, ncores = 1, po.ncores = 1,
                                                                 lambda.vector = l)),
                      ssmurf::glmsmurf(formula = formu, family = gaussian, data = rent, 
                                       pen.weights = "glm.stand", lambda = "cv1se.dev",
                                       control = glmsmurf.control(epsilon = 1e-8, print = F, k = 10, ncores = 1, po.ncores = 1,
                                                                  lambda.vector = l)), replications = 10)
