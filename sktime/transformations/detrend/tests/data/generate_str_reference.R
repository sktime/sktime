# Regenerate using Rscript path/to/generate_str_reference.R.
# Reference: robjhyndman/stR at c6bb49e7c6e42ea2d965195b9546bcb0edcae842.
# Uses the upstream STRmodel implementation, not Python-generated outputs.
# R 4.3.3, Matrix 1.6-5. R is needed only to regenerate these fixtures.
library(compiler)
library(Matrix)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE))
output_dir <- dirname(normalizePath(script))
base <- paste0("https://raw.githubusercontent.com/robjhyndman/stR/",
               "c6bb49e7c6e42ea2d965195b9546bcb0edcae842/R/")
for (file in c("STR.R", "lmSolver.R")) {
  local <- tempfile(fileext=".R")
  download.file(paste0(base, file), local, quiet=TRUE)
  source(local)
  unlink(local)
}
.solve.dgC.chol <- Matrix:::.solve.dgC.chol
make_predictor <- function(n, p, nk, nt, weights, trend=FALSE) {
  list(name=if (trend) "trend" else paste0("seasonal_", p),
       data=rep(1, n), times=0:(n-1),
       seasons=if (trend) rep(1, n) else (1:n) %% p,
       timeKnots=seq(0, n-1, length.out=nt),
       seasonalStructure=list(segments=list(c(0, p)),
         sKnots=if (trend) list(c(1,0)) else c(as.list(seq(p/nk, p-p/nk, length.out=nk-1)), list(c(p,0)))),
       lambdas=weights)
}
for (case in 1:2) {
  n <- 24
  t <- 0:(n-1)
  y <- 3 + .07*t + (1+.02*t)*sin(2*pi*(t+1)/3) + .15*cos(t*1.7)
  periods <- if (case==1) c(3) else c(3,6)
  if (case==2) y <- y + .4*cos(2*pi*(t+1)/6)
  predictors <- list(make_predictor(n,1,1,6,c(2,0,0),TRUE))
  for (p in periods) predictors[[length(predictors)+1]] <- make_predictor(n,p,3,4,c(1.5,.3,.8))
  result <- STRmodel(y, predictors)
  output <- data.frame(data=y,
    trend=as.vector(result$output$predictors[[1]]$data),
    resid=as.vector(result$output$random$data))
  for (i in seq_along(periods)) output[[paste0("seasonal_",periods[i])]] <- as.vector(result$output$predictors[[i+1]]$data)
  write.csv(output, file.path(output_dir, paste0("str_reference_", case, ".csv")), row.names=FALSE)
}
