library(tidyr)
library(ggplot2)
dat <- read.csv("analysis/hypotheses/likert-q3.csv")

mat <- as.matrix(dat)

res <- friedman.test(mat)
print(res)