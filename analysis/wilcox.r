library(tidyr)
library(ggplot2)
library(coin)
dat <- read.csv("analysis/CHI-hypotheses/H1c-accuracy.csv")

dat$groups <- as.factor(dat$groups)
dat$participants <- as.factor(dat$participants)

hyp <- dat[dat$groups %in% c("s_S", "s_H"), ]

res <- wilcox_test(hyp$values ~ hyp$groups | hyp$participants)

