
setwd("D:/Post_PhD/Programming/Py/HyperFree_tex/NoiseAwareBoost/experiments/RealCase/demsar")

f <- file.choose('reg_pivot.csv')
data<-read.csv(file='reg_pivot.csv')
data<-read.csv(f)
head(data)

source("demsar.r")
CD(3,367,0.05)
fr.test(data)
ranks(data)
plotDemsar(data, c("MTB", "RMTB",  "STL"))

								




