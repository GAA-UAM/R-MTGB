
setwd("D:/Post_PhD/Programming/Py/HyperFree_tex/NoiseAwareBoost/experiments/RealCase/demsar")

f <- file.choose('total.csv')
data<-read.csv(file='total.csv')
data<-read.csv(f)
head(data)

source("demsar.r")
CD(3,849,0.05)
fr.test(data)
ranks(data)
plotDemsar(data, c("MTGB", "R-MTGB",  "ST-GB"))

								




