
setwd("D:/Post_PhD/Programming/Py/NoiseAwareBoost/experiments/RealCase/demsar")

f <- file.choose('clf_pivot.csv')
data<-read.csv(file='clf_pivots.csv')
data<-read.csv(f)
head(data)

source("demsar.r")
CD(3,472,0.05)
fr.test(data)
ranks(data)
plotDemsar(data, c("MTGB", "R-MTGB",  "ST-GB"))

								




