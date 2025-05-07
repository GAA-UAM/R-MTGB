##
# Description: Implements Demsar's way of comparing multiple classifiers
# Author: Daniel Hernández Lobato
# Date: December 2007
#

##
# Function that performs a friedman test over a matrix whose rows 
# contain error rates of each classifier.
#
# x is a matrix kxn
#

fr.test <- function(x) friedman.test(as.matrix(x))$p.value

##
# Function that computes ranks over a matrix whose rows 
# contain error rates of each classifier.
#
# x is a matrix kxn
#

ranks <- function(x) {
	t.default(apply(as.matrix(x), 1, rank))
}

##
# Function that computes average ranks over a matrix whose rows 
# contain error rates of each classifier.
#
# x is a matrix kxn
#

avg_ranks <- function(x) {
	apply(t.default(apply(as.matrix(x), 1, rank)), 2, mean)
}

##
# Function that computes critical values of the Nemenyi test
#
# k <- Number of algorithms tested
# n <- Number of datasets
#

CD <- function(k, n, pvalue = 0.05) {

	# Critical values according to Demsar 

	qa <- c( 0, qtukey(1 - pvalue, 2:5000, +Inf) / sqrt(2))
	
	qa[ k ] * sqrt(k * (k + 1) / (6 * n))
}

##
# Function that computes a matrix of average rank differences
#
# x is a matrix kxn
#

avg_ranks_diff <- function(x) {

	v <- avg_ranks(x)
	
	m <- matrix(rep(v, length(v)), length(v), length(v))

	abs(m - t(m))
}

##
# Function that returns groups for each algorithms according to Nemenyi test
#
# x is a matrix kxn
#

groupsNemenyi <- function(x, pvalue = 0.05) {

	cd <- CD(ncol(x), nrow(x), pvalue)
	diffs <- avg_ranks_diff(x)
	ordering <- rev(sort(avg_ranks(x), index.return = T)$ix)

	groups <- list()

	# Function that searches for the group a fixed element belongs to

	searchGroup <- function(element, diffs, currentGroup = c()) {
			
		ret <- currentGroup
		if (is.null(currentGroup)) {
			currentGroup <- element	
			searchGroup(element, diffs, currentGroup)
		} else {
			for (i in setdiff(ordering, currentGroup )) {
				if (all(c(diffs[ i, currentGroup ]) < cd))	 {
					ret <- searchGroup(element, diffs, c(currentGroup, i))
					break;
				}
			}
			ret
		}
	}

	groups[[ 1 ]] <- searchGroup(ordering[ 1 ], diffs)
	current <- 2

	for (i in 2 : ncol(x)) {

		# We check if the group exists

		gnew <- searchGroup(ordering[ i ], diffs)

		exists <- FALSE
		for (j in 1 : length(groups)) {
			if (length(groups[[ j ]]) == length(gnew) && all(sort(groups[[ j ]]) == sort(gnew))) {
				exists <- TRUE 
				break;
			}
		}
		
		if (! exists) {
			groups[[ current ]] <- gnew
			current <- current + 1
		} 
	}

	groups
}

##
# Function that plots results as in demsar's work
#
# x is a matrix kxn
# names : vector with algorithms names
# postscript : name of the postscript file where the image will be saved. If null the image
# 	is just ploted on the screen.
# pvalue : confidence level of the Nemenyi test.
#

plotDemsar <- function(x, names = NULL, postscript = NULL, pvalue = 0.05) {

	ranks <- avg_ranks(x)	

	min_rank <- min(ranks)
	max_rank <- max(ranks)

	idx <- rev(sort(ranks, index.return = TRUE)$ix)
	idx <- c(idx[ 1 :  (as.integer(length(ranks) / 2)) ], rev(idx[  (as.integer(length(ranks) / 2) + 1) : length(idx) ]))
	
	y <- ncol(x) * 4
	offset <- y / 8

	# Check werther or not we have to save the image

	if (is.null(postscript))
		x11(width = 6.5, height = 3.5)
	else
		postscript(postscript, width = 6.5, height = 3.5)

	plot(min_rank, 0, xlim = c(floor(min_rank - .1), ceiling(max_rank + 1.1)), 
		ylim = c(0, y),  xlab = "", ylab = "", col = "white", bty = "n", yaxt = "n")
  #axis(at=(4:12),side=1)
	# We draw axises and algorithms labels

	for (i in 1 : (as.integer(length(ranks) / 2))) 
		text(max_rank + .1, i * offset + 3.5, 
			if (is.null(names)) 
				paste("#", idx[ i ], sep = "")
			else
				names[ idx[ i ] ]
		, xpd = TRUE, pos = 4)

	for (i in ((as.integer(length(ranks) / 2)) + 1) : length(ranks)) 
		text(min_rank - .1, (i - (as.integer(length(ranks) / 2))) * offset + 3.5, 
			if (is.null(names))
				paste("#", idx[ i ], sep = "")
			else 
				names[ idx[ i ] ]
			, xpd = TRUE, pos = 2)

	# We draw lines 

	for (i in 1 : (as.integer(length(ranks) / 2)))  {
		lines(c(max_rank + .1, ranks[ idx[ i ] ]),c(i * offset + 3.5, i * offset + 3.5), lwd = 1)
		lines(c(ranks[ idx[ i ] ], ranks[ idx[ i ] ]),c(i * offset + 3.5, 0), lwd = 1)
	}
	
	for (i in ((as.integer(length(ranks) / 2)) + 1) : length(ranks)) {
		lines(c(min_rank - .1, ranks[ idx[ i ] ]),c((i - (as.integer(length(ranks) / 2))) * offset + 3.5, 
			(i - (as.integer(length(ranks) / 2))) * offset + 3.5), lwd = 1)
		lines(c(ranks[ idx[ i ] ], ranks[ idx[ i ] ]),c((i - (as.integer(length(ranks) / 2))) * offset + 3.5, 0), lwd = 1)
	}

	# We draw groups

	CD <- CD(ncol(x), nrow(x), pvalue = pvalue)
	groups <- groupsNemenyi(x, pvalue = pvalue)

	for (i in 1 : length(groups)) {
		min <- min(ranks[ groups[[ i ]] ])
		max <- max(ranks[ groups[[ i ]] ])
		lines(c(min - 0.02, max + 0.02), c(i * (offset / 3), i * (offset / 3)), lwd = 3)
	}

	# We draw CD

	arrows(min_rank, y - offset / 2, CD + min_rank, y - offset / 2, code = 3, angle = 20)
	
	text(min_rank + CD / 2, y - offset / 8, "CD")

	if (!is.null(postscript)) dev.off()
}




