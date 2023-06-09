% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/lgb.Dataset.R
\name{setinfo}
\alias{setinfo}
\alias{setinfo.lgb.Dataset}
\title{Set information of an \code{lgb.Dataset} object}
\usage{
setinfo(dataset, ...)

\method{setinfo}{lgb.Dataset}(dataset, name, info, ...)
}
\arguments{
\item{dataset}{Object of class \code{lgb.Dataset}}

\item{...}{other parameters (ignored)}

\item{name}{the name of the field to get}

\item{info}{the specific field of information to set}
}
\value{
the dataset you passed in
}
\description{
Set one attribute of a \code{lgb.Dataset}
}
\details{
The \code{name} field can be one of the following:

\itemize{
    \item{\code{label}: vector of labels to use as the target variable}
    \item{\code{weight}: to do a weight rescale}
    \item{\code{init_score}: initial score is the base prediction lightgbm will boost from}
    \item{\code{group}: used for learning-to-rank tasks. An integer vector describing how to
        group rows together as ordered results from the same set of candidate results to be ranked.
        For example, if you have a 100-document dataset with \code{group = c(10, 20, 40, 10, 10, 10)},
        that means that you have 6 groups, where the first 10 records are in the first group,
        records 11-30 are in the second group, etc.}
}
}
\examples{
\donttest{
data(agaricus.train, package = "lightgbm")
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label = train$label)
lgb.Dataset.construct(dtrain)

labels <- lightgbm::getinfo(dtrain, "label")
lightgbm::setinfo(dtrain, "label", 1 - labels)

labels2 <- lightgbm::getinfo(dtrain, "label")
stopifnot(all.equal(labels2, 1 - labels))
}
}
