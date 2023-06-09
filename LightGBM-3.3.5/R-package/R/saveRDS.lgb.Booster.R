#' @name saveRDS.lgb.Booster
#' @title saveRDS for \code{lgb.Booster} models
#' @description Attempts to save a model using RDS. Has an additional parameter (\code{raw})
#'              which decides whether to save the raw model or not.
#' @param object \code{lgb.Booster} object to serialize.
#' @param file a connection or the name of the file where the R object is saved to or read from.
#' @param ascii a logical. If TRUE or NA, an ASCII representation is written; otherwise (default),
#'              a binary one is used. See the comments in the help for save.
#' @param version the workspace format version to use. \code{NULL} specifies the current default
#'                version (2). Versions prior to 2 are not supported, so this will only be relevant
#'                when there are later versions.
#' @param compress a logical specifying whether saving to a named file is to use "gzip" compression,
#'                 or one of \code{"gzip"}, \code{"bzip2"} or \code{"xz"} to indicate the type of
#'                 compression to be used. Ignored if file is a connection.
#' @param refhook a hook function for handling reference objects.
#' @param raw whether to save the model in a raw variable or not, recommended to leave it to \code{TRUE}.
#'
#' @return NULL invisibly.
#'
#' @examples
#' \donttest{
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(
#'   objective = "regression"
#'   , metric = "l2"
#'   , min_data = 1L
#'   , learning_rate = 1.0
#' )
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'     params = params
#'     , data = dtrain
#'     , nrounds = 10L
#'     , valids = valids
#'     , early_stopping_rounds = 5L
#' )
#' model_file <- tempfile(fileext = ".rds")
#' saveRDS.lgb.Booster(model, model_file)
#' }
#' @export
saveRDS.lgb.Booster <- function(object,
                                file,
                                ascii = FALSE,
                                version = NULL,
                                compress = TRUE,
                                refhook = NULL,
                                raw = TRUE) {

  # Check if object has a raw value (and if the user wants to store the raw)
  if (is.na(object$raw) && raw) {

    object$save()

    saveRDS(
      object
      , file = file
      , ascii = ascii
      , version = version
      , compress = compress
      , refhook = refhook
    )

    # Free model from memory
    object$raw <- NA

    return(invisible(NULL))

  } else {

    saveRDS(
      object
      , file = file
      , ascii = ascii
      , version = version
      , compress = compress
      , refhook = refhook
    )

    return(invisible(NULL))

  }

}
