######### To impute the missing categorical values  using Random Forest Algorithm m###########

library(missForest)
library(doParallel)

df = read.csv("C:\\Assignments\\xgb\\test.csv")

### Function to replace the blank values in categorical features to NA
cleaner <- function(x) {
  if (class(x)=="factor") {
    temp = as.factor(gsub("[?]",NA,as.character(x)))
  } else(x)
}

## Create test data frame with NA in place of blank cells 
test = data.frame(lapply(df,function(x) cleaner(x)))
test1 = test[,3:ncol(test)]

## Parallel processing - create cluster to speed up the Random Forest algorithm
cl <- makeCluster(4)
registerDoParallel(cl)
ptm <- proc.time()

df_imputed = missForest(test1,parallelize = "forests")
proc.time() - ptm
stopCluster(cl)

## Saving the new data frame with imputed values on hard disk
df_new = df_imputed$ximp
write.csv(x = df_new,"C:\\Assignments\\xgb\\test_imputed.csv")
