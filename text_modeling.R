library(MASS)
library(klaR)
# for the Naive Bayes modelling
library(ggplot2)
library(lattice)
library(caret)
# to process the text into a corpus
library(NLP)
library(tm)
# to get nice looking tables
library(pander)
# to simplify selections
library(dplyr)

library(doSNOW)
cl <- makeCluster(4, type="SOCK")
registerDoSNOW(cl)

frqtab <- function(x, caption) {
  round(100*prop.table(table(x)), 1)}

sumpred <- function(cm) {
  summ <- list(TN=cm$table[1,1],  # true negatives
               TP=cm$table[2,2],  # true positives
               FN=cm$table[1,2],  # false negatives
               FP=cm$table[2,1],  # false positives
               acc=cm$overall["Accuracy"],  # accuracy
               sens=cm$byClass["Sensitivity"],  # sensitivity
               spec=cm$byClass["Specificity"])  # specificity
  lapply(summ, FUN=round, 2)
}

convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("Absent", "Present"))
}

data <- read.csv("~/R/STAT295/Project/political_social_media.csv")

data.clean <- data[,c(-2, -3, -12,-13,-14,-16,-19)]

text_corpus <- Corpus(VectorSource(data.clean$text))
text_corpus_clean <- text_corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace) %>%
  tm_map(stemDocument)
text_dtm <- DocumentTermMatrix(text_corpus_clean)

text_dtm <- removeSparseTerms(text_dtm, 0.99)

text_dtm

set.seed(3)

train_index <- createDataPartition(data.clean$audience, p=0.75, list=FALSE)
text_raw_train <- data.clean[train_index,]
text_raw_test <- data.clean[-train_index,]
text_corpus_clean_train <- text_corpus_clean[train_index]
text_corpus_clean_test <- text_corpus_clean[-train_index]
text_dtm_train <- text_dtm[train_index,]
text_dtm_test <- text_dtm[-train_index,]

ft_orig <- frqtab(data.clean$audience)
ft_train <- frqtab(text_raw_train$audience)
ft_test <- frqtab(text_raw_test$audience)
ft_df <- as.data.frame(cbind(ft_orig, ft_train, ft_test))
colnames(ft_df) <- c("Original", "Training set", "Test set")
ft_df

text_dict <- findFreqTerms(text_dtm_train, lowfreq=5)
text_train <- DocumentTermMatrix(text_corpus_clean_train, list(dictionary=text_dict))
text_test <- DocumentTermMatrix(text_corpus_clean_test, list(dictionary=text_dict))


text_train <- text_train %>% apply(MARGIN=2, FUN=convert_counts)
text_test <- text_test %>% apply(MARGIN=2, FUN=convert_counts)

ctrl <- trainControl(method="cv", 10)

#Naive Bayes Model
set.seed(1)

text_model1 <- train(text_train, text_raw_train$audience, method="nb",
                     trControl=ctrl)
text_model1

text_predict1 <- predict(text_model1, text_test)
cm1 <- confusionMatrix(text_predict1, text_raw_test$audience, positive="constituency")
cm1

#Tree Model (working on it)

library(tree)

text_forest_dat <- data.frame(text_raw_train$audience, text_train)

text_tree <- tree(text_raw_train.audience~.,data=text_forest_dat, 
                  weights=text_raw_train$audience.confidence, mincut = 10)

plot(text_tree)
text(text_tree)

text_predict_tree <- predict(text_tree, newdata=data.frame(text_test))
text_predict_tree1 <- character()

for(i in 1:1249){
  if(text_predict_tree[i,1]>=0.5){
    text_predict_tree1[i] <- 'constituency'
  }
  else{
    text_predict_tree1[i] <- 'national'
  }
}

cm_tree <- confusionMatrix(text_predict_tree1, text_raw_test$audience, positive="constituency")
cm_tree

library(randomForest)

bag.text <- randomForest(text_raw_train.audience~.,data=text_forest_dat, mtry = 20,
                         weights=text_raw_train$audience.confidence, importance=TRUE)
bag.text

bag.predict <- predict(bag.text, newdata=data.frame(text_test))

cm_bag <- confusionMatrix(bag.predict, text_raw_test$audience, positive="constituency")
cm_bag

#Random Forest Model
set.seed(2)

text_model2 <- train(text_train, text_raw_train$audience, method="ranger",
                     trControl=ctrl, weights=text_raw_train$audience.confidence)
text_model2

save(text_model2,file="~/R/STAT295/Project/tree_model.RData" )

text_predict2 <- predict(text_model2, text_test)
cm2 <- confusionMatrix(text_predict2, text_raw_test$audience, positive="constituency")
cm2

#SVM model
text_dtm <- removeSparseTerms(text_dtm, 0.94)

text_dtm

set.seed(1)

train_index <- createDataPartition(data.clean$audience, p=0.75, list=FALSE)
text_raw_train <- data.clean[train_index,]
text_raw_test <- data.clean[-train_index,]
text_corpus_clean_train <- text_corpus_clean[train_index]
text_corpus_clean_test <- text_corpus_clean[-train_index]
text_dtm_train <- text_dtm[train_index,]
text_dtm_test <- text_dtm[-train_index,]

ft_orig <- frqtab(data.clean$audience)
ft_train <- frqtab(text_raw_train$audience)
ft_test <- frqtab(text_raw_test$audience)
ft_df <- as.data.frame(cbind(ft_orig, ft_train, ft_test))
colnames(ft_df) <- c("Original", "Training set", "Test set")
ft_df

text_dict <- findFreqTerms(text_dtm_train, lowfreq=5)
text_train <- DocumentTermMatrix(text_corpus_clean_train, list(dictionary=text_dict))
text_test <- DocumentTermMatrix(text_corpus_clean_test, list(dictionary=text_dict))

text_train <- text_train %>% apply(MARGIN=2, FUN=convert_counts)
text_test <- text_test %>% apply(MARGIN=2, FUN=convert_counts)

text_raw_train$audience2 <- as.character(text_raw_train$audience)
text_raw_test$audience2 <- as.character(text_raw_test$audience)

set.seed(3)
text_model3 <- train(text_train, text_raw_train$audience2, method="svmLinear",
                     trControl=ctrl)
text_model3

text_predict3 <- predict(text_model3, text_test)
cm3 <- confusionMatrix(text_predict3, text_raw_test$audience2, positive="constituency")
cm3
