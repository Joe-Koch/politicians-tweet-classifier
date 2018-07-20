#Importing Social Media Post Data
install.packages(c('tm', 'SnowballC', 'wordcloud', 'topicmodels', 'stringr'))
library(stringr)
library(tm)


data <- read.csv("~/R/STAT295/Project/political_social_media.csv")

data.clean <- data[,c(-2, -3, -12,-13,-14,-16,-19)]

bias <- Corpus(VectorSource(data.clean$text[data.clean$bias=='partisan'&data.clean$source=="twitter"]))
docs <- bias

docs <- Corpus(VectorSource(data.clean$text))


#Transform to lower case
text_corpus <- Corpus(VectorSource(data.clean$text))
text_corpus_clean <- text_corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace) %>%
  tm_map(stemDocument)
dtm <- DocumentTermMatrix(text_corpus_clean)

dtm <- removeSparseTerms(dtm, 0.99)

dtm

m <- as.matrix(dtm)
#write as csv file (optional)
write.csv(m,file="~/R/STAT295/Project/wordmatrix.csv")

#shorten rownames for display purposes
rownames(m) <- paste(substring(rownames(m),1,3),rep("..",nrow(m)),
                     substring(rownames(m), nchar(rownames(m))-12,nchar(rownames(m))-4))

#compute distance between document vectors
d <- dist(m)

#run hierarchical clustering using Ward's method
groups <- hclust(d,method="ward.D")

#plot dendogram, use hang to ensure that labels fall below tree
plot(groups, hang=-1)

groups_2 <- cutree(groups,5)

bias_1 <- bias[groups_2==1]
bias_2 <- bias[groups_2==2]
bias_3 <- bias[groups_2==3]
bias_4 <- bias[groups_2==4]
bias_5 <- bias[groups_2==5]

inspect(sample(bias_3, 10))

#cut into 2 subtrees - try 3 and 5
rect.hclust(groups,5)

#k means algorithm, 2 clusters, 100 starting configurations
kfit <- kmeans(d, 2, nstart=10)
#plot - need library cluster
library(cluster)
clusplot(m, kfit$cluster, color=T, shade=T, labels=2, lines=0)