library(text2vec)
library(glmnet)

myvocab = scan("myvocab.txt", what = character(), quiet = TRUE)
train = read.table("train.tsv", stringsAsFactors = FALSE, header = TRUE)
test = read.table("test.tsv", stringsAsFactors = FALSE, header = TRUE)

vectorizer = vocab_vectorizer(create_vocabulary(myvocab, ngram = c(1, 2)))
it_train = itoken(train$review, preprocessor = tolower,
                  tokenizer = word_tokenizer)
dtm_train = create_dtm(it_train, vectorizer)
it_test = itoken(test$review, preprocessor = tolower,
                 tokenizer = word_tokenizer)
dtm_test = create_dtm(it_test, vectorizer)
set.seed(0)
glm = cv.glmnet(
  dtm_train,
  train$sentiment,
  family = "binomial",
  alpha = 0,
  type.measure = "auc"
)
write.table(
  data.frame(
    id = test$id,
    prob = predict(glm, dtm_test, glm$lambda.min, type = "response")[, 1]
  ),
  "mysubmission.txt",
  row.names = FALSE,
  sep = "\t"
)