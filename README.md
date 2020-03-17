# NLP_proj1
url:
Input: an article(with or without title)
Output: a concised version of the article, which conclude the most significent part of the article.
Time_Consumption:about 10s

source of data:
1.Wikipedia Chinese Version
2.News Dataset

Desciption of Process:
1. Use jieba to divded the datasets into words
2. Train the word2vec model by processed data(300 dimension)
3. Use SIF tech to generate the sentence vector
    1. a/（a+p_w) (a is a smooth varible, p_w is the frequency of the word)
    2. Use SVD to decrease the dimension
4. Calculate the similarity of the sentence vector with the article vector
5. Using Convolution and Knn to smooth the similarity.
6. Choosing the top 20% sentences after smooth.
7. Sort the choosen sentence by original order.
