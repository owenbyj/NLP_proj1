# NLP_proj1
URL:http://35.194.183.66/key_extractor/


## Input: 
an article(with or without title)

## Output: 
a concised version of the article, which conclude the most significent part of the article.

## Time_Consumption:
about 10s

## source of data:
1.Wikipedia Chinese Version
2.News Dataset

## Desciption of Process:
1. Use jieba to divded the datasets into words
2. Train the word2vec model by processed data(300 dimension)
![Markdown](http://i2.tiimg.com/713520/70ff652c18dd8491.jpg)

3. Use SIF tech to generate the sentence vector
    1. a/（a+p_w) (a is a smooth varible, p_w is the frequency of the word)
    2. Use SVD to decrease the dimension
    3. Cope with the words not does not appear in the data sets. We create a UNK vector.
        We randomly pick 10 percent of the sentences and randomly change some word to UNK to train the UNK vector.
4. Calculate the similarity of the sentence vector with the article vector
5. Smooth to make the processed article more readable
    1. knn
        find the k nearest sentences, and calculate the mean Euclidean distance of the sentences as the similarity.
    2. convolution
        Using a method similar to text-cnn. Using a (1,n) dimension kernel to catch the pattern of context. 
        e.g. (1,4)kernel catch one sentences before and two sentences after the main sentences.
6. Choosing the top 20% sentences after smooth.
7. Sort the choosen sentence by original order.

## Adavantages
1. A very clean user interface, which is very user-friendly. 
2. Good to adapt mobile phone browser or computer web browser.
3. The main idea is completely concluded by processed.
4. We tried multiple smooth method for users to choose from.
5. The compressed article is with great readability.

## Disadvantages
1. It is not a real time program. Takes about 10s for long article.
2. To increase the completeness and readability of sentences. We separete the article by sentences, which may sometimes catch the important sentences with a little part not that related. 
