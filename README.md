# HeavyWater Machine Learning Problem



### Problem Statement

Generally, the goal of this problem is to build a text classification model based on machine learning. Input is given as a csv file, which contains all documents that serve as our dataset to train and test our model. The document entry format is like this:

```
CANCELLATION NOTICE,641356219cbc f95d0bea231b ... [lots more words] ... 52102c70348d b32153b8b30c
```

So each document is composed of a label which indicates document type at its front, and a series of obscured OCR(Optical Character Recognition) data seperated by space and each of them maps to a unique word in original document.



### General Steps

Since the label is given, we will do supervised training on our dataset. Our main steps are:
1. Dataset Preparation - Load our dataset and perform basic pre-processing. E.g. figure our our dataset's composition and split dataset into train and test sets.
2. Feature Engineering - Transform raw dataset into flat features that can be used in our machine learning model.
3. Model Training - Train the model on labelled dataset.



### Step 1: Dataset Preparation

Let's have a glance of our dataset first. There're 62204 documents in total and their distribution is shown below:

![](images/data_plot.jpeg)

Then we need to split our dataset for training and testing. The training part is used to train our model to make correct classification. And the testing part is used to validate the model's correctness. 
The widely used machine learning library *sklearn* provides us with a powerful method-*model_selection.train_test_split*- to do this split. 
The doc for *model_selection.train_test_split* is provided below:
```
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
```



### Step 2: Feature Engineering

Our raw obscured text dataset cannot work directly with machine learning algorithms, which needs numeric data representation. So in this step we need to convert the text to numbers. 

A simple and effective model for thinking about text documents in machine learning is called the *Bag-of-Words Model*, or *BoW*. The model is simple in that it throws away all of the order information in the words and focuses on the occurrence of words in a document. The *sklearn* library provides 3 different schemes(Word Counts with *CountVectorizer*, Word Frequencies with *TfidfVectorizer* and Hashing with *HashingVectorizer*) that we can use to achieve this goal, they are introduced in the following link:
```
https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
```
In our solution we will use *TfidfVectorizer* because it's a refined version of *CountVectorizer* and easier to implement than *HashingVectorizer*. Since the vectorizer requires the raw_documents to be str, unicode or file objects type, we will convert our dataframe into unicode before transforming the data. 

After we vectorize our data, let's print and check the numeric vector.
```
  (0, 876448)	0.019543002423305235
  (0, 875142)	0.08848690989316463
  (0, 873052)	0.023710767681637043
  (0, 870760)	0.08818256683793115
  (0, 870189)	0.027476905355595008
  (0, 869897)	0.03056342830737312
  (0, 863826)	0.015110268037881867
  (0, 862915)	0.019683220920096518
  :	:
  (49762, 51494)	0.036718963942246566
  (49762, 46393)	0.0736156545180594
  (49762, 31103)	0.04474390770983401
  (49762, 30617)	0.05245314736807223
  (49762, 30119)	0.0801536701925495
  (49762, 11862)	0.12520145925333384
  (49762, 11523)	0.014843975855234373
  (49762, 10932)	0.0646270429142683
```
Now we are good to use this feature set to train our model.
