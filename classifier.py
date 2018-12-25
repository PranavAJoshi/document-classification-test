# Import all modules needed
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Get dataset from csv file
names = ['type', 'content']
dataframe = pandas.read_csv('shuffled-full-set-hashed.csv', names=names)

# Show the composition of our dataset
print(dataframe.type.size)
dataframe.type.value_counts().plot(kind='pie', figsize=(16,16))

# Split dataset
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(dataframe, dataframe.type, test_size=0.20, random_state=0)

# Feature engineering
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(X_train.content.values.astype('U')))
