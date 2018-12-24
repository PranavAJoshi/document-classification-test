# Import all modules needed
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection

# Get dataset from csv file
names = ['type', 'content']
dataframe = pandas.read_csv('shuffled-full-set-hashed.csv', names=names)

# Show the composition of our dataset
print(dataframe.type.size)
dataframe.type.value_counts().plot(kind='pie', figsize=(16,16))
