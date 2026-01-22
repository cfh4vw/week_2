"""
ML1 In-Class
.py file
"""
# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
# (https://colab.research.google.com/github.com/UVADS/DS-3021/blob/main/
# 04_ML_Concepts_I_Foundations/ML1_inclass.py#scrollTo=9723a7ee)

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
# (https://colab.research.google.com/github.com/UVADS/DS-3001/blob/main/
# 04_ML_Concepts_I_Foundations/ML1_inclass.ipynb#scrollTo=9723a7ee)
# %%

# %%
# import packages
# from turtle import color
from pydataset import data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
# set the dimension on the images
import plotly.io as pio
pio.templates.default = "plotly_dark" # set dark theme

# %%
iris = data('iris')
iris.head()

# %%
# What mental models can we see from these data sets?
# A mental model that we can see from these data set is
# that there are different species of iris flowers that can be
# classified based on their sepal and petal dimensions.
# Another mental model is that the dimensions of the flowers
# can be used to predict the species of the flower.

# What data science questions can we ask?
# Some data science questions that we can ask from these data set are:
# 1. Can we classify the species of iris flowers based on their
#    sepal and petal dimensions?
# 2. What are the key features that differentiate the species of
#    iris flowers?

# %%
"""
Example: k-Nearest Neighbors
"""
# We want to split the data into train and test data sets. To do this,
# we will use sklearn's train_test_split method.
# First, we need to separate variables into independent and dependent
# dataframes.

X = iris.drop(['Species'], axis=1).values  # features
y = iris['Species'].values  # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
# we can change the proportion of the test size; we'll go with 1/3 for now

# %%
# Now, we use the scikitlearn k-NN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# %%
# now, we check the model's accuracy:
neigh.score(X_train, y_train)

# %%
# now, we test the accuracy on our testing data.
neigh.score(X_test, y_test)

# %%
"""
Patterns in data
"""
# Look at the following tables: do you see any patterns? How could a
# classification model point these out?
patterns = iris.groupby(['Species'])
patterns['Sepal.Length'].describe()

# %%
patterns['Sepal.Width'].describe()

# %%
patterns['Petal.Length'].describe()

# %%
patterns['Petal.Width'].describe()

# %%
# scatter plot using plotly
fig = px.scatter_3d(iris, x='Sepal.Length', y='Sepal.Width', z='Petal.Length',
                 color='Species', title='Iris Sepal Dimensions')
fig.show()
# %%
"""
Mild disclaimer
"""
# Do not worry about understanding the machine learning in this example!
# We go over kNN models at length later in the course; you do not need to
# understand exactly what the model is doing quite yet.
# For now, ask yourself:

# 1. What is the purpose of data splitting?
# Data splitting allows us to create separate datasets for training 
# and testing our model. This helps us to evaluate how well the model
# performs and therefore how well it can be generalized.

# 2. What can we learn from data testing/validation?
# Data testing/validation helps us to assess the performance
# of our model on unseen data. It provides insights into the model's
# accuracy, robustness, and ability to generalize to new data.

# 3. How do we know if a model is working?
# We can compare the model's predictions to a certain baseline
# (e.g., random guessing or majority class prediction) to see if
# it is performing better than that baseline. Also, examining the error
# rate on the test data can help us determine if the model is working.

# 4. How could we find the model error?
# We can find the model error by assesing the rate of incorrect predictions
# made by the model on the test data.

# If you want, try changing the size of the test data
# or the number of n_neighbors and see what changes!

# %%
