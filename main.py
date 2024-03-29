import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import matplotlib.pyplot as plt
import unicodedata
from html import unescape
from urllib import request
from scipy.stats import pearsonr
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sn


def request_json_data(url):
  """
  Request url data
  :param url:
  :return:
  """
  with request.urlopen(url) as j:
    data = json.loads(j.read().decode())
  return data


def clean_text(text):
  """
  Deal with artifacts and remove accent (e.g á -> a)
  :param text:
  :return:
  """
  text = unescape(text)
  return ''.join(c for c in unicodedata.normalize('NFD', text)
                 if unicodedata.category(c) != 'Mn')


# EXTRA data ----------------------------------------------------------------------
# actors_gender_data = pd.read_csv("https://raw.githubusercontent.com/taubergm/HollywoodGenderData/master/all_actors_movies_gender_gold.csv")
# writers_gender_data = pd.read_csv("https://raw.githubusercontent.com/taubergm/HollywoodGenderData/master/all_writers_gender.csv")
# all_movies = pd.DataFrame(request_json_data("https://bechdeltest.com/api/v1/getAllMovies"))
# imdb = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-03-09/movies.csv')
# ---------------------------------------------------------------------------------

# Download all datasets from repo
bechdel_data = pd.read_csv(
  "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-03-09/raw_bechdel.csv",
  dtype=str)
imdb = pd.DataFrame(request_json_data("https://raw.githubusercontent.com/brianckeegan/Bechdel/master/imdb_data.json"))
movies = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/bechdel/movies.csv", dtype=str)

# Data canonicalization
bechdel_data['year'] = bechdel_data['year'].astype('int')
bechdel_data['rating'] = bechdel_data['rating'].astype('int')
# movies['year'] = movies['year'].astype('object')
movies['imdbid'] = movies['imdb'].apply(lambda x: x.split('tt')[1])  # get the proper imdb id format

# To get more data on the movies we join with movies dataset that includes 1794 movies
# Merge datasets to create a new dataset
movies_data = pd.merge(bechdel_data[['imdb_id', 'rating']], movies, how='right', left_on=['imdb_id'],
                       right_on=['imdbid'])

# Imdb Dataset
imdb = imdb.dropna(subset=['imdbID'])
imdb = imdb[imdb['imdbRating'] != 'N/A']
imdb = imdb[imdb['imdbVotes'] != 'N/A']
imdb['imdbid'] = imdb['imdbID'].apply(lambda x: x.split('tt')[1])  # get the proper imdb id format
imdb['imdbRating'] = imdb['imdbRating'].astype(float)
imdb['imdbVotes'] = imdb['imdbVotes'].apply(lambda x: float(x.replace(',', '')))

# task 1 ---------------------------------------------------------------------------------------------------------------
# 1.
bechdel_data.loc[bechdel_data['year'] < 2022, ['year', 'rating']].groupby(['year']).mean().rename(
  columns={'rating': 'avg rating'}).plot()
plt.title('Average Bechdel Score Per Year')
plt.show()

# 2.
# todo: could be done using only pandas (in one line)
l1st = []
for year in sorted(set(bechdel_data['year'].values.tolist())):
  num_over_two = len(bechdel_data[(bechdel_data['year'] == year) & (bechdel_data['rating'] >= 2)])
  num_all = len(bechdel_data[bechdel_data['year'] == year])
  if num_over_two / num_all == 0.5:
    l1st.append(year)
print(f"The years where 50% of movies have a Bechdel score of 2 or higher are: {', '.join(str(y) for y in l1st)}")

# task 2 ---------------------------------------------------------------------------------------------------------------

# 1.
print("Movies, Dataset quality assessment")

print("Dataset data types and non-null rows")
movies_data.info()

movies_data.isna().sum().plot.barh()
# Missing values per column
plt.title('Missing values')
plt.show()

# Number of movies per year
print(movies_data['year'].value_counts().sort_index().plot.bar())
plt.title('Movies per year')
plt.show()

# Number of movies per rating
print(movies_data['rating'].value_counts().sort_index().plot.bar())
plt.title('Movies per rating')
plt.show()

# Search for duplicates in the dataset
# There are no duplicate rows in the dataset
print("Number of duplicate rows: ", len(movies_data[movies_data.duplicated()]))

# There are no duplicate imdb ids in the dataset
print("Number of imdb ids: ", len(movies_data[movies_data.duplicated(['imdb'], keep=False)]))

# Duplicate Titles
print("Duplicate titles")
print("Number of duplicate movie titles: ", len(movies_data[movies_data.duplicated(['title'], keep=False)]))
print("Sample of Duplicates")
print(movies_data[movies_data.duplicated(['title'], keep=False)].sort_values('title')[:15])

# We see that there are duplicate movie titles with identical budget/gross and different normalized budget/gross
print("Different movie with same budget")
print(movies_data[movies_data['title'] == 'Beautiful Creatures'][
        ['title', 'year', 'budget', 'domgross', 'intgross', 'budget_2013$', 'domgross_2013$', 'intgross_2013$']
      ])

# 2.
print(f" There are artifacts in the text (e.g. title: '{movies_data.iloc[0]['title']}')")

# We deal with the artifacts and strip and accent todo: do more cleaning if there is any
movies_data['title'] = movies_data['title'].apply(lambda x: clean_text(x))

# 3.

# Test correlation on movie features

# dataset_labels = movies_data[['imdbid', 'title', 'test', 'clean_test', 'binary', 'rating']]

movies_features = movies_data[['year', 'budget_2013$', 'domgross_2013$', 'intgross_2013$', 'rating', 'binary']]
for col in ['budget_2013$', 'domgross_2013$', 'intgross_2013$', 'rating']:
  movies_features[col] = movies_features[col].astype(float)

movies_features['binary'] = movies_features['binary'].apply(lambda x: 1 if x == "PASS" else 0)
movies_features = movies_features.dropna()

cor = movies_features.corr(method='pearson')
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
corr1, _ = pearsonr(movies_features['budget_2013$'], movies_features['binary'])
corr2, _ = pearsonr(movies_features['budget_2013$'], movies_features['rating'])
corr3, _ = pearsonr(movies_features['rating'], movies_features['binary'])

print("As we can see there is no strong correlation of any feature with the rating (Bechdel score) or the PASS or "
      "FAIL label. There is correlation of {%.2f} between the budget of the movie and the PASS or FAIL label but "
      "its not significant. There is an even lower correlation of {%.2f} between the budget of the movie and rating("
      "score) but its not significant. There is a strong correlation of {%.2f} between the `rating` and the `binary`, "
      "but there both score labels therefore can not be considered as ML features." % (corr1, corr2, corr3))

# Introduce more features

other_movies_features = pd.merge(
  imdb[['imdbid', 'imdbRating', 'imdbVotes']],
  movies_data[['imdbid', 'rating', 'binary']],
  how='right', on='imdbid'
)
other_movies_features['binary'] = other_movies_features['binary'].apply(lambda x: 1 if x == "PASS" else 0)
other_movies_features['bechdel_rating'] = other_movies_features['rating'].astype(float)
del other_movies_features['rating']
other_movies_features = other_movies_features.dropna()

cor = other_movies_features.corr(method='pearson')
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
corr, _ = pearsonr(other_movies_features['imdbVotes'], other_movies_features['binary'])
print(
  "There is correlation of {%.2f} between the imdb Votes of the movie and the PASS or FAIL label but its not significant." % corr)

# task 3 ---------------------------------------------------------------------------------------------------------------

# Create dataset from train including all possible features
movies_features = pd.merge(
  imdb[['imdbid', 'imdbRating', 'imdbVotes']],
  movies_data[['imdbid', 'rating', 'budget_2013$', 'domgross_2013$', 'intgross_2013$']],
  how='right', on='imdbid'
)
movies_features = movies_features.dropna()
movies_features['rating'] = movies_features['rating'].astype(int)
movies_features['budget_2013$'] = movies_features['budget_2013$'].astype(int)
movies_features['domgross_2013$'] = movies_features['domgross_2013$'].astype(int)
movies_features['intgross_2013$'] = movies_features['intgross_2013$'].astype(int)

# Split features and labels
X = movies_features[['imdbRating', 'imdbVotes', 'budget_2013$', 'domgross_2013$', 'intgross_2013$']]
y = movies_features[['rating']]

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=y)

# Create classifiers pipelines
pipe_svm = Pipeline([
  ('sclr', StandardScaler(with_mean=False)),
  ('clf', svm.SVC(random_state=2021))
])

pipe_rf = Pipeline([
  ('sclr', StandardScaler(with_mean=False)),
  ('clf', RandomForestClassifier(random_state=2021))
])

pipe_lr = Pipeline([
  ('sclr', StandardScaler(with_mean=False)),
  ('clf', LogisticRegression(random_state=2021))
])

# Setup GridSearch parameters list
param_range_C = [0.1, 1, 5, 10]
param_range_degree = [3, 4, 5]
grid_params_svm = [{'clf__kernel': ['linear', 'rbf', 'poly'],
                    'clf__degree': param_range_degree,
                    'clf__class_weight': ['balanced', None],
                    'clf__C': param_range_C}]

param_range = [9, 10]
grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],
                   'clf__max_depth': param_range,
                   'clf__min_samples_split': param_range[1:]}]

param_range_fl = [1.0, 0.5]
grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
                   'clf__C': param_range_fl,
                   'clf__solver': ['liblinear']}]

# Run GridSearch
jobs = -1
SVM = GridSearchCV(estimator=pipe_svm,
                   param_grid=grid_params_svm,
                   scoring='accuracy',
                   cv=10,
                   n_jobs=jobs)

LR = GridSearchCV(estimator=pipe_lr,
                  param_grid=grid_params_lr,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=jobs)

RF = GridSearchCV(estimator=pipe_rf,
                  param_grid=grid_params_rf,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=jobs)


# Fit to the GridSearch Models
gs_models = [SVM, LR, RF]
gs_models_dict = {0: 'Support Vector Machine', 1: 'Logistic Regression''Random Forest', 2: 'Logistic Regression'}
print('Performing model optimizations...')
best_acc = 0.0
best_f1 = 0.0
best_clf = 0
best_gs = ''
y_train = y_train.iloc[:, -1:].values.flatten()  # reshape to (n_samples, ) matrix
y_test = y_test.iloc[:, -1:].values.flatten()  # reshape to (n_samples, ) matrix
for idx, gs in enumerate(gs_models):
  print('\nEstimator: %s' % gs_models_dict[idx])
  gs.fit(X_train, y_train)
  print('Best params are : %s' % gs.best_params_)
  print('Best training accuracy: %.3f' % gs.best_score_)
  y_pred = gs.predict(X_test)
  print("*" * 88)
  print('Accuracy score of best params: %.3f ' % accuracy_score(y_test, y_pred))
  print("-" * 88)
  print('F1 score score of best params: %.3f ' % f1_score(y_test, y_pred, average='macro'))
  print("*" * 88)
  if accuracy_score(y_test, y_pred) > best_acc:
    best_acc = accuracy_score(y_test, y_pred)
    best_gs = gs
    best_clf = idx
print('\nClassifier with best accuracy: %s' % gs_models_dict[best_clf])


y_pred = best_gs.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
macro_precision = precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))
macro_precision = precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))
micro_precision = precision_score(y_test, y_pred, average='micro', labels=np.unique(y_pred))
weighted_precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
macro_recall = recall_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))
micro_recall = recall_score(y_test, y_pred, average='micro', labels=np.unique(y_pred))
weighted_recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))
f1_micro = f1_score(y_test, y_pred, average='micro', labels=np.unique(y_pred))
f1_weighted = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))

df_cm = pd.DataFrame(cm)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.show()

df_cm = pd.DataFrame(cm.sum(axis=1))
sn.heatmap(df_cm.round(2), annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g')
plt.title('Sums Matrix')
plt.show()

df_cm = pd.DataFrame(cm.astype('float') / cm.sum(axis=1))
sn.heatmap(df_cm.round(2), annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g')
plt.title('Normalized Confusion Matrix')
plt.show()



print('')
print('Testing samples: %i' % len(y_test))
print('')
print('Metrics')
print('-' * 40)
print('F1 Macro: {0:0.2f}'.format(100 * f1_macro))
print('Accuracy: {0:0.2f}'.format(100 * acc))
print('-' * 40)
print('F1 Micro: {0:0.2f}'.format(100 * f1_micro))
print('F1 Weighted: {0:0.2f}'.format(100 * f1_weighted))
print('Macro Precision: {0:0.2f}'.format(100 * macro_precision))
print('Micro Precision: {0:0.2f}'.format(100 * micro_precision))
print('Weighted Precision: {0:0.2f}'.format(100 * weighted_precision))
print('Macro Recall: {0:0.2f}'.format(100 * macro_recall))
print('Micro Recall: {0:0.2f}'.format(100 * micro_recall))
print('Weighted Recall: {0:0.2f}'.format(100 * weighted_recall))
print('\n')
print('\n')
print('Classification Report')
print('-' * 60)
print(classification_report(y_pred, y_test))
