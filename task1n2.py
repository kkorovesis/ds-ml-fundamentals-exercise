import pandas as pd
from utilities import request_json_data, clean_text
import matplotlib.pyplot as plt
import seaborn as sns


# EXTRA data ----------------------------------------------------------------------
actors_gender_data = pd.read_csv("https://raw.githubusercontent.com/taubergm/HollywoodGenderData/master/all_actors_movies_gender_gold.csv")
writers_gender_data = pd.read_csv("https://raw.githubusercontent.com/taubergm/HollywoodGenderData/master/all_writers_gender.csv")
# all_movies = pd.DataFrame(request_json_data("https://bechdeltest.com/api/v1/getAllMovies"))
# imdb = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-03-09/movies.csv')
# ---------------------------------------------------------------------------------


# TODO: move all to jupyter notebook

bechdel_data = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-03-09/raw_bechdel.csv", dtype=str)

# task 1 ---------------------------------------------------------------------------------------------------------------
# 1.
bechdel_data.loc[bechdel_data['year'] < 2022, ['year', 'rating']].groupby(['year']).mean().rename(columns={'rating': 'avg rating'}).plot()
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


# To get more data on the movies we join with movies dataset that includes 1794 movies
movies = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/bechdel/movies.csv", dtype=str)

# Merge datasets to create a new dataset
movies['imdbid'] = movies['imdb'].apply(lambda x: x.split('tt')[1])  # get the proper imdb id format
movies['year'] = movies['year'].astype('object')
movies_data = pd.merge(bechdel_data[['imdb_id', 'rating']], movies, how='right', left_on=['imdb_id'], right_on=['imdbid'])

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
'''
Index(['year', 'imdb', 'title', 'test', 'clean_test', 'binary', 'budget',
       'domgross', 'intgross', 'code', 'budget_2013$', 'domgross_2013$',
       'intgross_2013$', 'period code', 'decade code', 'imdbid'],
      dtype='object')
'''

# Test correlation on current features

dataset_labels = movies_data[['imdbid', 'title', 'test', 'clean_test', 'binary', 'rating']]

df = movies_data[['year', 'budget', 'domgross', 'intgross', 'budget_2013$', 'domgross_2013$', 'intgross_2013$', 'rating', 'binary']]
for col in ['budget', 'domgross', 'intgross', 'budget_2013$', 'domgross_2013$', 'intgross_2013$', 'rating']:
  df[col] = df[col].astype(float)

df['binary'] = df['binary'].apply(lambda x: 1 if x == "PASS" else 0)

cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Introduce more features

imdb = pd.DataFrame(request_json_data("https://raw.githubusercontent.com/brianckeegan/Bechdel/master/imdb_data.json"))
imdb = imdb.dropna(subset=['imdbID'])
imdb = imdb[imdb['imdbRating'] != 'N/A']
imdb = imdb[imdb['imdbVotes'] != 'N/A']
imdb['imdbid'] = imdb['imdbID'].apply(lambda x: x.split('tt')[1])  # get the proper imdb id format
imdb['imdbRating'] = imdb['imdbRating'].astype(float)
imdb['imdbVotes'] = imdb['imdbVotes'].apply(lambda x: float(x.replace(',', '')))
other_movies_features = pd.merge(
  imdb[['imdbid', 'imdbRating', 'imdbVotes']],
  movies_data[['imdbid', 'rating', 'binary']],
  how='right', on='imdbid'
)
other_movies_features['binary'] = other_movies_features['binary'].apply(lambda x: 1 if x == "PASS" else 0)
other_movies_features['bechdel_rating'] = other_movies_features['rating'].astype(float)
del other_movies_features['rating']


cor = other_movies_features.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


