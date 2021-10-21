import pandas as pd
from utilities import request_json_data, clean_text
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: move all to jupyter notebook

actors_gender_data = pd.read_csv("https://raw.githubusercontent.com/taubergm/HollywoodGenderData/master/all_actors_movies_gender_gold.csv")
writers_gender_data = pd.read_csv("https://raw.githubusercontent.com/taubergm/HollywoodGenderData/master/all_writers_gender.csv")


# task 1
all_movies = pd.DataFrame(request_json_data("https://bechdeltest.com/api/v1/getAllMovies"))

movies_data = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/bechdel/movies.csv")

# 2.
# There are artifacts in the text
print(movies_data.iloc[0]['title'])

# We deal with the artifacts and strip and accent todo: do more cleaning if there is any
movies_data['title'] = movies_data['title'].apply(lambda x: clean_text(x))

# 3.
imdb_data = pd.DataFrame(request_json_data("https://raw.githubusercontent.com/brianckeegan/Bechdel/master/imdb_data.json"))
merged_movies_data = pd.merge(movies_data, imdb_data, how='inner', left_on='imdb', right_on='imdbID')

# df =
#Using Pearson Correlation

cor = merged_movies_data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
