import pandas as pd
from utilities import request_json_data
import matplotlib.pyplot as plt

# TODO: move all to jupyter notebook

all_movies = pd.DataFrame(request_json_data("https://bechdeltest.com/api/v1/getAllMovies"))
imdb_data = pd.DataFrame(request_json_data("https://raw.githubusercontent.com/brianckeegan/Bechdel/master/imdb_data.json"))
movies_data = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/bechdel/movies.csv")


# task 1
all_movies.loc[all_movies['year'] < 2022, ['year', 'rating']].groupby(['year']).mean().rename(columns={'rating': 'avg rating'}).plot()
plt.title('Average Bechdel Score Per Year')
plt.show()

# todo: could be done using only pandas (in one line)
balanced_years = []
for year in sorted(set(all_movies['year'].values.tolist())):
  num_over_two = len(all_movies[(all_movies['year'] == year) & (all_movies['rating'] >= 2)])
  num_all = len(all_movies[all_movies['year'] == year])
  if num_over_two / num_all == 0.5:
    balanced_years.append(year)

print(f"The years where 50% of movies have a bechdel score of 2 or higher are: {', '.join(str(y) for y in balanced_years)}")

