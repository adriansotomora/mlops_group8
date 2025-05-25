# %% [MLOps: FEATURES]
# Genre parsing and mapping
artist_genres = df["artist_genres"].apply(lambda x: x.strip("[]").replace("'", "").split("; "))

all_genres = [genre for sublist in artist_genres for genre in sublist]
top_genres = [genre for genre, count in Counter(all_genres).most_common(20)]

genres_map = {
    'pop': r'(pop)',
    'rock': r'(rock)',
    'electronic': r'(house|edm|electro|progressive)',
    'latin': r'(latin|puerto|reggaeton)',
    'hip-hop': r'(hip|rap|urban)',
    'indie':r'(indie)',
    'jazz':r'(jazz)',
    'r&b':r'(r&b)',
    'soul':r'(soul)',
    'metal':r'(metal|punk)',
    'classic':r'(classic)',
    'country':r'(country)'
}
import re
for genre, pattern in genres_map.items():
    df[genre] = df['artist_genres'].str.contains(pattern, flags=re.IGNORECASE, regex=True).astype(int)

# %% [MLOps: FEATURES]
# Example inspection
pd.set_option('display.max_colwidth', None)

# %% [MLOps: FEATURES / MODEL]
exclude = ['year']
profiling_variables = ['track_popularity', 'artist_popularity', 'energy', 'mode']
X = df.select_dtypes(include=['number']).drop(columns=exclude).drop(columns=profiling_variables)

# %% [MLOps: FEATURES]
corr_matrix = X.corr()


# %% [MLOps: FEATURES - Linear Regression]
irrelevant_columns = ['track_name', 'album', 'artist_name', 'cluster', 'artist_genres', 'year']
df = df.drop(columns=irrelevant_columns)

# %% [MLOps: FEATURES]
audio_features = ['danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'duration_ms', 'artist_popularity']
genre_features = ['pop', 'rock', 'electronic', 'latin', 'hip-hop', 'indie', 'jazz', 'r&b', 'soul', 'metal', 'classic', 'country']

# %% [MLOps: FEATURES]
poly = PolynomialFeatures(degree=2, include_bias=False)
audio_poly = poly.fit_transform(df[audio_features])
genre_poly = poly.fit_transform(df[genre_features])

# %% [MLOps: FEATURES]
x = df.drop(columns=['track_popularity'])
y = df['track_popularity']