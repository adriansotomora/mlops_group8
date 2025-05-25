# %% [markdown]
# ## Machine Learning 1 Group Project. 
# ### Predicting the popularity on songs

# %% [markdown]
# ### Libraries Import

# %% [MLOps: GENERAL SETUP]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from collections import Counter

# %% [MLOps: DATA LOADING]
df = pd.read_csv("../data/raw/Songs_2025.csv")
df.info()

# %% [MLOps: DATA LOADING]
df.sample(6).to_csv("../tests/mock_data/mock_data_spotify.csv", index=False)

# %% [MLOps: DATA VALIDATION]
df.describe()

# %% [MLOps: DATA VALIDATION]
df.isna().describe()

# %% [MLOps: DATA VALIDATION]
df = df.dropna()
df.isnull().sum()

# %% [MLOps: DATA VALIDATION / EXPLORATORY ANALYSIS]
numeric_cols = df.select_dtypes(include=[np.number]).columns

# %% [MLOps: PREPROCESS]
def remove_outliers_iqr(df, features):
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  
        upper_bound = Q3 + 3 * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

# [MLOps: PREPROCESS]
df.drop(columns=['instrumentalness'], inplace=True)
df = remove_outliers_iqr(df, ['duration_ms'])

# %% [MLOps: PREPROCESS]
columns_to_scale = ['danceability', 'energy', 'loudness', 'speechiness',
                     'acousticness', 'liveness', 'valence', 'tempo', 'duration_ms', 'key']
scaler = MinMaxScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

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

# %% [MLOps: MODEL]
# KMeans clustering and elbow method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# %% [MLOps: MODEL]
k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
df['cluster'] = kmeans.predict(X)
cluster = df['cluster']

# %% [MLOps: EVALUATION]
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, df['cluster'])
centroids = np.array([X[cluster == i].mean(axis=0) for i in range(k)])
inertia = sum(
    np.sum(np.linalg.norm(X[cluster == i] - centroids[i], axis=1)**2)
    for i in range(k)
)

# %% [MLOps: EVALUATION]
clustersizes = df.groupby('cluster').agg({
    'track_name': 'count',
    'track_popularity': 'mean',
    'artist_popularity': 'mean',
})

# %% [MLOps: EVALUATION]
results = []
for column in X.columns:
    groups = [group[column].values for name, group in df.groupby('cluster')]
    f_stat, p_value = f_oneway(*groups)
    avg = df[column].mean()
    results.append({
        'Column': column,
        'F-statistic': f_stat,
        'p-value': p_value, 
        'Column mean': avg
    })
anova_results_df = pd.DataFrame(results)
anova_results_df.sort_values("p-value")

# %% [MLOps: EVALUATION]
df_prof = df.groupby("cluster")[X.columns].mean().transpose()
df_prof["Average"] = df_prof.mean(axis=1)
df_prof["Std.dev"] = df_prof.std(axis=1)


# %% [MLOps: MODEL]
# Linear Regression setup
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# %% [MLOps: FEATURES]
irrelevant_columns = ['track_name', 'album', 'artist_name', 'cluster', 'artist_genres', 'year']
df = df.drop(columns=irrelevant_columns)

# %% [MLOps: PREPROCESS]
def remove_outliers_iqr(df, features):
    df_clean = df.copy()
    for feature in features:
        Q1 = df_clean[feature].quantile(0.25)
        Q3 = df_clean[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR  
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[feature] >= lower_bound) & (df_clean[feature] <= upper_bound)]
    return df
df = remove_outliers_iqr(df, ['duration_ms'])
df = remove_outliers_iqr(df, ['track_popularity'])

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

# %% [MLOps: DATA VALIDATION / EVALUATION]
sns.histplot(data=df, x=y, kde=False)
plt.xlabel('Column Name')
plt.ylabel('Frequency')
plt.title('Histogram of Column Name')
print(y.describe())
plt.show()

# %% [MLOps: MODEL]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print ("Sample size train dataset: ", X_train.shape)
print ("Sample size test dataset: ", X_test.shape)

# %% [MLOps: MODEL]
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.05, 
                       threshold_out = 0.1, 
                       verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[included+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.4}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            print(included, worst_feature, pvalues)
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.4}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

# %% [MLOps: MODEL]
selected_features = stepwise_selection(X_train, y_train)
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# %% [MLOps: EVALUATION]
sns.pairplot(X_train_selected)
print("X_train_selected shape:", X_train_selected.shape)
print("X_test_selected shape:", X_test_selected.shape)

# %% [MLOps: EVALUATION]
sns.set(font_scale=1.1)
fig, ax = plt.subplots(figsize=(10,10))         
sns.heatmap(X_train_selected.corr(), annot=True, fmt=".2f", linewidths=1, ax=ax)

# %% [MLOps: MODEL]
ols_model = sm.OLS(y_train, sm.add_constant(X_train_selected)).fit()
print(ols_model.summary())

# %% [MLOps: INFERENCE]
y_pred = ols_model.predict(sm.add_constant(X_test_selected))

# %% [MLOps: INFERENCE]
coeff_df = pd.DataFrame(ols_model.params,X_train_selected.columns,columns=['Coefficient'])
coeff_df

# %% [MLOps: INFERENCE]
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# %% [MLOps: INFERENCE]
plt.title("Real vs. Fitted (train dataset)")
plt.scatter(y_test,y_pred)
coef = np.polyfit(y_test, y_pred, 1)  
poly1d_fn = np.poly1d(coef)  
plt.plot(y, poly1d_fn(y), color="red", label="Regression line")
plt.xlabel("Real")
plt.ylabel("Fitted")
plt.legend()
plt.show()
