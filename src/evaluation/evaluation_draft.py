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


# %% [MLOps: EVALUATION - Linear Regression]
sns.pairplot(X_train_selected)
print("X_train_selected shape:", X_train_selected.shape)
print("X_test_selected shape:", X_test_selected.shape)

# %% [MLOps: EVALUATION]
sns.set(font_scale=1.1)
fig, ax = plt.subplots(figsize=(10,10))         
sns.heatmap(X_train_selected.corr(), annot=True, fmt=".2f", linewidths=1, ax=ax)
