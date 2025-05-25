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

# %% [MLOps: MODEL - Linear Regression]
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

# %% [MLOps: MODEL]
ols_model = sm.OLS(y_train, sm.add_constant(X_train_selected)).fit()
print(ols_model.summary())