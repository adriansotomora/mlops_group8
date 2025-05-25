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