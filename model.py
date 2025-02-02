import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import root_mean_squared_error
import joblib

df = pd.read_csv('cleaned_data.csv')

X_train, X_test, y_train, y_test = train_test_split(df.drop("Performance_Score", axis = 1), df['Performance_Score'], test_size = 0.2)


features = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
}

model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, n_jobs=-1)

# grid_search = GridSearchCV(model, features, cv = 5, n_jobs = -1, verbose = 2, scoring = 'neg_mean_squared_error')

# grid_search.fit(X_train, y_train)

# print(grid_search.best_params_, grid_search.best_score_)

# model = grid_search.best_estimator_
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"rmse: {root_mean_squared_error(y_test, y_pred)}")

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")