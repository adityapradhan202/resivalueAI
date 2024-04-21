import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
from joblib import dump

df = pd.read_csv('house_price_pracdata.csv')

y = df['Price']
X = df.drop('Price', axis=1)

rmse_test_values = []
rmse_train_values = []
for d in range(1,6):
    poly_transformer = PolynomialFeatures(degree=d, include_bias=False)
    poly_features = poly_transformer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.30, random_state=89)
    
    test_model = LinearRegression(fit_intercept=True)
    test_model.fit(X_train, y_train)
    test_model_predictions_train = test_model.predict(X_train)
    test_model_predictions_test = test_model.predict(X_test)

    rmse_train = np.sqrt(mse(y_train, test_model_predictions_train))
    rmse_test = np.sqrt(mse(y_test, test_model_predictions_test))
    rmse_train_values.append(rmse_train)
    rmse_test_values.append(rmse_test)

plt.plot(range(1,6), rmse_train_values, color='orange', label='Erros on training data')
plt.plot(range(1,6), rmse_test_values, color='blue', ls='--', label='Errors on test data')
plt.legend(loc='upper left')
# plt.show()

chosen_poly_degree = 3
poly_converter = PolynomialFeatures(degree=chosen_poly_degree, include_bias=False)
pol_features = poly_converter.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(pol_features, y, test_size=0.30, random_state=89)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


elastic_cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], eps=0.001, n_alphas=100, max_iter=10000, cv=10)
elastic_cv_model.fit(X_train, y_train)
ecv_model_predictions = elastic_cv_model.predict(X_test)
ecv_model_rmse = np.sqrt(mse(y_test, ecv_model_predictions))

mae_values = cross_val_score(estimator=elastic_cv_model, X=X_train, y=y_train, scoring='neg_mean_absolute_error', cv=10)
final_mae_value = np.mean(mae_values)

best_l1_ratio = elastic_cv_model.l1_ratio_
optimum_alpha = elastic_cv_model.alpha_

print(f'RMSE value = {ecv_model_rmse}')
print(f'MAE value after Cross validating = {final_mae_value}')

final_model = ElasticNet(l1_ratio=best_l1_ratio, alpha=best_l1_ratio)
final_poly_converter = PolynomialFeatures(degree=3, include_bias=False)
full_transformed_X = final_poly_converter.fit_transform(X)
final_model.fit(full_transformed_X,y)

dump(final_poly_converter, 'fpc.joblib')
dump(final_model, 'fmodel.joblib')