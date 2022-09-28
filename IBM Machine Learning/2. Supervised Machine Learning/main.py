import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.pipeline import make_pipeline


df = pd.read_csv('insurance.csv')
df_copy = df.copy()
col = ['sex', 'smoker', 'region']

# print(df.smoker.unique())
# print(df.region.unique())



#df_ = df_copy.groupby(by=['region', 'smoker']).count().reset_index()
# sns.catplot(data=df_, col='smoker', x='region', y='age', kind='bar')
# plt.show()

#산점도 이따 맨 마지막에 한번 정리해서 넣어두자.
fig = plt.figure(figsize=(15,5))
sns.scatterplot(df['charges'], df['age'], hue=df['sex'], style=df['smoker'])

fig = plt.figure(figsize=(15,5))
sns.scatterplot(df['charges'], df['bmi'], hue=df['sex'], style=df['smoker'])
plt.grid(True)
plt.show()

# new_df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
# x = new_df.drop(columns=['charges'])
# y = new_df['charges']
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.35, random_state=12)
#
# lr=LinearRegression()
# lr.fit(x_train,y_train)
# y_pred = lr.predict(x_test)
# print(r2_score(y_test, y_pred))

# best_degree = [2,3,4,5,6]
# for i in best_degree:
#     pol = PolynomialFeatures(degree = i)
#     x_pol = pol.fit_transform(x)
#     x_train, x_test, y_train, y_test = train_test_split(x_pol, y, test_size=0.35, random_state=123)
#     plr = LinearRegression().fit(x_train, y_train)
#     y_pred = plr.predict(x_test)
#     print('degree: ',i,' ''s Accuracy: ', r2_score(y_test, y_pred))

#with Ridge
# alphas = [0.1, 0.05 ,0.001, 0.0001]
# ratios = [0.1,0.3,0.5,0.7,0.9]
#
# new_df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
# x = new_df.drop(columns=['charges'])
# y = new_df['charges']
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.35, random_state=12)
#
# for i in alphas:
#     model_Lasso = Lasso(alpha=i)
#     model_Ridge = Ridge(alpha=i)
#
#     model_Lasso.fit(x_train, y_train)
#     model_Ridge.fit(x_train, y_train)
#
#     y_pred_lasso = model_Lasso.predict(x_test)
#     y_pred_ridge = model_Ridge.predict(x_test)
#
#     print('Lasso alpha:', i, 'r2_score: ', r2_score(y_test, y_pred_lasso))
#     print('Ridge alpha:', i, 'r2_score: ', r2_score(y_test, y_pred_ridge))
#
#     for j in ratios:
#         model_elastic = ElasticNet(alpha=i, l1_ratio=j)
#         model_elastic.fit(x_train, y_train)
#         y_pred_elastic = model_elastic.predict(x_test)
#         print('Elastic alpha:', i, 'ratio: ',j ,'r2_score: ', r2_score(y_test, y_pred_elastic))
#     print('')
#
#
# alphas = [0.1, 0.05, 0.001, 0.001]
# ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
# x_pol = pol.fit_transform(x)
# x_train, x_test, y_train, y_test = train_test_split(x_pol, y, test_size=0.35, random_state=123)
# for i in alphas:
#     model_Lasso_pol = make_pipeline(PolynomialFeatures(2), Lasso(alpha=i))
#     model_Ridge_pol = make_pipeline(PolynomialFeatures(2), Ridge(alpha=i))
#
#     model_Lasso_pol.fit(x_train, y_train)
#     model_Ridge_pol.fit(x_train, y_train)
#
#     y_pred_lasso = model_Lasso_pol.predict(x_test)
#     y_pred_ridge = model_Ridge_pol.predict(x_test)
#
#     print('Lasso_Pol alpha:', i, 'r2_score: ', r2_score(y_test, y_pred_lasso))
#     print('Ridge_Pol alpha:', i, 'r2_score: ', r2_score(y_test, y_pred_ridge))
#
#     for j in ratios:
#         model_elastic_pol = make_pipeline(PolynomialFeatures(2), ElasticNet(alpha=i, l1_ratio=j))
#         model_elastic_pol.fit(x_train, y_train)
#         y_pred_elastic = model_elastic_pol.predict(x_test)
#         print('Elastic_Pol alpha:', i, 'ratio: ',j,'r2_score: ', r2_score(y_test, y_pred_elastic))
#     print('')






