import pandas as pd
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

df = pd.read_csv('archive/segmentation data.csv')
# print(df.info())

# EDA
df_copy = df.copy()
col = ['Sex', 'Marital status', 'Education', 'Occupation', 'Settlement size']
num = ['Age', 'Income']

df_copy.Sex.replace({0:"Male", 1:'Female'}, inplace=True)
df_copy.Education.replace({0:'other / unknown', 1:'high school', 2:'university', 3:'graduate school'}, inplace=True)
df_copy['Marital status'].replace({0:'single', 1:'non-single'}, inplace=True)
df_copy.Occupation.replace({0:'unemployed', 1:'official', 2: 'management / self-employed'}, inplace=True)
df_copy['Settlement size'].replace({0:'small city', 1:'mid-sized city', 2:'big city'}, inplace=True)

# fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,20))
# i,j = 0, 0
# k = 0
# for i in range(2):
#     for j in range(3):
#         df_ = df_copy.groupby(by=col[k]).count()
#         ax[i, j].title.set_text('Comparison of '+col[k])
#         ax[i, j].pie(data=df_, labels=df_.index,x = 'ID', autopct='%.2f', explode=[0.03]*len(df_.index))
#         j += 1
#         k += 1
#         if k == len(col):
#             break
#     i += 1
# plt.tight_layout()
# df_ = df_copy.groupby(by=['Sex', 'Settlement size']).count().reset_index()
# sns.catplot(data = df_, col='Settlement size', x='Sex', y='ID', kind='bar')
# plt.show()


#one hot encoding
# print(df_copy.info())
df_=pd.get_dummies(data=df_copy, columns=['Sex', 'Marital status', 'Education', 'Occupation', 'Settlement size'])
df_.to_csv('one_hot.csv')
data = pd.read_csv('one_hot.csv')
data.drop(['Unnamed: 0', 'ID'], axis='columns', inplace=True)
# print(df_.columns)
# print(df_.head(5))



#KMeans Clustering

inertia = []
silhouette_scores = []
list_num_clusters = list(range(2,11))


for num_clusters in list_num_clusters:
    km = KMeans(n_clusters=num_clusters)
    km.fit(data)
    labels = km.predict(data)
    silhouette = silhouette_score(df_, labels)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette)

fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].plot(inertia)
axs[0].set_xlabel('K Value')
axs[0].set_ylabel('Inertia')

axs[1].plot(silhouette_scores)
axs[1].set_xlabel('K Value')
axs[1].set_ylabel('Silhouette Scores')
plt.show()

kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,10))
visualizer.fit(data)
visualizer.show()

km = KMeans(n_clusters=4, random_state=777)
km.fit(data)
preds = kmeans.predict(data)
print(silhouette_score(data, preds))


# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters = 4)
agg_preds = agg.fit_predict(data)
data_copy = data.copy()
data_copy['agg_class'] = agg_preds
print(silhouette_score(data_copy, agg_preds))

# Mini Batch K Means Algorithm
mbk = MiniBatchKMeans(n_clusters=4, batch_size=45)

#Hierarchical Clustering
hc_complete = linkage(data, 'complete')
plt.figure(figsize=(15,10))
plt.title('Hierarchical Clustering')
dendrogram(hc_complete, leaf_font_size=10)
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_pred = cluster.fit(data)
data_ = data.copy()
data_['Hie Class'] = y_pred
print(silhouette_score(data_, y_pred))
