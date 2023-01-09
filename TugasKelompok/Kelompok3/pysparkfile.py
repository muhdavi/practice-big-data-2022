import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.shell import spark
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

data_iris = load_iris(as_frame=True)
df = pd.DataFrame(data_iris.data, columns = data_iris.feature_names)
spark_df = spark.createDataFrame(df)
spark_df.show(5)

assemble=VectorAssembler(inputCols=[
'sepal length (cm)',
'sepal width (cm)',
'petal length (cm)',
'petal width (cm)'],
outputCol = 'iris_features')
assembled_data=assemble.transform(spark_df)
assembled_data.show(5)

silhouette_scores=[]
evaluator = ClusteringEvaluator(featuresCol='iris_features', metricName='silhouette', distanceMeasure='squaredEuclidean')
for K in range(2,11):
        KMeans_=KMeans(featuresCol='iris_features', k=K)
        KMeans_fit=KMeans_.fit(assembled_data)
        KMeans_transform=KMeans_fit.transform(assembled_data)
        evaluation_score=evaluator.evaluate(KMeans_transform)
        silhouette_scores.append(evaluation_score)

fig, ax = plt.subplots(1,1, figsize=(10,8))
ax.plot(range(2,11), silhouette_scores)
ax.set_xlabel('Jumlah Cluster')
ax.set_ylabel('Hasil Silhouette')
plt.show()

KMeans_=KMeans(featuresCol='iris_features', k=3)
KMeans_Model=KMeans_.fit(assembled_data)
KMeans_Assignments=KMeans_Model.transform(assembled_data)

from pyspark.ml.feature import PCA as PCAml
pca = PCAml(k=2, inputCol="iris_features", outputCol="pca")
pca_model = pca.fit(assembled_data)

pca_transform = pca_model.transform(assembled_data)
x_pca = np.array(pca_transform.rdd.map(lambda row: row.pca).collect())

cluster_assignment = np.array(KMeans_Assignments.rdd.map(lambda row: row.prediction).collect()).reshape(-1,1)
pca_data = np.hstack((x_pca,cluster_assignment))
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal","cluster_assignment"))
sns.FacetGrid(pca_df,hue="cluster_assignment", height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()