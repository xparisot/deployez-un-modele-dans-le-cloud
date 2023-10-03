from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, element_at, split, concat_ws, udf
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
import pandas as pd
import numpy as np
import io
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model

# Initialisation de Spark
spark = SparkSession.builder \
    .appName("ImageProcessing") \
    .getOrCreate()
sc = spark.sparkContext

# Chemins pour les données et les résultats
PATH = 's3://oc-p8-xparisot'
PATH_Data = PATH+'/Test'
PATH_Result = PATH+'/Results' 

print('PATH:        '+\
      PATH+'\nPATH_Data:   '+\
      PATH_Data+'\nPATH_Result: '+PATH_Result)

# Charger les images depuis le chemin spécifié
images = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .option("recursiveFileLookup", "true") \
    .load(PATH_Data)
images.show(10)

images = images.withColumn('label', element_at(split(images['path'], '/'),-2))
print(images.printSchema())
print(images.select('path','label').show(10,False))
model = MobileNetV2(weights='imagenet',
                    include_top=True,
                    input_shape=(224, 224, 3))
new_model = Model(inputs=model.input,
                  outputs=model.layers[-2].output)
new_model.summary()
brodcast_weights = sc.broadcast(new_model.get_weights())
def model_fn():
    """
    Returns a MobileNetV2 model with top layer removed 
    and broadcasted pretrained weights.
    """
    model = MobileNetV2(weights='imagenet',
                        include_top=True,
                        input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    new_model = Model(inputs=model.input,
                  outputs=model.layers[-2].output)
    new_model.set_weights(brodcast_weights.value)
    return new_model
def preprocess(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)

def featurize_series(model, content_series):
    """
    Featurize a pd.Series of raw images using the input model.
    :return: a pd.Series of image features
    """
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    '''
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).

    :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
    '''
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches.  This amortizes the overhead of loading big models.
    model = model_fn()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")
features_df = images.repartition(20).select(col("path"),
                                            col("label"),
                                            featurize_udf("content").alias("features")
                                           )
print(PATH_Result)
features_df.write.mode("overwrite").parquet(PATH_Result)
df = pd.read_parquet(PATH_Result, engine='pyarrow')
df.head(10)
df.loc[0,'features'].shape

# Initialisation de Spark
spark = SparkSession.builder.appName("PCA Example").getOrCreate()

# Supposons que vous ayez un DataFrame Spark `features_df` avec une colonne "features" 
# contenant des listes qui doivent être converties en vecteurs denses.

# Convertir la colonne "array" en un vecteur dense
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
features_df = features_df.withColumn("features_vec", list_to_vector_udf("features"))

# Appliquer PCA
pca = PCA(k=2, inputCol="features_vec", outputCol="pcaFeatures")
model = pca.fit(features_df)
transformed = model.transform(features_df)

transformed.show(5)

# Effectuer une PCA complète
n = len(features_df.select("features_vec").first()[0])
pca = PCA(k=n, inputCol="features_vec", outputCol="pcaAllFeatures")
model = pca.fit(features_df)

# Calculer la variance expliquée
explained_var = model.explainedVariance.toArray()

# Calculer la variance cumulée
cumulative_explained_var = np.cumsum(explained_var)

# Créer un DataFrame pour stocker les résultats
results_df = pd.DataFrame({
    'Component Number': range(1, n+1),
    'Explained Variance': explained_var,
    'Cumulative Explained Variance': cumulative_explained_var
})

# Identifier le nombre de composantes nécessaires pour atteindre 90% et 100% de la variance cumulée
num_components_90 = np.where(cumulative_explained_var >= 0.9)[0][0] + 1
num_components_100 = n

#  Ajouter des informations sur le nombre de composantes pour 90% et 100% de la variance cumulée
results_df.loc[num_components_90 - 1, 'Note'] = '90% of variance is explained by this component'
results_df.loc[num_components_100 - 1, 'Note'] = '100% of variance is explained by this component'

#  Afficher le DataFrame avec les résultats
print(results_df)

features_df.select("path", "label", "features_vec")
features_df.show(10)

# Assurez-vous d'avoir une session Spark active
spark = SparkSession.builder.appName("DirectJSONSave").getOrCreate()

# Convertir la colonne "features" en une chaîne
features_df = features_df.withColumn("features_str", concat_ws(",", "features"))

# Écrire le DataFrame au format JSON
output_path_json = "s3://oc-p8-xparisot/original_features.json"
features_df.coalesce(1).write.mode("overwrite").json(output_path_json)

transformed.select("path", "label", "pcaFeatures")
transformed.show(10)

# Convertir la colonne "features" en une chaîne
transformed = transformed.withColumn("features_str", concat_ws(",", "features"))

# Écrire le DataFrame au format JSON
output_path_json = "s3://oc-p8-xparisot/transformed_features.json"
transformed.coalesce(1).write.mode("overwrite").json(output_path_json)

## Chargement des données enregistrées et validation du résultat
df_wk = spark.read.parquet(PATH_Result)
df_wk = df_wk.toPandas()
df_wk.head(10)

spark.stop()