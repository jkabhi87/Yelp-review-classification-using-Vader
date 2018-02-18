from pyspark import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
import numpy as np
from pyspark.sql import functions as fn
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when
from pyspark.sql.functions import coalesce, lit
from pyspark.ml.feature import RegexTokenizer
import requests

review = spark.read.json("./dataset/review.json")
review = review.withColumn("user_id", review.user_id.cast('string')).withColumn("business_id", review.business_id.cast('string')).withColumn("stars", review.stars.cast('float'))
sentiment = coalesce((col("stars") >= 3.0).cast("int"), lit(1))
review = review.withColumn("sentiment", sentiment)
review = review.filter(review.user_id.isNotNull()).filter(review.business_id.isNotNull())

#review.show(10)
stop_words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words').text.split()
#stop_words[0:10]

tokenizer = RegexTokenizer().setGaps(False)\
  .setPattern("\\p{L}+")\
  .setInputCol("text")\
  .setOutputCol("words")

from pyspark.ml.feature import StopWordsRemover
sw_filter = StopWordsRemover()\
  .setStopWords(stop_words)\
  .setCaseSensitive(False)\
  .setInputCol("words")\
  .setOutputCol("filtered")

from pyspark.ml.feature import CountVectorizer

# we will remove words that appear in 5 docs or less
cv = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17)\
  .setInputCol("filtered")\
  .setOutputCol("tf")

# we now create a pipelined transformer
cv_pipeline = Pipeline(stages=[tokenizer, sw_filter, cv]).fit(review)
cv_pipeline.transform(review).show(5)

from pyspark.ml.feature import IDF
idf = IDF().\
    setInputCol('tf').\
    setOutputCol('tfidf')

idf_pipeline = Pipeline(stages=[cv_pipeline, idf]).fit(review)

tfidf_df = idf_pipeline.transform(review)

tfidf_df.show(10)
#training_df, validation_df, testing_df = review.randomSplit([0.6, 0.3, 0.1], seed=0)



#training_df, validation_df, testing_df = review.randomSplit([0.6, 0.3, 0.1], seed=0)
#[training_df.count(), validation_df.count(), testing_df.count()]

import pandas as pd

training_df, validation_df, testing_df = review.randomSplit([0.6, 0.3, 0.1], seed=0)
[training_df.count(), validation_df.count(), testing_df.count()]

lambda_par = 0.02
alpha_par = 0.3
en_lr = LogisticRegression().\
        setLabelCol('sentiment').\
        setFeaturesCol('tfidf').\
        setRegParam(lambda_par).\
        setMaxIter(100).\
        setElasticNetParam(alpha_par)

en_lr_pipeline = Pipeline(stages=[idf_pipeline, en_lr]).fit(review)
en_lr_pipeline.transform(review).select(fn.avg(fn.expr('float(prediction = sentiment)'))).show()

en_weights = en_lr_pipeline.stages[-1].coefficients.toArray()
en_coeffs_df = pd.DataFrame({'word': vocabulary, 'weight': en_weights})

#en_coeffs_df.sort_values('weight').head(15)
#en_coeffs_df.sort_values('weight', ascending=False).head(15)
#en_coeffs_df.query('weight == 0.0').shape
en_coeffs_df.query('weight == 0.0').shape[0]/en_coeffs_df.shape[0]

from pyspark.ml.tuning import ParamGridBuilder

en_lr_estimator = Pipeline(stages=[idf_pipeline, en_lr])

grid = ParamGridBuilder().\
    addGrid(en_lr.regParam, [0., 0.01, 0.02]).\
    addGrid(en_lr.elasticNetParam, [0., 0.2, 0.4]).\
    build()

#grid

all_models = []
for j in range(len(grid)):
    print("Fitting model {}".format(j+1))
    model = en_lr_estimator.fit(training_df, grid[j])
    all_models.append(model)
    
import numpy as np

# estimate the accuracy of each of them:
accuracies = [m.\
    transform(validation_df).\
    select(fn.avg(fn.expr('float(sentiment = prediction)')).alias('accuracy')).\
    first().\
    accuracy for m in all_models]

best_model_idx = np.argmax(accuracies)
grid[best_model_idx]

best_model = all_models[best_model_idx]
accuracies[best_model_idx]