import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from pyspark import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml import Pipeline
import numpy as np
from pyspark.sql import functions as fn
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when
from pyspark.sql.functions import coalesce, lit
import requests
from pyspark.ml.feature import StopWordsRemover

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
review = sqlContext.read.json("./dataset/review.json")
#review = review.withColumn("user_id", review.user_id.cast('string')).withColumn("business_id", review.business_id.cast('string')).withColumn("stars", review.stars.cast('float'))
#sentiment = coalesce((col("stars") >= 3.0).cast("int"), lit(1))
#review = review.withColumn("sentiment", sentiment)
#review = review.filter(review.user_id.isNotNull()).filter(review.business_id.isNotNull())

sentiments = sqlContext.sql('SELECT *, case when stars <= 2.5 then 0 when stars >= 3.5 then 1 end as sentiment from review where stars<=2.5 or stars>= 3.5')

df1 = sentiments[['review_id','text','sentiment']]
#df1.show(1)
#df1.count()

#print(df1)
df2 = df1.rdd
#df2.take(1)

sid = SentimentIntensityAnalyzer()

#calculate the compound score for the review text and then use that to derive the sentiment.
#Then use this predicted sentiment and compare it with the actual sentiment to calculate accuracy.
df3 = df2.map(lambda x : (x[0],sid.polarity_scores(x[1])['compound'],x[2])).\
        map(lambda y: (y[0],1,y[2]) if y[1]>0 else (y[0],0,y[2])).\
        map(lambda z: 1 if(z[1] == z[2]) else 0)
#df3.count()
#df3.take(2)
total_count = df3.count()
accuracy = (df3.filter(lambda a: a==1).count()*1.0)/total_count
print(accuracy)