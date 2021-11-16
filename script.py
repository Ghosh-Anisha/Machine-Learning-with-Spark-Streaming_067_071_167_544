from pyspark import SparkContext
from pyspark.sql import *
from pyspark.streaming import StreamingContext

spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    
lines = spark \
    .readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 6100) \
    .load()
    
lines.printSchema()


