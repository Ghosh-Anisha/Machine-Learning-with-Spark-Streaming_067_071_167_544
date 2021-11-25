from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.types as tp
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import Row

sc= SparkContext("local[2]","sent")
spark = SparkSession.builder.appName("Sentiment").getOrCreate()
ssc = StreamingContext(sc, 1)

lines = ssc.socketTextStream('localhost', 6100)

columns=["score","tweet"]

def preprocessing(df):
	stage_1 = RegexTokenizer(inputCol= 'tweet' , outputCol= 'tokens', pattern= '\\W')
	stage_2 = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
	stage_3 = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 100)
	model = LogisticRegression(featuresCol= 'vector', labelCol= 'score')
	pipeline = Pipeline(stages= [stage_1, stage_2, stage_3,model])

	pipelineFit = pipeline.fit(df)
	df1=pipelineFit.transform(df)
	df1.show()
	accuracy = df1.filter(df.score == df1.prediction).count() / df1.count()
	print(accuracy)
	
def temp(rdd):
	df=spark.read.json(rdd)
	for row in df.rdd.toLocalIterator():
		conv_df = spark.createDataFrame(row,columns)
		#conv_df.show()
		preprocessing(conv_df)
			
lines.foreachRDD(lambda rdd : temp(rdd))

ssc.start()
ssc.awaitTermination()


