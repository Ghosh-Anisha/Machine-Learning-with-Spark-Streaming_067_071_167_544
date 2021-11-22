from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession

sc= SparkContext("local[2]","sent")
spark = SparkSession.builder.appName("Sentiment").getOrCreate()
ssc = StreamingContext(sc, 1)

lines = ssc.socketTextStream('localhost', 6100)

def temp(rdd):
	df=spark.read.json(rdd)
	for row in df.rdd.toLocalIterator():
		for i in range(1000):
			print(row[str(i)]['feature0'],row[str(i)]['feature1'])
	#df.show()
lines.foreachRDD(lambda rdd : temp(rdd))

ssc.start()
ssc.awaitTermination()

