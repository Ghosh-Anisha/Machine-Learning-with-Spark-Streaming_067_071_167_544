from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.types as tp
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import Row
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import pickle

sc= SparkContext("local[2]","sent")
spark = SparkSession.builder.appName("Sentiment").getOrCreate()
ssc = StreamingContext(sc, 1)

lines = ssc.socketTextStream('localhost', 6100)

columns=["score","tweet"]
model = SGDClassifier(alpha=.0001, loss='log', penalty='l2', n_jobs=-1, shuffle=True)
def preprocessing(df):
	df_temp=np.array(df.select('tweet').collect())
	
	"""
	stage_1 = RegexTokenizer(inputCol= 'tweet' , outputCol= 'tokens', pattern= '\\W')
	stage_2 = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
	stage_3 = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 100)
	model = LogisticRegression(featuresCol= 'vector', labelCol= 'score')
	pipeline = Pipeline(stages= [stage_1, stage_2, stage_3])
	pipelineFit = pipeline.fit(df)
	df1=pipelineFit.transform(df)
	df1.show()
	print(df1["vector"])
	"""
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(df_temp.ravel())
	tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
	X_train_tf = tf_transformer.transform(X_train_counts)
	y=np.reshape(np.array(df.select('score').collect()),(10000,1))
	batchsize=10000
	
	X_train, X_test, y_train, y_test = train_test_split(X_train_tf, y, test_size=0.33, random_state=42)
	#df2=model.partial_fit(np.reshape(np.array(df1.select('vector').collect()),(10000,100)),np.reshape(np.array(df1.select('score').collect()),(10000,1)),classes=[0,4])


	df2=model.partial_fit(X_train,y_train,classes=[0,4])
	pickle.dump(df2,open('model.pkl','wb'))
	model = pickle.load(open('model.pkl','rb'))
	y_pred=df2.predict(X_test)
	accuracy=sklearn.metrics.accuracy_score(y_test,y_pred)	
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


