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
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import BernoulliNB


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

sc= SparkContext("local[2]","sent")
spark = SparkSession.builder.appName("Sentiment").getOrCreate()
ssc = StreamingContext(sc, 1)

lines = ssc.socketTextStream('localhost', 6100)
columns=["score","tweet"]

clf = BernoulliNB()
def preprocessing(df):
	df_temp=np.array(df.select('tweet').collect())
	df_n= list()
	stemmer = WordNetLemmatizer()
	for sen in range(0, len(df_temp)):
	    df_temp1 = re.sub(r'\W', ' ', str(df_temp[sen])) 
	    df_temp1 = re.sub(r'\s+[a-zA-Z]\s+', ' ', df_temp1) 
	    df_temp1 = re.sub(r'\^[a-zA-Z]\s+', ' ', df_temp1)  
	    df_temp1 = re.sub(r'\s+', ' ', df_temp1, flags=re.I) 
	    df_temp1 = re.sub(r'^b\s+', '', df_temp1)
	    df_temp1 = df_temp1.lower()
	    df_temp1 = df_temp1.split()
	    df_temp1 = [stemmer.lemmatize(word) for word in df_temp1]
	    df_temp1 = ' '.join(df_temp1)
	    df_n.append(df_temp1)
	
	#count_vect = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
	#X_train_counts = count_vect.fit_transform(df_n)
	
	#tfidfconverter = TfidfTransformer()
	#tf_transformer= tfidfconverter.fit_transform(X_train_counts)
	
	#tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
	#X = tfidfconverter.fit_transform(df_n)
	
	X=np.array(df_n)
	hashvect=HashingVectorizer(stop_words=stopwords.words('english'),alternate_sign=False)
	X_f=hashvect.fit_transform(X)

	y=np.reshape(np.array(df.select('score').collect()),(10000,1))
	
	X_train, X_test, y_train, y_test = train_test_split(X_f.reshape(10000,-1), y.reshape(-1,1), test_size=0.33, random_state=42)
	
	try:
		clf= pickle.load(open('model.pkl','rb'))
		print(0000)
	except:
		clf = BernoulliNB()
		
	model1=clf
	df2=model1.partial_fit(X_train,y_train,classes=[0,4])
	pickle.dump(df2,open('model.pkl','wb'))
	y_pred=df2.predict(X_test)
	accuracy=sklearn.metrics.accuracy_score(y_test,y_pred)	
	print(accuracy)

def temp(rdd):
	df=spark.read.json(rdd)
	#count=0
	for row in df.rdd.toLocalIterator():
		conv_df = spark.createDataFrame(row,columns)
		#conv_df.show()
		preprocessing(conv_df)
		#count+=1

			
lines.foreachRDD(lambda rdd : temp(rdd))

ssc.start()
ssc.awaitTermination()


