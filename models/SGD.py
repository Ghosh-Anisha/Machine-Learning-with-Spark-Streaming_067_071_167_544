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
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
#import warnings

nltk.download('stopwords')
nltk.download('wordnet')

sc= SparkContext("local[2]","sent")
spark = SparkSession.builder.appName("Sentiment").getOrCreate()
ssc = StreamingContext(sc, 1)

lines = ssc.socketTextStream('localhost', 6100)
columns=["score","tweet"]

model = SGDClassifier(alpha=.0001, loss='log', penalty='l2', n_jobs=-1, shuffle=True)
#warnings.filterwarnings("ignore")
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
	
	
	hashvect=HashingVectorizer(stop_words=stopwords.words('english'))
	X=hashvect.fit_transform(df_n)
	
	y=np.reshape(np.array(df.select('score').collect()),(10000,1))
	
	#X_train, X_test, y_train, y_test = train_test_split(X.reshape(10000,-1), y, test_size=0.33, random_state=42)
	
	try:
		model= pickle.load(open('model_SGD.pkl','rb'))
		
	except:
		model = SGDClassifier(alpha=.0001, loss='log', penalty='l2', n_jobs=-1, shuffle=True)
		
	model1=model
	df2=model1.partial_fit(X.reshape(10000,-1),y,classes=[0,4])
	pickle.dump(df2,open('model_SGD.pkl','wb'))
	#y_pred=df2.predict(X_test)
	#accuracy=sklearn.metrics.accuracy_score(y_test,y_pred)	
	#print(accuracy)

def temp(rdd):
	df=spark.read.json(rdd)
	
	for row in df.rdd.toLocalIterator():
		conv_df = spark.createDataFrame(row,columns)
		#conv_df.show()
		preprocessing(conv_df)
		

			
lines.foreachRDD(lambda rdd : temp(rdd))

ssc.start()
ssc.awaitTermination()

