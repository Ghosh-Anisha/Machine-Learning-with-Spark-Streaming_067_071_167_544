# Machine-Learning-with-Spark-Streaming_067_071_167_544
Sentiment Analysis

Models Implemented:
1. SGD - Stochastic Gradient Descent 
2. MNB - Multinomial Naive Bayes 
3. BNB - Bernoulli Naive Bayes
4. KMC - K-Means Clustering 
5. MLP - Multilayer Perceptron

The folders are as follows:
1. models: all the model files mentioned above are present here
2. test: there are 2 test files, one for KMC and the other for all the models

Instructions for running and testing models:

1. Run stream.py and appropriate script file for model to train:
   <br>
   python3 stream.py -f sentiment -b <batchsize>
   <br>
   $SPARK_HOME/bin/spark-submit script.py > out.text 
   <br>
2. Run stream.py (uncomment test) and run testfile.py with appropriate batch size

  
