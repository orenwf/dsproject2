from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, IDF

inputFile = '/home/oren/Downloads/project2_test.txt'

spark = SparkSession.builder.getOrCreate()
corpus = spark.sparkContext.textFile(inputFile)
df = corpus.map(
    lambda toStrip: toStrip.strip()).map(
    lambda toSplit: toSplit.split()).map(
    lambda toPart: (toPart[: 1],
                    toPart[1:])).toDF(
    ['DocId', 'words'])

countvectorizer = CountVectorizer(inputCol='words', outputCol='tfFeat')
model = countvectorizer.fit(df)
tfFeatures = model.transform(df)
tfFeatures.show()

idf = IDF(inputCol='tfFeat', outputCol='idfFeat')
idfModel = idf.fit(tfFeatures)
idfFeatures = idfModel.transform(tfFeatures)
idfFeatures.show()

spark.stop()
