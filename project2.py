from pyspark.sql import SparkSession
from math import log, sqrt

inputFile = '/home/oren/Downloads/project2_test.txt'
spark = SparkSession.builder.getOrCreate()


def getCorpus(path=inputFile):
    rawcorpus = spark.sparkContext.textFile(inputFile)
    corpus = rawcorpus.map(
        lambda toStrip: toStrip.strip()).map(
        lambda toSplit: toSplit.split()).map(
        lambda toPart: (toPart[0],
                        toPart[1:]))
    return corpus


def tfidf2similarity(d1, d2):
    return sum([val * d2[doc] for doc, val in d1.items() if doc in d2]) / (
        sqrt(sum([val ** 2 for val in d1.values()])) *
        sqrt(sum([val ** 2 for val in d2.values()])))


def terms2freq(total, words):
    d = {}
    for word in words:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1
    return [(word, count/total) for word, count in d.items()]


def getSimilarityMatrix(tfidf):
    matrix = tfidf.cartesian(tfidf).map(
        lambda x: (
            tfidf2similarity(
                x[0][1], x[1][1]), (x[0][0], x[1][0]))).sortByKey(
                ascending=False).map(
                    lambda x: (
                         x[1][0], (x[1][1], x[0]))).groupByKey()
    return matrix.map(lambda x: (x[0], dict(x[1])))


def simrank(word, matrix):
    res = matrix.lookup(word)
    if res:
        return {term: sim for term, sim in res[0].items() if sim > 0}
    else:
        return None


def getDocKwFreqFrame():
    corpus = getCorpus()
    doclengths = corpus.map(lambda x: (*x, len(x[1])))
    dockwonly = doclengths.map(
        lambda x: (
            x[0], x[2], [
                word for word in x[1] if 'gene_' in word and '_gene' in word or 'dis_' in word and '_dis' in word]))
    # get frequencies of words in doc
    return dockwonly.map(lambda x: (x[0], terms2freq(x[1], x[2])))


def getTfidfFrame():
    corpus = getCorpus()
    doccount = corpus.count()
    # get sizes of each document
    doclengths = corpus.map(lambda x: (*x, len(x[1])))
    # get keywords we care about only
    dockwonly = doclengths.map(
        lambda x: (
            x[0], x[2], [
                word for word in x[1] if 'gene_' in word and '_gene' in word or 'dis_' in word and '_dis' in word]))
    # get frequencies of words in doc
    dockwfreqs = dockwonly.map(lambda x: (x[0], terms2freq(x[1], x[2])))
    # get docs and freq for each word
    kwdocfreqs = dockwfreqs.flatMap(
        lambda x: [(word, (x[0], freq)) for word, freq in x[1]]).groupByKey()
    # get idfs
    kwidfs = kwdocfreqs.map(lambda x: (x[0], log(doccount/len(x[1])), x[1]))
    # get tfidfs
    return kwidfs.map(
        lambda x: (
            x[0], {
                docid: x[1]*freq for docid, freq in x[2]}))
