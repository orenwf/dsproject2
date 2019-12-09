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


def makeDocumentDictionary(listofwords):
    d = {}
    for word in listofwords:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1
    return d


def dictionaryReducer(left, right):
    for key in right:
        if key in left:
            left[key] |= right[key]
        else:
            left[key] = right[key]
    return left


def getSimilarity(pairs):
    num = sum(x*y for x, y in pairs)
    denom = sqrt(sum(x**2 for x, y in pairs)
                 ) * sqrt(sum(y**2 for x, y in pairs))
    return num/denom


def term_term_similarity(term1, term2, tfidf, index):
    termdocs = tfidf.filter(
        lambda x: x[0] in index[term1] or x[0] in index[term2])
    idfpairs = termdocs.map(
        lambda x: (
            x[2][term1] if term1 in x[2] else 0,
            x[2][term2] if term2 in x[2] else 0))
    numerator = idfpairs.map(lambda x: x[0]*x[1]).reduce(lambda x, y: x+y)
    denomleftterm = sqrt(idfpairs.map(
        lambda x: x[0]**2).reduce(lambda x, y: x+y))
    denomrightterm = sqrt(idfpairs.map(
        lambda x: x[1]**2).reduce(lambda x, y: x+y))
    return numerator/denomleftterm/denomrightterm


def term_corpus_similarity(queryterm, tfidf, index):
    table = {}
    for other in index.keys() - {queryterm}:
        sim = term_term_similarity(queryterm, other, tfidf, index)
        table[other] = sim
    return table
#    return sorted(table.items(), key=lambda x: x[1])
#    return spark.sparkContext.parallelize(
#        table.items()).map(
#        lambda x: (x[1],
#                   x[0])).sortByKey().map(
#        lambda y: (queryterm, y[1],
#                   y[0]))


def all_terms_similarity():
    tfidf, index = getTfidfFrame()
    return {k: term_term_similarity(k, tfidf, index) for k in index.keys()}


def terms2freq(total, words):
    d = {}
    for word in words:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1
    return ((word, count/total) for word, count in d.items())


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
    dockwfreqs = dockwonly.map(lambda x: (x[0], *terms2freq(x[1], x[2])))
    # get docs and freq for each word
    kwdocfreqs = dockwfreqs.flatMap(
        lambda x: [(word, (x[0], freq)) for word, freq in x[1:]]).groupByKey()
    # get idfs
    kwidfs = kwdocfreqs.map(lambda x: (x[0], log(doccount/len(x[1])), x[1]))
    # get tfidfs
    return kwidfs.map(
        lambda x: (
            x[0], {
                docid: x[1]*freq for docid, freq in x[2]}))
