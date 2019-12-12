import argparse
from pprint import pprint, pformat
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from math import log, sqrt

inputFile = '/home/oren/Downloads/project2_test.txt'
spark = SparkSession.builder.getOrCreate()


def get_corpus(path=inputFile):
    with open(path, 'r') as f:
        rawcorpus = spark.sparkContext.parallelize(f.readlines())
        corpus = rawcorpus.map(
            lambda toStrip: toStrip.strip()).map(
            lambda toSplit: toSplit.split()).map(
            lambda toPart: (toPart[0],
                            toPart[1:]))
        return corpus


# returns a similarity score for two terms
# d1, d2 are dicts of documentId where a term appears: tfidf in that doc
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


def get_similarity_matrix(tfidf):
    return tfidf.cartesian(tfidf).map(
        lambda x: (
            x[0][0], (x[1][0],
                      tfidf2similarity(
                x[0][1], x[1][1])))).groupByKey().map(lambda x: (x[0], dict(x[1])))


def simrank(word, matrix):
    res = matrix.lookup(word)
    if res:
        similar_terms = spark.sparkContext.parallelize(
            list(res[0].items())).map(lambda x: (x[1], x[0]))
        ranked_terms = similar_terms.filter(lambda x: x[0] > 0).sortByKey(
            ascending=False)
        return ranked_terms.map(lambda x: (x[1], x[0])).collect()
    else:
        return None


def getDocKwFreqFrame(corpus):
    doclengths = corpus.map(lambda x: (*x, len(x[1])))
    dockwonly = doclengths.map(
        lambda x: (
            x[0], x[2], [
                word for word in x[1] if 'gene_' in word and '_gene' in word or 'dis_' in word and '_dis' in word]))
    # get frequencies of words in doc
    return dockwonly.map(lambda x: (x[0], terms2freq(x[1], x[2])))


def get_tfidf(corpus):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Similarity score tool using map-reduce on spark.')
    parser.add_argument(
        'corpus', metavar='CORPUS', help='The file path of the corpus to use.')
    parser.add_argument(
        'terms', metavar='TERM', nargs='+', help='Some terms to search for.')
    parser.add_argument(
        '--table', help='Dump the entire similarity score table.')
    args = parser.parse_args()

    filepath = args.corpus
    corpus = get_corpus(filepath)
    tfidf = get_tfidf(corpus)
    matrix = get_similarity_matrix(tfidf)
    terms = args.terms

    rdict = {}

    while(terms):
        term = terms.pop(0)
        res = simrank(term, matrix)
        if res:
            rdict[term] = res
        else:
            rdict[term] = 'No term matching {} has been found.'.format(term)

    with open('mapreduce.result', 'w') as f:
        f.write(pformat(rdict))
        f.write('\n')
