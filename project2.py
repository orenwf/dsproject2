import argparse
import re
from pprint import pformat
from pyspark.sql import SparkSession
from math import log, sqrt

spark = SparkSession.builder.getOrCreate()
RESULT_FILE_NAME_ROOT = 'mapreduce.result'


def get_corpus(path):
    with open(path, 'r') as f:
        rawcorpus = spark.sparkContext.parallelize(f.readlines())
        corpus = rawcorpus.map(
            lambda toStrip: toStrip.strip()).map(
            lambda toSplit: toSplit.split()).map(
            lambda toPart: (toPart[0],
                            toPart[1:]))
        return corpus


# returns a similarity score for two terms
# term1, term2 are dicts of documentId where a term appears: tfidf in that doc
def tfidf2similarity(term1, term2):
    numerator = sum([tfidf * term2[doc] for doc, tfidf in term1.items() if doc in term2])
    leftdenom = sqrt(sum([tfidf ** 2 for tfidf in term1.values()]))
    rightdenom = sqrt(sum([tfidf ** 2 for tfidf in term2.values()]))
    return numerator / (leftdenom * rightdenom)


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
    similar_terms = matrix.filter(
        lambda x: x[0] == word).flatMap(
        lambda x: [
            (word, simscore) for word, simscore in x[1].items()]).map(
                lambda x: (
                    x[1], x[0]))
    ranked_terms = similar_terms.filter(lambda x: x[0] > 0).sortByKey(
        ascending=False)
    return ranked_terms.map(lambda x: (x[1], x[0]))


def get_tfidf(corpus, regex):
    doccount = corpus.count()
    # get sizes of each document
    doclengths = corpus.map(lambda x: (*x, len(x[1])))
    # get keywords we care about only
    dockwonly = doclengths.map(
        lambda x: (
            x[0], x[2], x[1])) if regex is None else doclengths.map(
            lambda x: (
                x[0], x[2], [
                     word for word in x[1] if regex.search(word)]))
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
        '--table',
        action='store_true',
        help='Dump the entire similarity score table.')
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test the mapreduce operation against serial operation.')
    parser.add_argument(
        '--nofilter',
        action='store_true',
        help='Generate similarty matrix without filtering.')
    parser.add_argument(
        '--filter',
        type=str,
        default='.*dis_.*_dis.*|.*gene_.*_gene.*',
        help='Filter the terms that go into the similarity matrix.')
    parser.add_argument(
        '--search',
        nargs=1,
        type=str,
        help='Search and rank only matching terms.')
    parser.add_argument(
        '--max',
        metavar='M',
        type=int,
        nargs='?',
        help='Maximum rank to display.',
        default=5)
    args = parser.parse_args()

    regex = None if args.nofilter else re.compile(
        args.filter)
    filepath = args.corpus
    filename = filepath.split('/')[-1]
    corpus = get_corpus(filepath)
    tfidf = get_tfidf(corpus, regex)
    matrix = get_similarity_matrix(tfidf)
    matrix.cache()
    terms = args.terms

    # sanity check to test against a single threaded python app
    if args.test:
        from similarityscoretests import test
        similarityscores = test(filepath)
        p2matrix = dict(matrix.collect())

        for term1 in similarityscores:
            for term2 in similarityscores[term1]:
                if abs(p2matrix[term1][term2] - similarityscores[term1]
                       [term2]) > 0.0000001:
                    print(
                        'TEST:{}:{}:{} not equal to MR:{}'.format(
                            term1,
                            term2,
                            similarityscores[term1][term2],
                            p2matrix[term1][term2]))
            else:
                print('TEST PASSED')

    # can attempt to dump entire similarity matrix for introspection
    if args.table:
        dump = dict(matrix.collect())
        with open('MATRIX_DUMP.{}'.format(filename), 'w') as f:
            f.write(pformat(dump))
            f.write('\n')

    # otherwise proceed to pretty file output for lookup results
    rdict = {}
    while(terms):
        term = terms.pop(0)
        searchresult = simrank(
            term, matrix).filter(
                # filter out the word itself from the search rankings
                lambda x: x[0] != term)
        # if we have asked for result filtering for only certain regex, do it
        if args.search:
            searchresult = searchresult.filter(
                lambda x: re.compile(
                    args.search[0]).search(
                    x[0]))
        rdict[term] = searchresult.collect()

    with open('{}.{}'.format(RESULT_FILE_NAME_ROOT, filename), 'w') as f:
        f.write('CSCI 795 \t Big Data Seminar \t Oren Friedman \t Project 2\n')
        f.write('{}\n'.format('='*128))
        for word, result in rdict.items():
            if not result:
                f.write('No term similar to {} has been found.\n'.format(word))
            else:
                f.write('FOUND\t{}\n'.format(word))
                f.write(
                    '{:<8}{:64}{:<}\n'.format(
                        'RANK',
                        'TERM',
                        'COSINE SIMILARITY'))
                for rank, pair in enumerate(result[:args.max], start=1):
                    f.write('{:<8}{:64}{:<}\n'.format(rank, pair[0], pair[1]))
            f.write('{}\n'.format('='*128))
        f.write('\n')
