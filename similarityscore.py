import argparse
from datetime import datetime
import re
from pprint import pformat
from pyspark.sql import SparkSession
from math import log, sqrt

RESULT_FILE_NAME_ROOT = 'mapreduce.result'

# builds the RDD from the file assuming 1 per line and first word = docId


def get_corpus(path):
    spark = SparkSession.builder.getOrCreate()
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
    numerator = sum([tfidf * term2[doc]
                     for doc, tfidf in term1.items() if doc in term2])
    leftdenom = sqrt(sum([tfidf ** 2 for tfidf in term1.values()]))
    rightdenom = sqrt(sum([tfidf ** 2 for tfidf in term2.values()]))
    return numerator / (leftdenom * rightdenom)


# does a word count on a document
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


# maps the document corpus to an RDD that describes the term frequencies
def get_frequencies(corpus, regex, use_original_doc_len):
    table = corpus.map(lambda doc: (*doc, len(doc[1])))
    # filter only words that match the regex for the matrix
    if regex is not None:
        table = table.map(
            lambda doc: (
                doc[0], [
                    word for word in doc[1] if regex.search(word)], doc[2]))
    return table.map(lambda doc: (doc[0], terms2freq(
        doc[2] if use_original_doc_len else len(doc[1]), doc[1])))


# returns an invserse index of word -> ((doc1, tf1), (doc2, tf2), ...)
# used to calculate idf
def get_id_index(frequencies):
    table = frequencies.flatMap(
        lambda doc: [
            (word, (doc[0], frequency)) for word, frequency in doc[1]]).groupByKey()
    return table.map(lambda word: (word[0], dict(word[1])))


# maps the inverse index -> IDF with word frequencies together in an RDD
def get_idf_table(table, corpus_size):
    return table.map(lambda word: (
        word[0], log(corpus_size/len(word[1])), word[1]))


# generates the table of tfidf for each word -> document:tfidf pairs
def get_tfidf_table(input_table):
    return input_table.map(lambda word: (
        word[0], {docId: word[1]*tf for docId, tf in word[2].items()}))


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


# runs the lookups and ranks the similarities for a list of lookup terms
def run_queries(terms, matrix, args):
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
    return rdict


# will overwrite any existing output file
def write_result_file(rdict, filename):
    with open('{}.{}.{}'.format(RESULT_FILE_NAME_ROOT, filename, datetime.isoformat(datetime.today())), 'w') as f:
        f.write(
            '{:<8}{:^64}{:<}\n'.format(
                'CSCI 795',
                'Big Data Seminar - Oren Friedman',
                'Project 2'))
        f.write('{}\n'.format('='*128))
        for word, result in rdict.items():
            if not result:
                f.write('No term similar to {} has been found.\n'.format(word))
            else:
                f.write('{:<8}{}\n'.format('FOUND', word))
                f.write(
                    '{:<8}{:64}{:<}\n'.format(
                        'RANK',
                        'TERM',
                        'COSINE SIMILARITY'))
                for rank, pair in enumerate(result[:args.max], start=1):
                    f.write(
                        '{:<8}{:.<64}{:<}\n'.format(
                            rank, pair[0], pair[1]))
            f.write('{}\n'.format('='*128))
        f.write('\n')


def parse_cli():
    parser = argparse.ArgumentParser(
        description='Similarity score tool using map-reduce on spark.')
    parser.add_argument(
        'corpus', metavar='CORPUS', help='The file path of the corpus to use.')
    parser.add_argument(
        'terms', metavar='TERM', nargs='*', help='Some terms to search for.')
    parser.add_argument(
        '--table',
        action='store_true',
        help='Dump the entire similarity score table.')
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
        '--original-doc-len',
        action='store_true',
        help='Preserve the original document lengths for TF calcs.')
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_cli()

    # the default filter for terms in matrix is 'dis_X_dis' and 'gene_X_gene'
    regex = None if args.nofilter else re.compile(
        args.filter)

    filepath = args.corpus
    filename = filepath.split('/')[-1]
    corpus = get_corpus(filepath)
    frequencies = get_frequencies(corpus, regex, args.original_doc_len)
    id_index = get_id_index(frequencies)
    idf_table = get_idf_table(id_index, corpus.count())
    tfidf = get_tfidf_table(idf_table)
    tfidf.cache()
    matrix = get_similarity_matrix(tfidf)
    matrix.cache()

    terms = args.terms

    # can attempt to dump entire similarity matrix for introspection
    if args.table:
        dump = dict(matrix.collect())
        with open('MATRIX_DUMP.{}'.format(filename), 'w') as f:
            f.write(pformat(dump))
            f.write('\n')

    # run the queries against the matrix
    rdict = run_queries(terms, matrix, args)

    # proceed to pretty file output for lookup results
    write_result_file(rdict, filename)
