from project2 import *


def get_test_data(filepath, args):
    f = open(filepath)
    lines = f.readlines()
    f.close()

    lines = [line.strip().split() for line in lines]
    assert lines, "@@@@@@@@@@@@@@@         NO INPUT          @@@@@@@@@@@@@@@@@"

    corpus = [(line[0], line[1:]) for line in lines]

    regex = None if args.nofilter else re.compile(args.filter)
    frequencies = {}
    for docid, text in corpus:
        matches = {}
        for word in text:
            if regex is None or regex.search(word):
                if word in matches:
                    matches[word] += 1
                else:
                    matches[word] = 1
        frequencies[docid] = {
            word: total/len(text) for word, total in matches.items()}

    assert len(
        frequencies) > 1, "@@@@@@@@@@@@@@@         NO INPUT          @@@@@@@@@@@@@@@@@"

    inverseindex = {}
    for docid, matches in frequencies.items():
        for word in matches:
            if word in inverseindex:
                inverseindex[word].append(docid)
            else:
                inverseindex[word] = [docid]
    assert len(
        inverseindex) > 1, "@@@@@@@@@@@@@@@         NO INPUT          @@@@@@@@@@@@@@@@@"

    tfidf = {}
    for word in inverseindex:
        idf = log(len(corpus)/len(inverseindex[word]))
        doctfidf = {}
        for doc in inverseindex[word]:
            doctfidf[doc] = frequencies[doc][word]*idf
        tfidf[word] = doctfidf

    assert len(
        tfidf) > 1, "@@@@@@@@@@@@@@@         NO INPUT          @@@@@@@@@@@@@@@@@"

    similarityscores = {}

    for word1 in tfidf:
        w1dict = {}
        for word2 in tfidf:
            pairs = []
            w1tfidf = tfidf[word1].values()
            w2tfidf = tfidf[word2].values()
            for docid in tfidf[word1]:
                if docid in tfidf[word2]:
                    pairs.append((tfidf[word1][docid], tfidf[word2][docid]))
            if len(pairs) > 0:
                num = sum([x*y for x, y in pairs])
                denomleft = sqrt(sum([x**2 for x in w1tfidf]))
                denomright = sqrt(sum([y**2 for y in w2tfidf]))
                sim = num/denomleft/denomright
                w1dict[word2] = sim
        similarityscores[word1] = w1dict

    assert len(
        similarityscores) > 1, "@@@@@@@@@@@@@@@         NO INPUT          @@@@@@@@@@@@@@@@@"

    return similarityscores


if __name__ == '__main__':
    args = parse_cli()

    filepath = args.corpus
    filename = filepath.split('/')[-1]
    corpus = get_corpus(filepath)
    frequencies = get_frequencies(
        corpus, None if args.nofilter else re.compile(
            args.filter), args.original_doc_len)
    idf_table = get_idf_table(get_id_index(frequencies), corpus.count())
    matrix = get_similarity_matrix(get_tfidf_table(idf_table))
    p2matrix = dict(matrix.collect())

    similarityscores = get_test_data(filepath, args)
    with open('TEST_DATA.{}.{}.{}'.format(RESULT_FILE_NAME_ROOT, filename, datetime.isoformat(datetime.today())), 'w') as f:
        for term1 in similarityscores:
            for term2 in similarityscores[term1]:
                if abs(p2matrix[term1][term2] - similarityscores[term1]
                        [term2]) > 0.0000001:
                    f.write(
                        'TEST:{}:{}:{} not equal to MR:{}\n'.format(
                            term1,
                            term2,
                            similarityscores[term1][term2],
                            p2matrix[term1][term2]))
            else:
                f.write('TEST:{}:PASSED\n'.format(term1))
