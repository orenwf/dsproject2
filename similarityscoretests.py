import re
import sys
from math import log, sqrt

f = open('/home/oren/Downloads/project2_test.txt')
lines = f.readlines()
f.close()

words = [line.split() for line in lines]

matches = {}
for line in words:
    for word in line:
        if re.compile(
                '.*dis_.*_dis.*').match(word) or re.compile('.*gene_.*_gene.*').match(word):
            if word in matches:
                matches[word] += 1
            else:
                matches[word] = 1

corpus = {line[0]: line[1:] for line in words}

frequencies = {}
for line in words:
    dict = {}
    for word in line:
        if re.compile(
                '.*dis_.*_dis.*').match(word) or re.compile('.*gene_.*_gene.*').match(word):
            if word in dict:
                dict[word] += 1
            else:
                dict[word] = 1
    frequencies[line[0]] = {
        word: total/len(corpus[line[0]]) for word, total in dict.items()}

inverseindex = {}
for docid, dict in frequencies.items():
    for word in dict:
        if word in inverseindex:
            inverseindex[word].append(docid)
        else:
            inverseindex[word] = [docid]
tfidf = {}
for word in inverseindex:
    idf = log(100/len(inverseindex[word]))
    doctfidf = {}
    for doc in inverseindex[word]:
        doctfidf[doc] = frequencies[doc][word]*idf
    tfidf[word] = doctfidf

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
