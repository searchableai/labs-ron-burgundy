from functools import lru_cache
from typing import List, NamedTuple, Iterable, Tuple
from collections import Counter

import re
import copy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# snowball stopwords and stemmer
STOPWORDS = set(stopwords.words('english'))
_stemmer = SnowballStemmer('english')


@lru_cache(maxsize=4096)
def stemmer(word: str) -> str:
    """memoized wrapper around PorterStemmer"""
    return _stemmer.stem(word)

NGram = NamedTuple("NGram", [("gram", str), ("position", int)])
Token = NamedTuple("Token",
                   [("word", str),
                    ("position", int),
                    ("is_stopword", bool)])


def tokenize(
        sentence: str,
        stem: bool = True,
        alpha_only: bool = True) -> List[Token]:
    """ Lowercase a sentence, split it into tokens,
        label the stopwords, and throw out words that
        don't contain alphabetic characters
    """
    pre_tokens = [Token(stemmer(w) if stem else w, i, w in STOPWORDS)
                  for i, w in enumerate(word_tokenize(sentence.lower()))]

    if alpha_only:
        clean_tokens = [
            token
            for token in pre_tokens if re.match(r"^[a-z]+$", token.word)
        ]
    else:
        clean_tokens = [
            token
            for token in pre_tokens if re.match(r"^[a-z0-9$]+", token.word)
        ]
    return clean_tokens

def ngrams(n: int, tokens: List[Token], skip: bool=False) -> List[NGram]:
    """ Generate all the ngrams of size n.
        do not allow ngrams that contain stopwords, except that a
        3-gram may contain a stopword as its middle word
    """

    def stopwords_filter(subtokens: List[Token]) -> bool:
        """ Filter stopwords from token list """
        if n == 3:
            return (
                not subtokens[0].is_stopword and not subtokens[2].is_stopword
            )
        else:
            return all(not token.is_stopword for token in subtokens)

    def make_gram(subtokens: List[Token]) -> NGram:
        """ Make a gram using the position of the leftmost
            work and skipping the middle maybe
        """
        words = [
            token.word
            if not skip or i == 0 or i == len(subtokens) - 1 else "_"
            for i, token in enumerate(subtokens)
        ]
        return NGram(" ".join(words), subtokens[0].position)

    # if n is 1, we want len(tokens), etc..
    slices = [tokens[i:(i+n)] for i in range(len(tokens) - n + 1)]

    return [make_gram(sl) for sl in slices if stopwords_filter(sl)]

def distinct_grams(grams: List[NGram]) -> List[str]:
    """ Return the distinct grams from a bunch of ngrams """
    return list({gram.gram for gram in grams})


def all_grams_from_tokens(tokens: List[Token]) -> List[NGram]:
    """ Make all the 1, 2, 3, and skip-3 grams from some tokens """
    return (ngrams(1, tokens) +
            ngrams(2, tokens) +
            ngrams(3, tokens) +
            ngrams(3, tokens, skip=True))


def all_grams(
        sentence: str,
        stem: bool=True,
        alpha_only: bool=True) -> List[NGram]:
    """ Tokenize the sentence and make all the grams """
    return all_grams_from_tokens(
        tokenize(sentence, stem, alpha_only=alpha_only)
    )


class AnswerConfidence(NamedTuple):
    text: str
    score: float
    em: float


def ngram_overlap(sentences, query, remove_list: List, stem=True, alpha_only=False) -> [AnswerConfidence]:
    if not isinstance(sentences, List):
        if not isinstance(sentences, str):
            raise Exception('Sentences must be a list of strings or single string')
        else:
            sentences = [sentences]
    if not isinstance(query, str):
        raise Exception('Query must be a string')

    #print('Processing {} sents'.format(len(sentences)))
    query = query.lower().strip()
    query = re.sub(r'[^A-Za-z0-9.\s\']', '', query)
    sentences = [x.lower().strip() for x in sentences]
    sentences = [re.sub(r'[^A-Za-z0-9.\s\']', '', x) for x in sentences]

    fac = {1:1., 2:5., 3:5.}
    query_ngrams = dict(Counter([x.gram for x in all_grams(query, stem, alpha_only)]))
    query_ngrams = {k:v for k,v in query_ngrams.items() if k not in remove_list}
    query_weights = {k:fac[len(k.replace('_','').split())] for k in query_ngrams.keys()}
    #print({k:query_weights[k] for k in query_ngrams.keys()})

    if not query_ngrams:
        return [AnswerConfidence(text=sentences[n], score=0., em=0.)
            for n in range(len(sentences))]

    norm_factor = sum([v*query_weights[k] for k,v in query_ngrams.items()])

    sent_ngrams = [Counter([y.gram for y in all_grams(x, stem, alpha_only)]) for x in sentences]
    scores = []
    olaps = {}
    #print('\t query: ', query, query_ngrams)
    for n,s in enumerate(sent_ngrams):
        olaps[n] = copy.deepcopy(query_ngrams)
        for k,v in query_ngrams.items():
            if k in s:
                #print('\t', 'sentence', n, k)
                olaps[n][k] -= s[k]
                olaps[n][k] = max(olaps[n][k], 0.)

        #print('===', s, '\n', query_ngrams, len(query_ngrams), '\nscore', olaps[n], 1-sum([j*query_weights[i] for i,j in olaps[n].items()])/norm_factor, norm_factor, sum([query_ngrams[i]-j for i,j in olaps[n].items()]), sum([query_ngrams[i]-j for i,j in olaps[n].items()])/len(query_ngrams))
        scores.append(
            AnswerConfidence(
                text=sentences[n],
                score=1-sum([j*query_weights[i] for i,j in olaps[n].items()])/norm_factor,
                em=sum([query_ngrams[i]-j for i,j in olaps[n].items()])/len(query_ngrams),
            )
        )
    return scores
