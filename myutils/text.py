import re
from typing import List, Tuple, Union, Dict, Optional
from pathlib import Path
from itertools import chain

import joblib
import nltk
import numpy as np
import pandas as pd
from pymystem3 import Mystem
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

from myutils.json import save_json, load_json

__all__ = [
    'is_word',
    'tokenize_document',
    'tokenize_documents',
    'TopicExtractor',
]

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

TOKENIZER = nltk.tokenize.WordPunctTokenizer()

TAGGER = nltk.tag.PerceptronTagger()


def perceptron_tag_to_wordnet_tag(tag: str) -> str:
    """Convert a Perceptron Tagger tag into a WordNet tag.

    :param tag: tag
    :return: converted tag
    """

    if tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif tag == 'RB':
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN


MYSTEM = Mystem()
LEMMATIZER = nltk.stem.WordNetLemmatizer()

STOP_WORDS = set(nltk.corpus.stopwords.words('english') + nltk.corpus.stopwords.words('russian'))

WORD_CHAR_RE = re.compile(r'[a-zA-Zа-яА-ЯёË]')
NON_WORD_CHAR_RE = re.compile(r'[^a-zA-Zа-яА-ЯёË0-9\-]')


def __tokenize_document(document: str, lemmatize: bool = True) -> List[str]:
    """Tokenize the given document. Performs lower-casing before tokenizing.

    :param document: str
    :param lemmatize: whether to lemmatize tokens
    :return: tokens (list of str)
    """

    if not isinstance(document, str):
        return []

    document = document.lower()

    if lemmatize:
        tokens: List[str] = MYSTEM.lemmatize(document)  # only lemmatizes russian

        tagged_tokens: List[Tuple[str, str]] = TAGGER.tag(tokens)
        tokens: List[str] = list(map(
            lambda x: LEMMATIZER.lemmatize(word=x[0], pos=perceptron_tag_to_wordnet_tag(x[1])),
            tagged_tokens
        ))  # lemmatizes english

        return tokens

    else:
        return TOKENIZER.tokenize(document)


def is_word(token: str) -> bool:
    """Checks whether the given token is an English or Russian word.
    Rule: contains only alphabet characters and possibly a dash.

    :param token: str
    :return: bool
    """

    return (
        (WORD_CHAR_RE.search(token) is not None)
        and (NON_WORD_CHAR_RE.search(token) is None)
    )


def is_stop_word(token: str) -> bool:
    """Checks whether the given token is an English or Russian stop word.

    :param token: str
    :return: bool
    """

    return token in STOP_WORDS


def tokenize_document(
        document: str,
        lemmatize: bool = True,
        remove_non_word_tokens: bool = True,
        remove_stop_words: bool = True,
) -> List[str]:
    """Tokenize the given document.

    :param document: str
    :param lemmatize: whether to lemmatize tokens
    :param remove_non_word_tokens: whether to remove non-word tokens
    :param remove_stop_words: whether to remove stop words
    :return: tokens (list of str)
    """

    tokens = __tokenize_document(document, lemmatize=lemmatize)

    if remove_non_word_tokens:
        tokens = list(filter(is_word, tokens))

    if remove_stop_words:
        tokens = list(filter(lambda x: not is_stop_word(x), tokens))

    return tokens


def tokenize_documents(
        documents: List[str],
        lemmatize: bool = True,
        remove_non_word_tokens: bool = True,
        remove_stop_words: bool = True,
) -> List[List[str]]:
    """Apply `tokenize_document` to multiple documents.

    :param documents: list of documents
    :param lemmatize: whether to lemmatize tokens
    :param remove_non_word_tokens: whether to remove non-word tokens
    :param remove_stop_words: whether to remove stop words

    :return: list of tokenized documents (i.e. lists of tokens)
    """

    return list(map(
        lambda x: tokenize_document(
            x,
            lemmatize=lemmatize,
            remove_non_word_tokens=remove_non_word_tokens,
            remove_stop_words=remove_stop_words,
        ),
        documents
    ))


class TopicExtractor:
    N_COMPONENTS = 512

    TOPIC_EXTRACTOR_TRANSFORM_BATCH_SIZE = 32

    TOKENS_JSON = 'tokens.json'
    TFIDF_JOBLIB = 'tfidf.joblib'
    TOPIC_EXTRACTOR_JOBLIB = 'topic_extractor.joblib'

    def __init__(self) -> None:
        self.tokens = None
        self.tokens_set = None
        self.n_tokens = None
        self.token2id = None
        self.id2token = None

        self.tfidf_transformer = None
        self.topic_extractor = None

        self.fitted = False

    def fit_transform(self, documents: List[str]) -> sparse.csr_matrix:
        tokenized_documents = tokenize_documents(documents)

        self.tokens_set = set(chain.from_iterable(tokenized_documents))
        self.tokens = list(self.tokens_set)
        self.n_tokens, self.token2id, self.id2token = self._encode_tokens(self.tokens)

        document_token_mx = self._build_document_token_matrix(tokenized_documents)

        self.tfidf_transformer = TfidfTransformer(sublinear_tf=True, use_idf=True)
        document_token_mx_tfidfed = self.tfidf_transformer.fit_transform(document_token_mx)

        n_components = min(self.n_tokens - 1, self.N_COMPONENTS)
        self.topic_extractor = TruncatedSVD(n_components=n_components).fit(document_token_mx_tfidfed)

        self.fitted = True

        document_topic_mx = self._extract_document_topics(document_token_mx_tfidfed)

        return document_topic_mx

    def transform(self, documents: List[str]) -> sparse.csr_matrix:
        if self.fitted is False:
            raise Exception('Attempted transform using an untrained model. '
                            'You should use `.fit_transform()` method first.')

        tokenized_documents = tokenize_documents(documents)

        document_token_mx = self._build_document_token_matrix(tokenized_documents)
        document_token_mx_tfidfed = self.tfidf_transformer.transform(document_token_mx)

        document_topic_mx = self._extract_document_topics(document_token_mx_tfidfed)

        return document_topic_mx

    def save(self, dirpath: Union[str, Path]) -> None:
        dirpath = Path(dirpath)

        dirpath.mkdir(exist_ok=True)

        save_json(self.tokens, dirpath / self.TOKENS_JSON)
        joblib.dump(self.tfidf_transformer, dirpath / self.TFIDF_JOBLIB)
        joblib.dump(self.topic_extractor, dirpath / self.TOPIC_EXTRACTOR_JOBLIB)

    @classmethod
    def load(cls, dirpath: Union[str, Path]) -> 'TopicExtractor':
        dirpath = Path(dirpath)

        topic_extractor = TopicExtractor()
        topic_extractor.tokens = load_json(dirpath / cls.TOKENS_JSON)

        topic_extractor.tokens_set = set(topic_extractor.tokens)
        (
            topic_extractor.n_tokens,
            topic_extractor.token2id,
            topic_extractor.id2token
        ) = topic_extractor._encode_tokens(topic_extractor.tokens)

        topic_extractor.tfidf_transformer = joblib.load(dirpath / cls.TFIDF_JOBLIB)
        topic_extractor.topic_extractor = joblib.load(dirpath / cls.TOPIC_EXTRACTOR_JOBLIB)

        topic_extractor.fitted = True

        return topic_extractor

    def _build_document_token_matrix(self, tokenized_documents: List[List[str]]) -> sparse.csr_matrix:
        n_documents = len(tokenized_documents)

        mx = sparse.lil_matrix((n_documents, self.n_tokens))

        for i, tokens in enumerate(tokenized_documents):
            tokens = filter(lambda token: token in self.tokens_set, tokens)
            token_ids = list(map(self.token2id.get, tokens))

            token_id_counts = pd.Series(token_ids).value_counts()
            token_ids = token_id_counts.index.to_numpy()
            token_id_counts = token_id_counts.values

            document_ids = np.ones(token_ids.size) * i

            mx._set_arrayXarray(document_ids, token_ids, token_id_counts)

        return mx.tocsr()

    def _extract_document_topics(
            self,
            document_token_mx: sparse.csr_matrix,
            batch_size: Optional[int] = None,
    ) -> sparse.csr_matrix:
        batch_size = batch_size or self.TOPIC_EXTRACTOR_TRANSFORM_BATCH_SIZE

        document_topic_mx = sparse.vstack(
            blocks=[
                sparse.csr_matrix(self.topic_extractor.transform(document_token_mx[offset:offset + batch_size]))
                for offset in range(0, document_token_mx.shape[0], batch_size)
            ],
            format='csr',
        )

        return document_topic_mx

    @staticmethod
    def _encode_tokens(tokens: List[str]) -> Tuple[int, Dict[str, int], Dict[int, str]]:
        n_tokens = len(tokens)

        token2id = dict(zip(tokens, range(n_tokens)))
        id2token = {value: key for key, value in token2id.items()}

        return n_tokens, token2id, id2token
