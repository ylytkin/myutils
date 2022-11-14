import re
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import joblib
import nltk
import numpy as np
import pandas as pd
from pymystem3 import Mystem
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

from myutils.json import load_json, save_json

__all__ = [
    "is_word",
    "tokenize_document",
    "tokenize_documents",
    "TopicExtractor",
]


def download_nltk_module_if_does_not_exist(module_path: str) -> None:
    module_name = module_path.split("/")[-1]

    try:
        nltk.data.find(module_path)
    except LookupError:
        nltk.download(module_name)


for module_path_ in [
    "corpora/stopwords",
    "corpora/omw-1.4",
    "corpora/wordnet",
    "taggers/averaged_perceptron_tagger",
]:
    download_nltk_module_if_does_not_exist(module_path_)

TOKENIZER = nltk.tokenize.WordPunctTokenizer()

TAGGER = nltk.tag.PerceptronTagger()

MYSTEM = Mystem()
LEMMATIZER = nltk.stem.WordNetLemmatizer()

STOP_WORDS = set(nltk.corpus.stopwords.words("english") + nltk.corpus.stopwords.words("russian"))

WORD_CHAR_RE = re.compile(r"[a-zA-Zа-яА-ЯёË]")
NON_WORD_CHAR_RE = re.compile(r"[^a-zA-Zа-яА-ЯёË0-9\-]")


def perceptron_tag_to_wordnet_tag(tag: str) -> str:
    """Convert a Perceptron Tagger tag into a WordNet tag.

    :param tag: tag
    :return: converted tag
    """

    if tag.startswith("J"):
        wordnet_tag: str = nltk.corpus.wordnet.ADJ
    elif tag.startswith("V"):
        wordnet_tag = nltk.corpus.wordnet.VERB
    elif tag == "RB":
        wordnet_tag = nltk.corpus.wordnet.ADV
    else:
        wordnet_tag = nltk.corpus.wordnet.NOUN

    return wordnet_tag


def _tokenize_document(document: str, lemmatize: bool = True) -> List[str]:
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

        def _lemmatize_english(word: str, tag: str) -> str:
            token: str = LEMMATIZER.lemmatize(word=word, pos=perceptron_tag_to_wordnet_tag(tag))

            return token

        tagged_tokens: List[Tuple[str, str]] = TAGGER.tag(tokens)

        tokens = [_lemmatize_english(word, tag) for word, tag in tagged_tokens]

        return tokens

    tokens = TOKENIZER.tokenize(document)

    return tokens


def is_word(token: str) -> bool:
    """Checks whether the given token is an English or Russian word.
    Rule: contains only alphabet characters and possibly a dash.

    :param token: str
    :return: bool
    """

    return (WORD_CHAR_RE.search(token) is not None) and (NON_WORD_CHAR_RE.search(token) is None)


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

    tokens = _tokenize_document(document, lemmatize=lemmatize)

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

    return list(
        map(
            lambda x: tokenize_document(
                x,
                lemmatize=lemmatize,
                remove_non_word_tokens=remove_non_word_tokens,
                remove_stop_words=remove_stop_words,
            ),
            documents,
        )
    )


class TopicExtractor:
    # pylint: disable=too-many-instance-attributes

    N_COMPONENTS = 512

    TOPIC_EXTRACTOR_TRANSFORM_BATCH_SIZE = 32

    TOKENS_JSON = "tokens.json"
    TFIDF_JOBLIB = "tfidf.joblib"
    TOPIC_EXTRACTOR_JOBLIB = "topic_extractor.joblib"

    def __init__(self) -> None:
        self.tokens: Optional[List[str]] = None
        self.tokens_set: Optional[Set[str]] = None
        self.n_tokens: Optional[int] = None
        self.token2id: Optional[Dict[str, int]] = None
        self.id2token: Optional[Dict[int, str]] = None

        self.tfidf_transformer: Optional[TfidfTransformer] = None
        self.topic_extractor: Optional[TruncatedSVD] = None

        self.fitted = False

    def fit_transform(self, documents: List[str]) -> sparse.csr_matrix:
        tokenized_documents = tokenize_documents(documents)

        self.tokens_set = set(chain.from_iterable(tokenized_documents))
        self.tokens = list(self.tokens_set)
        self.n_tokens, self.token2id, self.id2token = self._encode_tokens(self.tokens)

        document_token_matrix = self._build_document_token_matrix(tokenized_documents)

        self.tfidf_transformer = TfidfTransformer(sublinear_tf=True, use_idf=True)
        document_token_matrix_tfidfed = self.tfidf_transformer.fit_transform(document_token_matrix)

        n_components = min(self.n_tokens - 1, self.N_COMPONENTS)
        self.topic_extractor = TruncatedSVD(n_components=n_components).fit(
            document_token_matrix_tfidfed
        )

        self.fitted = True

        document_topic_matrix = self._extract_document_topics(document_token_matrix_tfidfed)

        return document_topic_matrix

    def transform(self, documents: List[str]) -> sparse.csr_matrix:
        if self.fitted is False or self.tfidf_transformer is None:
            raise AttributeError(
                "Attempted transform using an untrained model. "
                "You should use `.fit_transform()` method first."
            )

        tokenized_documents = tokenize_documents(documents)

        document_token_matrix = self._build_document_token_matrix(tokenized_documents)

        document_token_matrix_tfidfed = self.tfidf_transformer.transform(document_token_matrix)

        document_topic_matrix = self._extract_document_topics(document_token_matrix_tfidfed)

        return document_topic_matrix

    def save(self, dirpath: Union[str, Path]) -> None:
        if (
            self.fitted is False
            or self.tokens is None
            or self.tfidf_transformer is None
            or self.topic_extractor is None
        ):
            raise AttributeError(
                "Attempted transform using an untrained model. "
                "You should use `.fit_transform()` method first."
            )

        dirpath = Path(dirpath)
        dirpath.mkdir(exist_ok=True, parents=True)

        save_json(self.tokens, dirpath / self.TOKENS_JSON)
        joblib.dump(self.tfidf_transformer, dirpath / self.TFIDF_JOBLIB)
        joblib.dump(self.topic_extractor, dirpath / self.TOPIC_EXTRACTOR_JOBLIB)

    @classmethod
    def load(cls, dirpath: Union[str, Path]) -> "TopicExtractor":
        dirpath = Path(dirpath)

        topic_extractor = TopicExtractor()
        topic_extractor.tokens = load_json(dirpath / cls.TOKENS_JSON)  # type: ignore

        if topic_extractor.tokens is None:
            raise RuntimeError("this should not be reachable")

        topic_extractor.tokens_set = set(topic_extractor.tokens)
        (
            topic_extractor.n_tokens,
            topic_extractor.token2id,
            topic_extractor.id2token,
        ) = topic_extractor._encode_tokens(topic_extractor.tokens)

        topic_extractor.tfidf_transformer = joblib.load(dirpath / cls.TFIDF_JOBLIB)
        topic_extractor.topic_extractor = joblib.load(dirpath / cls.TOPIC_EXTRACTOR_JOBLIB)

        topic_extractor.fitted = True

        return topic_extractor

    def _build_document_token_matrix(
        self, tokenized_documents: List[List[str]]
    ) -> sparse.csr_matrix:
        if self.fitted is False or self.tokens_set is None or self.token2id is None:
            raise AttributeError(
                "Attempted transform using an untrained model. "
                "You should use `.fit_transform()` method first."
            )

        n_documents = len(tokenized_documents)

        matrix = sparse.lil_matrix((n_documents, self.n_tokens))

        for i, tokens in enumerate(tokenized_documents):
            tokens = [token for token in tokens if token in self.tokens_set]
            token_ids = list(map(self.token2id.get, tokens))

            token_id_counts = pd.Series(token_ids).value_counts()
            token_ids_array = token_id_counts.index.to_numpy()
            token_id_counts = token_id_counts.values

            document_ids = np.ones(token_ids_array.size) * i

            matrix._set_arrayXarray(  # pylint: disable=protected-access
                document_ids,
                token_ids_array,
                token_id_counts,
            )

        return matrix.tocsr()

    def _extract_document_topics(
        self,
        document_token_matrix: sparse.csr_matrix,
        batch_size: Optional[int] = None,
    ) -> sparse.csr_matrix:
        if self.fitted is False or self.topic_extractor is None:
            raise AttributeError(
                "Attempted transform using an untrained model. "
                "You should use `.fit_transform()` method first."
            )

        batch_size = batch_size or self.TOPIC_EXTRACTOR_TRANSFORM_BATCH_SIZE

        document_topic_matrix = sparse.vstack(
            blocks=[
                sparse.csr_matrix(
                    self.topic_extractor.transform(
                        document_token_matrix[offset : offset + batch_size]
                    )
                )
                for offset in range(0, document_token_matrix.shape[0], batch_size)
            ],
            format="csr",
        )

        return document_topic_matrix

    @staticmethod
    def _encode_tokens(tokens: List[str]) -> Tuple[int, Dict[str, int], Dict[int, str]]:
        n_tokens = len(tokens)

        token2id = dict(zip(tokens, range(n_tokens)))
        id2token = {value: key for key, value in token2id.items()}

        return n_tokens, token2id, id2token
