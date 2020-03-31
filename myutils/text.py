import re
from typing import List, Tuple

import nltk
from pymystem3 import Mystem

__all__ = [
    'is_word',
    'tokenize_document',
    'tokenize_documents',
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
