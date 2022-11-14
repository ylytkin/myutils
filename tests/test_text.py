from myutils.text import tokenize_documents


def test_text():
    documents = ["My cat is a very strange creature.", "Да ну, это какая-то фигня..."]
    tokenized_documents_real = [["cat", "strange", "creature"], ["это", "какой-то", "фигня"]]

    tokenized_documents = tokenize_documents(
        documents,
        lemmatize=True,
        remove_non_word_tokens=True,
        remove_stop_words=True,
    )

    assert len(tokenized_documents) == len(tokenized_documents_real)

    assert all(
        tokenized_document == tokenized_document_real
        for tokenized_document, tokenized_document_real in zip(
            tokenized_documents,
            tokenized_documents_real,
        )
    )
