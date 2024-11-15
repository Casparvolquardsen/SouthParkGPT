import pytest

from evaluate import evaluate_bleu_score


def test_equal_input():
    prediction = "the squirrel is eating the nut"
    reference = "the squirrel is eating the nut"

    bleu_score = evaluate_bleu_score(prediction, reference)

    assert bleu_score == 1, "BLEU score should be 1"


def test_no_word_equal():
    prediction = "the squirrel is eating the nut"
    reference = "My cat has a great day"

    bleu_score = evaluate_bleu_score(prediction, reference)

    assert bleu_score == 0, "BLEU score should be 0"
