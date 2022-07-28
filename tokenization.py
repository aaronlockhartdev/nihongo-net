# Typing
from typing import List

# Libraries
import spacy
import tensorflow as tf

from nltk.lm import Vocabulary


def tokenize_eng(text: List(str)):
    nlp = spacy.load("en_core_web_trf")

    for t in text:
        yield from nlp(t)
