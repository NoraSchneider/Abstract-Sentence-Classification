from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import torch
import re

from .preprocess import preprocess_text

class W2VEmbedder:

    def __init__(
        self, 
        model_checkpoint : str,
        batched : bool = False,
    ) -> None:

        # Set up model and tokenizer
        self.tokenizer = lambda s: word_tokenize(preprocess_text(s, mode="lemmatization", remove_numplaceholder=False))
        self.kv = KeyedVectors.load_word2vec_format(model_checkpoint, binary=True)
        self.batched = batched

    def __call__(self, x):

        embeddings = []
        list_of_sentences = x["sentence"] if self.batched else [x["sentence"]]

        for sentences in list_of_sentences:
            
            # Tokenize sentences and embed the sentences
            embeddings.append([[self.kv.get_vector(token) for token in self.tokenizer(s) if token in self.kv.vocab] for s in sentences])

        return {
            "embeddings": embeddings if self.batched else embeddings[0]
        }


