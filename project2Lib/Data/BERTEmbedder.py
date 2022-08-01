import torch
from transformers import AutoTokenizer, AutoModel

class BERTEmbedder:

    def __init__(
        self, 
        model_checkpoint : str,
        batched : bool = False, 
        device : str = "cuda:0"
    ) -> None:

        # Set up model and tokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, model_max_length=512)
        self.model = AutoModel.from_pretrained(model_checkpoint).to(device = self.device)
        self.batched = batched

    def __call__(self, x):

        embeddings = []
        list_of_sentences = x["sentence"] if self.batched else [x["sentence"]]

        for sentences in list_of_sentences:
            
            # Tokenize sentences using same padding and truncation
            tokenized_sentences = self.tokenizer(sentences, padding=True, truncation=True)["input_ids"]

            # Embed and use pooled outputs from bert model
            with torch.no_grad():
                embeddings.append(self.model(torch.Tensor(tokenized_sentences).to(device = self.device, dtype=torch.int64))[1])

        return {
            "embeddings": embeddings if self.batched else embeddings[0]
        }
