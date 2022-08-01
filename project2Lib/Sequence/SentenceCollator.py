import torch

class SentenceCollator:

    def __call__(self, features):

        # Get lengths
        lengths = [len(x["embeddings"]) for x in features]

        # Get shape of embedding
        B = len(features)
        L = max(lengths)
        N = len(features[0]["embeddings"][0])

        # Pad with zeros
        embeddings = torch.zeros((B, L, N))

        # Fill in with relevant embeddings
        for i in range(B):
            embeddings[i, :lengths[i]] = torch.FloatTensor(features[i]["embeddings"])

        batch = {
            "labels": torch.concat([torch.Tensor(x["label"]).to(dtype=torch.int64) for x in features]),
            "embeddings": embeddings,
            "lengths": lengths
        }
        
        return batch
