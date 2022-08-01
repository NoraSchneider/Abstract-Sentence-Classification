import torch

class TokenCollator:

    def __call__(self, features):

        # Get lengths
        lengths = [len(x["embeddings"]) for x in features]
        embeddings = []

        # Fill in with relevant embeddings
        for x in features:

            # Get shape of embedding
            S = len(x["embeddings"])
            L = max(len(s) for s in x["embeddings"])
            N = len(x["embeddings"][0][0])

            # Pad with zeros at end in 2nd dimension
            x_embed = torch.zeros((S, L, N))

            # Load embeddings
            for i, s in enumerate(x["embeddings"]):
                if s:
                    x_embed[i, :len(s)] = torch.FloatTensor(s)

            # Add to list of embeddings
            embeddings.append(x_embed)
            

        batch = {
            "labels": torch.concat([torch.Tensor(x["label"]).to(dtype=torch.int64) for x in features]),
            "embeddings": embeddings,
            "lengths": lengths
        }
        
        return batch
