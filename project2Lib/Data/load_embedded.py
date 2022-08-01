import imp
from datasets import load_from_disk
import inspect
from pathlib import Path
from typing import Optional, List

from .load import load_data
from .BERTEmbedder import BERTEmbedder
from .W2VEmbedder import W2VEmbedder

embedding_dict = {
    "bert": BERTEmbedder,
    "w2v": W2VEmbedder
}

def load_embedded(
    dataset_path : Path, 
    embedding : Optional[str] = None,
    save : Optional[bool] = True,
    fields : Optional[List[str]] = ["embeddings", "label"],
    **kwargs
):

    # Load if exists already
    if dataset_path.exists():
        embedded_dataset = load_from_disk(dataset_path)
    
    # Create if not already exists
    else:

        # Load raw data
        load_kwargs = {kw: kwargs[kw] for kw in inspect.signature(load_data).parameters if kw in kwargs}
        dataset = load_data(**load_kwargs)

        # Initialize embedding function
        embedding_kwargs = {kw: kwargs[kw] for kw in inspect.signature(embedding_dict[embedding].__init__).parameters if kw in kwargs}
        embedder = embedding_dict[embedding](**embedding_kwargs)

        # Add embedding
        embedded_dataset = dataset.map(
            embedder, 
            batched = False
        )

        # Save if requested
        if save:
            dataset_path.mkdir(parents=True, exist_ok=True)
            embedded_dataset.save_to_disk(dataset_path)
    
    # Drop unwanted fields if requested
    if not fields is None:
        all_columns = {k for subset in embedded_dataset.data.values() for k in subset.column_names}
        remove_columns = [k for k in all_columns if not k in fields]
        embedded_dataset = embedded_dataset.map(remove_columns=remove_columns)

    # Return embedded dataset
    return embedded_dataset
