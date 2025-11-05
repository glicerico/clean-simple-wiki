"""Data manipulation utility functions."""

from typing import List, Any, Iterator


def batched(xs: List[Any], n: int) -> Iterator[List[Any]]:
    """Yield successive n-sized chunks from a list.
    
    Args:
        xs: List to batch
        n: Batch size
        
    Yields:
        Batches of size n (last batch may be smaller)
    """
    for i in range(0, len(xs), n):
        yield xs[i:i+n]
