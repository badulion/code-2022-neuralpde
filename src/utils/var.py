from typing import Dict, Any
import hashlib
import json
import numpy as np

def dict_hash(dictionary: Dict[str, Any], *args) -> str:
    # if multiple dictionaries are passed join them first
    for d in args:
        dictionary.update(d)

    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()