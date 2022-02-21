import torch  
import librosa   

def handler_same(x):
    """return the same thing as the content of the file

    Args:
        fname (_type_): str, file path

    Returns:
        _type_: [samples, ]
    """
    return x, x

