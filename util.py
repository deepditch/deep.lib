from collections import Iterable

def listify(x, y):
    if not isinstance(x, Iterable): x=[x]
    n = y if type(y)==int else len(y)
    if len(x)==1: x = x * n
    return x