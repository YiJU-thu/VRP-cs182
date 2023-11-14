import numpy as np

class cstring:
    def blue(text): return "\033[94m{}\033[00m" .format(text)
    def green(text): return "\033[92m{}\033[00m" .format(text)
    def red(text): return "\033[91m{}\033[00m" .format(text)
    def yellow(text): return "\033[93m{}\033[00m" .format(text)

class cprint:
    def blue(text): print(cstring.blue(text))
    def green(text): print(cstring.green(text))
    def red(text): print(cstring.red(text))
    def yellow(text): print(cstring.yellow(text))

def safe_divide_np(a, b):
    # what should the shape be like?
    return np.divide(a, b, out=np.zeros_like(...), where=b!=0)