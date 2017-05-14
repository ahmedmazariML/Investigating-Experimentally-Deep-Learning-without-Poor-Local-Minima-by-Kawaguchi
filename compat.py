import sys


def callSuper(cls):

    if sys.version[0] == 2:
        super(cls).__init__()
    else:
        pass
