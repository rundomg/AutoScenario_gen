"""Small compatibility subset for LMDrive's unused EasyDict import."""


class EasyDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
