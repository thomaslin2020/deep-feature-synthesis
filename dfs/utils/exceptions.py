class UnknownFeature(Exception):
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        Exception.__init__(self, *args)


class UnusedPrimitiveWarning(UserWarning):
    pass
