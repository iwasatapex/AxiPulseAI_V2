try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total
        def __iter__(self):
            return iter(self.iterable or ())
        def update(self, *_args, **_kwargs):
            return None
        def close(self):
            return None
        def set_description(self, *_args, **_kwargs):
            return None


class DummyProgress:
    def update(self, *_args, **_kwargs):
        return None
    def close(self):
        return None
    def set_description(self, *_args, **_kwargs):
        return None


_progress = None


def get_progress(*args, **kwargs):
    global _progress
    _progress = tqdm(*args, **kwargs)
    return _progress


def update(*args, **kwargs):
    if _progress is not None:
        return _progress.update(*args, **kwargs)


def close():
    if _progress is not None:
        return _progress.close()


def set_description(*args, **kwargs):
    if _progress is not None:
        return _progress.set_description(*args, **kwargs)
