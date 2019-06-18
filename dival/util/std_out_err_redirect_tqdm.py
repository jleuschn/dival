# -*- coding: utf-8 -*-
"""Redirecting writing to tqdm (the progressbar).

Taken from `tqdm docs <https://github.com/tqdm/tqdm#redirecting-writing>`_.
"""
import contextlib
import sys
from tqdm import tqdm


class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


@contextlib.contextmanager
def std_out_err_redirect_tqdm(yield_=None):
    if yield_ is not None:
        yield yield_
    else:
        orig_out_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
            yield orig_out_err[0]
        # Relay exceptions
        except Exception as exc:
            raise exc
        # Always restore sys.stdout/err if necessary
        finally:
            sys.stdout, sys.stderr = orig_out_err
