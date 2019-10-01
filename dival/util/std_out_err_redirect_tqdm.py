# -*- coding: utf-8 -*-
"""Redirecting writing to ``tqdm`` (the progressbar).

Taken from `tqdm docs <https://github.com/tqdm/tqdm#redirecting-writing>`_.
"""
import contextlib
import sys
from tqdm import tqdm


class DummyTqdmFile(object):
    """Dummy file-like that will write to ``tqdm``.

    Attributes
    ----------
    file : file object
        File to write to using :func:`tqdm.write`.
    """
    file = None

    def __init__(self, file):
        """
        Parameters
        ----------
        file : file object
            File to write to using :func:`tqdm.write`.
        """
        self.file = file

    def write(self, x):
        """
        Call ``tqdm.write``.

        Parameters
        ----------
        x : str
            Text.
        """
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        """
        Flush :attr:`file`.
        """
        return getattr(self.file, "flush", lambda: None)()


@contextlib.contextmanager
def std_out_err_redirect_tqdm(std_out=None):
    """
    Context manager that redirects :obj:`sys.stdout` and :obj:`sys.stderr` to
    ``tqdm``.

    std_out : `None` or file object
        If `None` is passed, :obj:`sys.stdout` and :obj:`sys.stderr` are
        redirected to ``tqdm``.
        If a file object is passed, it is yielded immediately and no
        redirection is done. This is useful for nested calls:

        .. code-block:: python

            def func(std_out=None)
                with std_out_err_redirect_tqdm(std_out) as std_out:
                    for i in tqdm(range(7), file=std_out):
                        print(i)

            with std_out_err_redirect_tqdm() as std_out:
                func(std_out=std_out)

    Yields
    ------
    std_out : file object
        The original :obj:`sys.stdout` if ``std_out=None`` is passed, else
        `std_out`.
    """
    if std_out is not None:
        yield std_out
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
