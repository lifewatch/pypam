# Basic support for unit tests.

from os import getenv
import unittest


def with_plots() -> bool:
    return getenv('PYPAM_TEST_NO_PLOTS') is None


def skip_unless_with_plots():
    def decorator(test_item):
        if with_plots():
            return test_item
        else:
            reason = 'PYPAM_TEST_NO_PLOTS is set'
            return unittest.skip(reason)(test_item)
    return decorator

