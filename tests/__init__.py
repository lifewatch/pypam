# Basic support for unit tests.

from os import getenv
import pytest


def with_plots() -> bool:
    """
    Return True if the env var PYPAM_TEST_NO_PLOTS is not set.
    """
    return getenv("PYPAM_TEST_NO_PLOTS") is None


def skip_unless_with_plots():
    """
    Decorator to skip a test if the env var PYPAM_TEST_NO_PLOTS is set.
    """

    def decorator(test_item):
        if with_plots():
            return test_item
        reason = "PYPAM_TEST_NO_PLOTS is set"
        return pytest.mark.skip(reason)(test_item)

    return decorator
