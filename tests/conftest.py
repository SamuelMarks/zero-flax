import pytest
import sys
import os


@pytest.fixture(autouse=True)
def dummy_config():
    # Placeholder to keep autouse fixture mechanism if needed later
    yield
