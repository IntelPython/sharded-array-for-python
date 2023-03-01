import os
import ddptensor as dt


def pytest_configure(config):
    dt.init()


def pytest_unconfigure(config):
    dt.fini()
