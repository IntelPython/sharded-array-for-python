import sharpy as sp


def pytest_configure(config):
    sp.init()


def pytest_unconfigure(config):
    sp.fini()
