import os
import ddptensor as dt

ddpt_cw = bool(int(os.environ.get('DDPT_CW')))

def pytest_configure(config):
    print(f"DDPT_CW={bool(ddpt_cw)}")
    dt.init(ddpt_cw)

def pytest_unconfigure(config):
    dt.fini()
