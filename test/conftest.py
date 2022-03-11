import os
import ddptensor as dt

ddpt_cw = os.environ.get('DDPT_CW')
ddpt_cw = False if ddpt_cw is None else bool(int(ddpt_cw))

def pytest_configure(config):
    dt.init(ddpt_cw)

def pytest_unconfigure(config):
    dt.fini()
    #pytest.exit("", returncode=0)
