import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_addoption(parser):
    parser.addoption("--weights", default="weights", help="weights folder (see README.md)")
