"""
Common paths.
!!! DO NOT CHANGE THIS FILE'S LOCATION !!!
"""

import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR_NAME = 'raw'
DATA_FILENAME = 'data.csv'
# DB_DIR = ROOT_DIR / "dataset1"  # The initial small dataset
# DB_DIR = ROOT_DIR / "dataset1 and 2021-11-06 19-51-29"  # The initial small dataset merged with 2021-11-06 19-51-29
# DB_DIR = ROOT_DIR / "dataset2"  # Big dataset, all-level comments.
# DB_DIR = ROOT_DIR / "dataset3"  # Big dataset, up to 2nd-level comments.
DB_DIR = ROOT_DIR / "dataset3 and 2021-11-06 19-51-29"  # Big dataset, up to 2nd-level comments merged with 2021-11-06 19-51-29
EXPECTED_SIZE_DICT = {ROOT_DIR / 'dataset1': 10204, ROOT_DIR / 'dataset2': 126782, ROOT_DIR / 'dataset3': 164452,
                      ROOT_DIR / 'dataset1 and 2021-11-06 19-51-29': 10607,
                      ROOT_DIR / 'dataset3 and 2021-11-06 19-51-29': 164855}
CREDENTIALS_DIR = ROOT_DIR / 'cred'
EXTRA_DIR = ROOT_DIR / 'extra'
FUNCTION_WORDS_LST_FILE = 'fw.txt'
EMOTICONS_LST_FILE = 'emoticons.txt'

EXPLICIT_DS = lambda name: ROOT_DIR / name / RAW_DIR_NAME / DATA_FILENAME
RAW_DATA_PATH = DB_DIR / RAW_DIR_NAME / DATA_FILENAME
IMAGES_DIR = DB_DIR / "images"
RESULTS_DIR = DB_DIR / "results"
