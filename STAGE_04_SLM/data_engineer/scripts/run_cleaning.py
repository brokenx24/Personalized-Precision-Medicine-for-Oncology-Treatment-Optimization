import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
cleaning_dir = os.path.join(os.path.dirname(script_dir), "cleaning")
sys.path.insert(0, cleaning_dir)

import clean_dataset

if __name__ == "__main__":
    clean_dataset.clean_entire_dataset()
