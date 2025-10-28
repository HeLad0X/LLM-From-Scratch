import re
import os
DATA_PATH = os.path.join(os.getcwd(), 'data/')

def get_text():
    all_paths = []
    for root, dirs, files in os.walk(DATA_PATH):
        for name in dirs + files:
            path = os.path.join(root, name)
            all_paths.append(path)

    # print all collected paths
    for p in all_paths:
        print(p)

get_text()