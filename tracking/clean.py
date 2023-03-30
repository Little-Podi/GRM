import os
import shutil


def rm_pycache(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir == '__pycache__':
                shutil.rmtree(os.path.join(root, dir))


if __name__ == '__main__':
    rm_pycache('.')
