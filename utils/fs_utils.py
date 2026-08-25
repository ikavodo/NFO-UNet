import itertools
import os
from collections import namedtuple
from os import listdir
from os.path import join, isfile, isdir
from shutil import copyfile
from typing import List

import cv2
import numpy as np

File = namedtuple('File', 'name path')


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    # exist_ok=True, not a check-then-create: multiple processes race to create the same
    # dir (e.g. every parallel occlusion-sample worker shares out_dir/occlusion)
    os.makedirs(directory, exist_ok=True)


def copy_cv_img(img: np.ndarray, target_path: str):
    ensure_dir(target_path)
    cv2.imwrite(target_path, img)


def copy_file(source_path: str, target_path: str):
    ensure_dir(target_path)
    copyfile(source_path, target_path)


def search_files(directory: str, search_str: str = None, recursive: bool = False) -> List[File]:
    try:
        return [
            File(f, join(directory, f)) for f in listdir(directory)
            if isfile(join(directory, f)) and (search_str is None or search_str in f)
        ] if not recursive else [
            File(name, join(root, name)) for root, dirs, files in os.walk(directory, topdown=False)
            for name in files
            if search_str is None or search_str in name
        ]
    except FileNotFoundError:  # needed for config
        return []


def search_folders(directory: str, search_str: str = None, recursive: bool = True) -> List[str]:
    return [
        join(directory, f) for f in listdir(directory)
        if isdir(join(directory, f)) and (search_str is None or search_str in f)
    ] if not recursive else [
        join(root, name) for root, dirs, files in os.walk(directory, topdown=False)
        for name in dirs
        if search_str is None or search_str in name
    ]


def get_folder_for_file(file_path: str) -> str:
    if file_path.endswith("\\") or file_path.endswith("/"):
        return file_path.replace("\\", "/")
    split = [substring.split("/") for substring in file_path.split("\\")]
    flattened = [string for sublist in split for string in sublist]
    return "/".join(flattened).replace(flattened[-1], '')


def flatten_list(in_list: List) -> List:
    return list(itertools.chain(*in_list))
