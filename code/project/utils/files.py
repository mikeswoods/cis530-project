import re
import tempfile
import os
import sys
from os.path import dirname, join, abspath
from itertools import chain
from contextlib import contextmanager

################################################################################
# File and file-system related utility functions
################################################################################

def resolve(*parts):
    """
    Example usage:

    base = "/Users/mike/src/school/cis530-project/code"
    resolve(base, '..', 'data', 'text') =>

        "/Users/mike/src/school/cis530-project/data/text"
    """
    return None if len(parts) == 0 else abspath(reduce(join, parts[1:], parts[0]))


def get_files(directory):
    """
    Lists all immediate files in the given directory

    @param str directory The directory to list the files of
    @returns [str]
    """
    return [os.path.join(directory, d) for d in os.listdir(directory) \
                                       if os.path.isfile(os.path.join(directory, d))]


def get_sub_dirs(directory):
    """
    Lists all immediate subdirectories in the given directory

    @param str directory The directory to list the subdirectories of
    @returns [str]
    """
    return [os.path.join(directory, d) for d in os.listdir(directory) \
                                       if os.path.isdir(os.path.join(directory, d))]


def get_all_files(directory):
    """
    Returns a listing of files to arbitrary depth within the given directory

    @param str directory The directory to list
    @returns [str] A list of relative file paths for all files in the 
    input directory
    """
    listing = []
    for (root,_,files) in os.walk(directory):
        for f in files:
            listing.append(os.path.join(root,f))
    return sorted(listing)


def read_file(input_file):
    """
    Reads the entire file (or what remains of it), returning the contents as
    a string

    @param str|file input_file The input file
    @returns str
    """
    if isinstance(input_file, file):
        return input_file.read()
    else:
        with open(input_file, 'r') as f:
            contents = f.read()
            f.close()
            return contents


@contextmanager
def switch_dir(directory):
    """
    context manager to switch to another directory temporarily

    Usage:

    with switch_dir("/switch/to/this/dir") as old_dir:
        ...
    """
    pwd = os.getcwd()
    os.chdir(directory)
    yield pwd
    os.chdir(pwd)


def ls(file_or_directory):
    """
    If given a directory, a listing of the directory will be returned. If 
    file_or_directory is a file name, then a singleton list with 
    file_or_directory will be returned

    @returns [str]
    """
    if os.path.isfile(file_or_directory):
        return [file_or_directory]
    else:
        return sorted([os.path.join(file_or_directory, f) for f in os.listdir(file_or_directory)])


def lines(input_file):
    """
    Returns all lines in the given file, where input_file is a filename or
    a file object

    @returns [str]
    """
    if isinstance(input_file, file):
        return map(lambda s: s.strip(), input_file.readlines())
    else:
        with open(input_file) as f:
            return map(lambda s: s.strip(), f.readlines())

