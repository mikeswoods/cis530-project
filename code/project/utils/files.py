import re
import tempfile
import os
import sys
from os import listdir, remove, walk, getcwd, chdir
from os.path import dirname, join, realpath, expanduser, isdir, isfile
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
    return None if len(parts) == 0 \
                else realpath(reduce(join, parts[1:], expanduser(parts[0])))


def get_files(directory):
    """
    Lists all immediate files in the given directory

    @param str directory The directory to list the files of
    @returns [str]
    """
    return [join(directory, d) for d in listdir(directory) \
                               if isfile(join(directory, d))]


def get_sub_dirs(directory):
    """
    Lists all immediate subdirectories in the given directory

    @param str directory The directory to list the subdirectories of
    @returns [str]
    """
    return [join(directory, d) for d in listdir(directory) \
                               if isdir(join(directory, d))]


def get_all_files(directory):
    """
    Returns a listing of files to arbitrary depth within the given directory

    @param str directory The directory to list
    @returns [str] A list of relative file paths for all files in the 
    input directory
    """
    listing = []
    for (root,_,files) in walk(directory):
        for f in files:
            listing.append(join(root,f))
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
def cd(directory):
    """
    context manager to switch to another directory temporarily

    Usage:

    with switch_dir("/switch/to/this/dir") as old_dir:
        ...
    """
    pwd = getcwd()
    chdir(directory)
    yield pwd
    chdir(pwd)


def ls(file_or_directory):
    """
    If given a directory, a listing of the directory will be returned. If 
    file_or_directory is a file name, then a singleton list with 
    file_or_directory will be returned

    @returns [str]
    """
    if isfile(file_or_directory):
        return [file_or_directory]
    else:
        return sorted([join(file_or_directory, f) for f in listdir(file_or_directory)])


def from_lines(input_file):
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


def to_lines(items):
    """
    Given a list of strings, this function produces a single string separated
    by newline characters
    """
    return "\n".join(items)+"\n"


def read(from_file):
    """
    Reads the contents of from_file to a string, returning the results

    @param str|file from_file
    @returns str
    """
    if isinstance(from_file, file):
        return from_file.read()
    else:
        with open(from_file, 'r') as f:
            contents = f.read()
            f.close()
            return contents


def write(to_file, contents, append=True):
    """
    Writes contents to the file to_file, appending if append=True

    @param str|file to_file
    @param str contents
    @param bool append=True
    """
    if isinstance(to_file, file):
        to_file.write(contents)
    else:
        with open(to_file, 'a' if append else 'w+') as f:
            f.write(contents)
            f.close()


def write_temp_file(contents, directory=None):
    """
    Writes contents to a a temporary file optionally created in directory dir,
    returning the name of the temp file
    """
    (_,name) = tempfile.mkstemp(dir=directory)
    with open(name, 'w') as f:
        f.write(contents)
        f.close()
    return name


@contextmanager
def write_temp(contents):
    """
    Like write_temp_file, but works in the context of the context manager,
    deleting the temporary file at the end of the block
    """
    temp_file = write_temp_file(contents)
    yield temp_file
    remove(temp_file)
