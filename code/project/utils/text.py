from os.path import basename

from string import punctuation
from nltk.corpus import stopwords

from project import resources 
from project import utils

################################################################################
# Text processing code lives here
################################################################################

STOP_WORDS =   set(stopwords.words('english')) \
             | set(punctuation) \
             | set(['--', "''", '""'])


def read_file(filetype, index=None):
    """
    Utility function to fetch a listing of the file or files that constitute 
    the training and test data set. 

    @param str filetype One of 'train' or 'test'
    @param int|None index If given, the file in the file list at the given index
    will be read from, otherwise all files will be read from
    @returns str|[str]
    """
    filelist = []

    if filetype == 'train':
        filelist = resources.train_data_files()
    elif filetype == 'test':
        filelist = resources.test_data_files()
    else:
        raise ValueError('type must be "train" or "test"; got ' + filetype)

    return map(utils.files.read_file, filelist) \
           if index is None else utils.files.read_file(filelist[index])

def filename_to_id(filenames):
    """
    Given a filename or filenames, this function converts the 
    filename to an observation ID (the basename of the file) or list of 
    observation IDs
    """
    process = lambda f: basename(f.strip()) 

    if isinstance(filenames, list):
        return map(process, filenames)
    else:   
        return process(filenames)


def is_number(s):
    """
    Test if the input is a number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_junk_token(token):
    """
    Tests if the given token is considered "junk"
    """
    return token in STOP_WORDS


def strip_junk_tokens(tokens):
    """
    Returns all tokens in tokens that are not in STOP_WORDS
    """
    return [token for token in tokens if token not in STOP_WORDS]

