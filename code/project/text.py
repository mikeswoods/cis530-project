from project import resources 
from project import utils

################################################################################
# Text processing code lives here
################################################################################

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
