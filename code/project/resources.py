from os.path import dirname
from project.utils.files import resolve, get_files, from_lines

################################################################################
# Paths to all training resources, configuration default, 
# resource locations, etc.
################################################################################

text_base         = resolve(dirname(__file__), '..', '..', 'data', 'text')
text_train_data   = resolve(text_base, 'train_data')
text_train_labels = resolve(text_base, 'train_labels.txt')
text_test_data    = resolve(text_base, 'test_data')


def train_data_files():
    """
    @returns: [str]
    """
    return get_files(text_train_data)


def train_data_labels():
    """
    @returns: [str]
    """
    def process(filename, label, score):
        return (filename, int(label), float(score))
    return map(lambda line: process(*(line.split())), from_lines(text_train_labels))


def test_data_files():
    """
    @returns: [str]
    """
    return get_files(text_test_data)
