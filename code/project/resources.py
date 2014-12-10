from os.path import dirname
from project.utils.files import resolve, get_files, from_lines


################################################################################
# Paths to all training resources, configuration default, 
# resource locations, etc.
################################################################################


text_base            = resolve(dirname(__file__), '..', '..', 'data', 'text')
CoreNLP_base         = resolve(dirname(__file__), '..', '..', 'data', 'CoreNLP')

text_train_data      = resolve(text_base, 'train_data')
text_train_labels    = resolve(text_base, 'train_labels.txt')
text_test_data       = resolve(text_base, 'test_data')

CoreNLP_train_data   = resolve(CoreNLP_base, 'train_data')
CoreNLP_train_labels = resolve(CoreNLP_base, 'train_labels.txt')
CoreNLP_test_data    = resolve(CoreNLP_base, 'test_data')


def train_data_files(type='text'):
    """
    @returns: [str]
    """
    if type not in ('text', 'CoreNLP'):
        raise ValueError('type must be one of: "text", "CoreNLP"')
    return get_files(globals()["{}_train_data".format(type)])


def train_data_labels():
    """
    Returns a dict where the key is the training file label and the value is
    the classification label (-1,1)
    @returns: {str:int}
    """
    def process(filename, label, score):
        return (filename, int(label))
    return dict(map(lambda line: process(*(line.split())), from_lines(text_train_labels)))


def train_data_labels_with_overlap_score():
    """
    Returns a list where each entry is a 3-tuple, where the first entry is 
    the training file label, the second is the classification label (-1,1), and
    the third is the 

    @returns: [(str, int, float)]
    """
    def process(filename, label, score):
        return (filename, int(label), float(score))
    return map(lambda line: process(*(line.split())), from_lines(text_train_labels))


def test_data_files(type='text'):
    """
    @returns: [str]
    """
    if type not in ('text', 'CoreNLP'):
        raise ValueError('type must be one of: "text", "CoreNLP"')
    return get_files(globals()["{}_test_data".format(type)])
