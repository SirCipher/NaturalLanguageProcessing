import os
import re
from copy import deepcopy
import numpy as np
import tensorflow as tf


def get_tasks(task=None, split=None):
    """
    Args:
        task: (optional) babi task id (int), defaults to all tasks
        split: (optional) data split (string), can be train, test or valid

    Returns: list of babi tasks as list of objects with properties:
        filename: full path to file (*.txt)
        task: babi task id (1 to 20)
        split: data split (train, test or valid)
    """

    DIR = './data/babi/en-valid-10k/'
    FILE_REGEX = re.compile(r'qa(\d+)_(test|train|valid)\.txt')

    tasks = []

    for file in os.listdir(DIR):
        match = re.search(FILE_REGEX, file)
        if match:
            tasks.append({
                'filename': DIR + match.group(0),
                'task': int(match.group(1)),
                'split': match.group(2)})

    if task:
        tasks = [x for x in tasks if x['task'] == task]
    if split:
        tasks = [x for x in tasks if x['split'] == split]

    return tasks


def get_glove_embedding():
    """
    Returns: dict
        key: (string) word from babi vocabulary
        value: (numpy float array) corresponding vector in glove embeddings
    """

    vocabulary = set()
    remove = re.compile(r'[^a-zA-Z]+')

    # get filenames of all tasks
    all_tasks = [l['filename'] for s in [get_tasks(x) for x in range(20)] for l in s]

    for filename in all_tasks:
        # get unique words from file and add to vocabulary
        with open(filename, 'r') as f:
            lines = f.readlines()
            unique = set(re.sub(remove, ' ', ' '.join(lines).lower()).split(' '))
            vocabulary |= unique

    embedding = dict()

    with open('./data/glove/glove.6B.50d.txt', 'r', encoding='utf8') as f:
        for line in f:
            # read line in glove file: first word - word, rest - embedding
            word, vec = line.strip().split(' ', 1)

            # if word in babi vocabulary add to embedding
            if word in vocabulary:
                embedding.update({word: np.array(vec.split(' '), dtype=float)})

    # get all words without embedding and assign them random vector of same size
    rest = [x for x in vocabulary if x not in embedding.keys()]
    for word in rest:
        embedding.update({word: np.random.uniform(0.0, 1.0, (50,))})

    return embedding


"""
    initialize embedding and word_index: { word (string): index in embedding (int) }
"""
embedding = get_glove_embedding()
word_index = {x: i for i, x in enumerate(embedding.keys())}
attend_init = tf.random_normal_initializer(stddev=0.1)


# From https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
def weight_variable(shape, n):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_value=initial, name=n)


# From https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
def bias_variable(shape, n):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial, name=n)


def variable_summaries(var):
    """Attach summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def get_data(tasks=None):
    """
    Args:
        tasks: (list or int) babi ids
    Returns:
        list: train dataset containing babi stories as list of dictionaries:
            text: (list) of sentences (string)
            question: sentence (string)
            answer: single word (string)
            text_vec: (list) of sentences (list) containing
                      word embeddings (numpy float array)
            question_vec: (list) of words in sentence (numpy float array)
            answer_vec: (int) index of word in embedding
        list: test dataset (same as above)
        list: validation dataset (same as above)
        dict: embedding dictionary { word (string) : vector (numpy float array) }
    """

    # if tasks number convert to list
    if type(tasks) == int:
        task_ids = [tasks]

    data = dict()

    for split in ['train', 'valid', 'test']:
        # load all tasks for task ids and current split
        tasks = [get_tasks(x, split) for x in task_ids]
        tasks = [l for s in tasks for l in s]

        stories = []

        for task in tasks:
            # read content of file for current task
            with open(task['filename']) as f:
                lines = f.readlines()

            """
            lines example:
                '1 Mary got the milk there.\n'
                '2 John moved to the bedroom.\n'
                '3 Is John in the kitchen? \tno\t2\n'

            1. Get id from beggining of sentence. Id range from 1 to N and
               resets to 1 for new story.
            2. If id is 1 => new story, reset current.
            3. Remove id and \n and convert to lowercase.
            4. Look for question mark.
                4.1. If question => get question, answer and save story
                4.2. Else add line to story text
            """
            for line in lines:

                id = int(line[0:line.find(' ')])
                if id == 1:
                    current = {'text': [], 'question': '', 'answer': ''}

                line = line.strip().lower()
                line = line.replace('.', '')
                line = line[line.find(' ') + 1:]

                question_index = line.find('?')
                if question_index == -1:
                    current['text'].append(line)
                else:
                    current['question'] = line[:question_index]
                    current['answer'] = line[question_index:].split('\t')[1]
                    stories.append(deepcopy(current))

        for story in stories:
            # convert answer string to corresponding index in embedding
            story['answer_vec'] = word_index[story['answer']]

            # apply word2vec on question and text lists
            story['question_vec'] = [embedding[x] for x in story['question'].split(' ')]
            story['text_vec'] = [[embedding[w] for w in s.split(' ')] for s in story['text']]

        data[split] = stories

    return data['train'], data['valid'], data['test'], embedding


def add_padding(array):
    """
    Args:
        array: sequence of array of arrays with variable lengths
    Returns:
        sequence of padded items so that they have the same shape
        for input [ [[1], [2]], [[2], [3], [4]], [[5]] ]
        returns [ [[1], [2], [0]], [[2], [3], [4]], [[5], [0], [0]] ]
    """

    lengths = [np.shape(x)[0] for x in array]

    max_length = max(lengths)
    padding_lengths = [max_length - x for x in lengths]

    array = np.array([np.pad(x, ((0, p), (0, 0)), 'constant', constant_values=0)
                      for x, p in zip(array, padding_lengths)])

    return array


def positional_encoding(shape):
    """
    implements positional encoding as described in
    "End-To-End Memory Networks" (https://arxiv.org/pdf/1503.08895v5.pdf)
    """

    sentence_size, embedding_size = shape
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)

    ls = sentence_size + 1
    le = embedding_size + 1

    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)

    encoding = 1 + 4 * encoding / embedding_size / sentence_size

    return np.transpose(encoding)


def print_accuracy(label, prediction):
    total = np.shape(label)[0]
    correct = np.sum(label == prediction)

    print('\n-----------------------------------')
    print('Validation: correct:', correct, '\ttotal:', total)
    print('Validation: accuracy:', correct / total * 100, '%')
    print('-----------------------------------')


def batch_norm(x, is_training):
    """
    Performs batch normalisation with an exponential moving average over a provided memory

    :param x:
    :param is_training: Whether or not the current iteration is training
    :return: x normalised
    """

    # https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/34634291#34634291
    with tf.variable_scope('batch_normalisation'):
        inputs_shape = x.get_shape()
        axis = list(range(len(inputs_shape) - 1))
        param_shape = inputs_shape[-1:]

        batch_mean, batch_var = tf.nn.moments(x, axis)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        normed = tf.nn.batch_normalization(x, mean, var,
                                           tf.get_variable('beta', param_shape,
                                                           initializer=tf.constant_initializer(0.)),
                                           tf.get_variable('gamma', param_shape,
                                                           initializer=tf.constant_initializer(1.)),
                                           1e-3)
    return normed
