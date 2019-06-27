import tensorflow as tf
import os
from models.improved import DMN
import utils


def main(_):
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    flags = tf.app.flags

    # Network hyperparameters

    flags.DEFINE_string('log_dir', 'logs', 'Summaries log directory')

    flags.DEFINE_integer('task', 1, 'Task number [1]')
    flags.DEFINE_integer('batch_size', 128, 'Batch number [128]')
    flags.DEFINE_integer('epochs', 1000, 'Number of epochs [1000]')
    flags.DEFINE_integer('cell_size', 128, 'Number of hidden states in RNN [128]')
    flags.DEFINE_integer('hidden_size', 80, 'Number of hidden states in attention [80]')
    flags.DEFINE_integer('episodes', 3, 'Number of episodes in episodic memory [3]')
    flags.DEFINE_integer('embedding_size', 50, 'Vector size representation of word')

    flags.DEFINE_float('dropout_rate', 0.9, 'The probability of discarding a sentence [0.9]')
    flags.DEFINE_float('l2_regularisation_loss', 0.00001, 'L2 regularisation for weight decay [0.00001]')
    flags.DEFINE_float('learning_rate', 0.01, 'Learning rate [0.01]')

    FLAGS = flags.FLAGS

    # load train, validation and test dataset and word embedding dictionary
    stories_train, stories_valid, stories_test, embedding = utils.get_data(FLAGS.task)

    # From github repository
    flags.DEFINE_integer('vocabulary_size', len(embedding), 'Number of unique words in the vocabulary')

    model = DMN(FLAGS)

    sess = tf.Session()
    model.init_writers(sess)

    init = tf.global_variables_initializer()
    sess.run(init)

    model.train(sess)
    model.print_results(sess)

    sess.close()


if __name__ == '__main__':
    tf.app.run()
