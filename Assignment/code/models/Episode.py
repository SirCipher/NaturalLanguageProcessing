import tensorflow as tf
import utils


class Episode(object):

    def __init__(self, flags, facts, question_state):
        self.flags = flags
        self.facts = facts
        self.max_sentence_size = tf.shape(facts)[1]
        self.question_state = question_state

        # convert question from shape [BATCH_SIZE, CELL_SIZE] to [BATCH_SIZE, MAX_SENTENCE_SIZE, CELL_SIZE]
        self.question_tiled = tf.tile(tf.reshape(question_state, [-1, 1, self.flags.cell_size]),
                                      [1, self.max_sentence_size, 1])

        # initialize weights for attention
        self.w1 = utils.weight_variable([1, 4 * self.flags.cell_size, self.flags.hidden_size], "episode_weight_1")
        self.b1 = utils.bias_variable([1, 1, self.flags.hidden_size], "episode_bias_1")

        self.w2 = utils.weight_variable([1, self.flags.hidden_size, 1], "episode_weight_2")
        self.b2 = utils.bias_variable([1, 1, 1], "episode_bias_2")

    def new(self, memory, reuse):
        with tf.variable_scope("attention", reuse=reuse):
            question_tiled = tf.tile(tf.reshape(self.question_state, [-1, 1, self.flags.cell_size]),
                                     [1, self.max_sentence_size, 1])

            # extend and tile memory to [BATCH_SIZE, MAX_SENTENCE_SIZE, CELL_SIZE] shape
            memory = tf.tile(tf.reshape(memory, [-1, 1, self.flags.cell_size]), [1, self.max_sentence_size, 1])

            # interactions between facts, memory and question as described in paper
            attending = tf.concat([self.facts * memory, self.facts * question_tiled,
                                   tf.abs(self.facts - memory), tf.abs(self.facts - question_tiled)], 2)

            # get current batch size
            batch = tf.shape(attending)[0]

            # first fully connected layer
            h1 = tf.matmul(attending, tf.tile(self.w1, [batch, 1, 1]))
            h1 = h1 + tf.tile(self.b1, [batch, self.max_sentence_size, 1])
            h1 = tf.nn.tanh(h1)

            # second and final fully connected layer
            h2 = tf.matmul(h1, tf.tile(self.w2, [batch, 1, 1]))
            h2 = h2 + tf.tile(self.b2, [batch, self.max_sentence_size, 1])

            # returns softmax so attention scores are from 0 to 1
            return tf.nn.softmax(h2, 1)
