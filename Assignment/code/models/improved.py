import random
import numpy as np
import tensorflow as tf
import utils
from models import Episode
from models.Episode import Episode
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
Over the preceding weeks I have been very unwell and as such, I have used an existing implementation of the 
Dynamic Memory Networks for Visual and Textual Question Answering research paper. This implementation has been 
heavily modified as some of the existing parts were not inline with the research paper itself and these have been 
adjusted accordingly. 

A large part of the code had been written before this choice was made, however, I have added in notes where I did use
code from the GitHub repository. This code was largely used for batching, GloVe file parsing and the displaying of 
attention scores.

The original code has been refactored such that the network code is contained within this class and all other functions
that were required are in the utils.py class. 

Original code: https://github.com/mbilos/Dynamic-Memory-Networks
'''


class DMN:
    def __init__(self, flags):
        self.flags = flags

        self.adam_optimizer = None
        self.train_op = None
        self.attention = None
        self.predication_labels = None
        self.lr = None

        self.summary_batch = None
        self.writer = None
        self.writer_train = None
        self.writer_val = None
        self.is_training = None

        self.input_sentence = None,
        self.len_sentence = None,
        self.input_question = None,
        self.len_question = None,
        self.answer_labels = None,
        self.is_training = None,

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        print("Initialising network...")
        self.build()
        print("Initialised network...")

    # From github repository
    def get_batch(self, split='train'):
        stories_train, stories_valid, stories_test, embedding = utils.get_data(self.flags.task)

        # choose data source based on split, if train take random sample
        if split == 'train':
            batch = [stories_train[i] for i in random.sample(range(len(stories_train)), self.flags.batch_size)]
        elif split == 'valid':
            batch = stories_valid
        else:
            batch = stories_test

        # get text embeddings from batch
        text = [x['text_vec'] for x in batch]

        # prepare sentence by applying positional encoding to each sentence
        # get sentence lengths and add padding to each sentence so they are same size
        _sentence = [[np.sum(s * utils.positional_encoding(np.shape(s)), axis=0) for s in t] for t in text]
        _sentence_length = [len(x) for x in _sentence]
        _sentence = utils.add_padding(_sentence)

        # extract question sequences, get lengths and add padding
        _question = [x['question_vec'] for x in batch]
        _question_length = [len(x) for x in _question]
        _question = utils.add_padding(_question)

        # get output word labels
        val_label = np.array([x['answer_vec'] for x in batch])

        # store everything into feed dictionary (input to DMN)
        val_feed = {
            self.input_sentence: _sentence,
            self.len_sentence: _sentence_length,
            self.input_question: _question,
            self.len_question: _question_length,
            self.answer_labels: val_label,
            # Added is_training so that global_steps increases
            self.is_training: split == 'train'
        }

        # if training return only feed, else return also label for evaluation
        if split == 'train':
            return val_feed
        elif split == 'valid':
            return val_feed, val_label
        else:
            return val_feed, val_label, batch

    def validate(self, sess, feed, i):
        summary = sess.run(self.summary_batch, feed)
        self.writer_train.add_summary(summary, i)

        val_feed, val_label = self.get_batch('valid')
        summary_val, val_pred = sess.run([self.summary_batch, self.predication_labels], val_feed)
        self.writer_val.add_summary(summary_val, i)

        utils.print_accuracy(val_label, val_pred)

    def build(self):
        with tf.variable_scope('input_fusion'):
            self.input_sentence = tf.placeholder(tf.float32, [None, None, self.flags.embedding_size], name='sentence')
            self.len_sentence = tf.placeholder(tf.int32, [None], 'len_sentence')

            forward = tf.contrib.rnn.GRUCell(self.flags.cell_size)
            backward = tf.contrib.rnn.GRUCell(self.flags.cell_size)
            facts, _ = tf.nn.bidirectional_dynamic_rnn(forward, backward, self.input_sentence, dtype=np.float32,
                                                       sequence_length=self.len_sentence)
            facts = tf.reduce_sum(tf.stack(facts), axis=0)
            self.is_training = tf.placeholder(tf.bool)

        with tf.variable_scope('input') as scope:
            self.input_question = tf.placeholder(tf.float32, [None, None, self.flags.embedding_size], name='question')
            self.len_question = tf.placeholder(tf.int32, [None], name='len_question')

            cell = tf.contrib.rnn.GRUCell(self.flags.cell_size)
            _, question_state = tf.nn.dynamic_rnn(cell, self.input_question, sequence_length=self.len_question,
                                                  dtype=tf.float32,
                                                  scope=scope)

        self.answer_labels = tf.placeholder(tf.int32, [None], name='answer')

        # Adopted from github repository
        with tf.variable_scope('episodic', reuse=tf.AUTO_REUSE):
            max_sentence_size = tf.shape(facts)[1]
            memory = question_state

            self.attention = []

            for p in range(self.flags.episodes):
                episode = Episode(self.flags, facts, question_state)

                # get attention from memory (and question and facts which are defined before)
                self.attention.append(episode.new(memory, bool(p)))

                # initialize GRU cell for RNN which returns final episodic memory
                gru = tf.contrib.rnn.GRUCell(num_units=self.flags.cell_size, reuse=bool(p))

                # run loop for length of longest sentence
                def valid(state, i):
                    return tf.less(i, max_sentence_size)

                # in each step update state with attention (how much to keep from past)
                def body(state, i):
                    state = self.attention[-1][:, i, :] * gru(facts[:, i, :], state)[0] + (
                            1 - self.attention[-1][:, i, :]) * state
                    return state, i + 1

                # get episode by applying GRU attention
                episode = tf.while_loop(cond=valid, body=body, loop_vars=[memory, 0])[0]

                # initialize weights for updating between episodes
                episode_weight = utils.weight_variable([3 * self.flags.cell_size, self.flags.cell_size],
                                                       "episode_" + str(p) + "_weight_1")

                episode_bias = utils.bias_variable([self.flags.cell_size], "episode_" + str(p) + "_bias_1")

                m = tf.concat([memory, episode, question_state], 1)
                m = tf.matmul(m, episode_weight) + episode_bias

                # Add batching
                m = utils.batch_norm(m, self.is_training)

                # new memory state is dependent on previous, current memory and question
                memory = tf.nn.relu(m)

        with tf.variable_scope('answer_module'):
            # input into answer module is concatenation of last memory and question state
            concat = tf.concat([memory, question_state], 1)
            concat = tf.nn.dropout(concat, rate=1 - self.flags.dropout_rate)

            weight = utils.weight_variable([self.flags.cell_size * 2, self.flags.vocabulary_size], "weight")
            utils.variable_summaries(weight)

            bias = utils.bias_variable([self.flags.vocabulary_size], "bias")
            utils.variable_summaries(bias)

            # outputs predictions
            pred = tf.matmul(concat, weight) + bias
            self.predication_labels = tf.argmax(pred, axis=1, name='prediction_label')

            softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_labels, logits=pred)

        with tf.variable_scope('loss'):
            # Must be added to collection so L2 loss is calculated correctly
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weight))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(bias))

            # Loss only calculated on weights, not bias as per section 6.2 of research paper
            regularisation_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
            total_loss = tf.reduce_mean(softmax) + regularisation_loss * self.flags.l2_regularisation_loss

        with tf.variable_scope('optimisation'):
            self.lr = tf.minimum(self.flags.learning_rate,
                                 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))

            self.adam_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-8)
            self.train_op = self.adam_optimizer.minimize(total_loss, global_step=self.global_step)

        tf.summary.scalar("loss", total_loss)
        self.summary_batch = tf.summary.merge_all()

    # Adopted from github repository
    def print_results(self, sess):
        test_feed, test_label, test_batch = self.get_batch('test')
        summary_val, test_pred, attentions = sess.run([self.summary_batch, self.predication_labels, self.attention],
                                                      test_feed)
        self.writer_val.add_summary(summary_val, self.flags.epochs)

        print("\n\n\nTEST SET\n")
        utils.print_accuracy(test_label, test_pred)

        test_text = [x['text'] for x in test_batch]
        test_question = [x['question'] for x in test_batch]

        attentions = np.transpose(np.squeeze(attentions), [1, 2, 0])
        sentence_attentions = [list(zip(x, y)) for x, y in zip(test_text, attentions)]

        sample_indexes = random.sample(range(len(test_text)), 10)

        j = 0

        # Adapted graph printing from: https://www.oreilly.com/ideas/question-answering-with-tensorflow
        for i in sample_indexes:
            atts = []
            j += 1

            print('Question:', test_question[i])
            for item in sentence_attentions[i]:
                print(item[0], '\t', item[1])
                atts.append(item[1])

            print('\n\n')

            plt.xticks(range(self.flags.episodes, 0, -1))
            plt.yticks(range(1, len(atts), 1))
            plt.xlabel("Episode")
            plt.ylabel("Question " + str(j))
            plt.pcolor(atts, cmap=plt.cm.BuGn, alpha=0.7)
            plt.show()

    def train(self, sess):
        sess.run(tf.global_variables_initializer())
        self.writer.add_graph(sess.graph)

        global_step = max(sess.run(self.global_step), 1)

        for _ in tqdm(range(global_step, self.flags.epochs + 1)):
            global_step = sess.run(self.global_step) + 1

            sess.run(self.train_op, self.get_batch('train'))

            # validation every 20 steps
            if global_step != 0 and global_step % 20 == 0:
                self.validate(sess, self.get_batch('train'), global_step)

    def init_writers(self, sess):
        """
        Initialise TensorBoard summary writers
        :param sess:
        :return:
        """

        self.summary_batch = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.flags.log_dir)
        self.writer.add_graph(sess.graph)

        logging_base_path = self.flags.log_dir + '/'

        self.writer_train = tf.summary.FileWriter(logging_base_path + 'training')
        self.writer_val = tf.summary.FileWriter(logging_base_path + 'validation')
