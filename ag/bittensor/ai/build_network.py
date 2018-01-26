 #!/usr/bin/env python
"""Provides A GUI for specific Machine Learning Use-cases.

TF_Curses is a frontend for processing datasets into machine
learning models for use in predictive functions.
"""

import os, sys, datetime, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import collections
import random
import redis
import inflect
import tensorflow as tf
from tensorflow.contrib import ffmpeg
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import ag.logging as log
import database_interface as DB
log.set(log.DEBUG)

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2017, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.1"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Prototype"


def elapsed(sec):
    """I cant remember whos function this was.

    Sorry bro... ill look it up soon.
    """
    if sec < 60:
        return "{0:.2f} {1:}".format(sec, "sec")
    elif sec < (60*60):
        return "{0:.2f} {1:}".format(sec/60, "min")
    else:
        return "{0:.2f} {1:}".format(sec/(60*60), "hr")


class App(object):
    """Shit, there are no docstrings here.

    TODO: docstrings.
    """

    def __init__(self):
        """Gotta have docstrings."""
        root_path = '/home/eric/repos/pycharm_repos/TF_Curses'
        self.database = DB.Database(db=0)
        self.rev_dict = DB.Database(db=1)
        self.p = inflect.engine()
        self.n_input = 3
        self.vocab_size = 10000
        self.n_hidden = 512
        self.logs_path = '/pub/models/chatbot/'
        self.filename = 'alphagriffin'
        self.train_iters = int(5e2)
        self.converter = inflect.engine()
        self.sess = None
        self.iters = 50

    def main(self, args):
        """The Commandline exec of the chatbot network."""
        log.info("Beginning commandline exec of the chatbot network.")
        # get a text file... say lincoln.txt
        try:
            file_ = args[1]
            log.info("Recieving Documents: {}".format(file_))
        except:
            file_ = None
        if file_ is None:
            # file_ = "../text/sample.txt"
            file_ = "../text/lincoln.txt"
            log.debug("Going with sample: {}".format(file_))
        # Get some data
        log.info("Opening File: {}".format(file_))
        sample_set = self.read_data(file_)
        if not sample_set:
            return False
        # clean your data
        log.info("Building Database Dictionary")
        sample_set = self.build_redis_dataset(sample_set)
        if not sample_set:
            return False
        """
        # build a tensorboard
        log.info("build tensorflow network")
        log.debug("Trying to Load Old Model")
        if self.load_tf_model(self.logs_path):
            network = self.load_model_params()
        else:
            log.debug("Creating a New Model")
            network = self.build_network(sample_set)
        log.debug("Working with Final Layer {}".format(network.final_layer))

        # do some work
        msg = "Train Iters: {}".format(self.train_iters)
        log.info("Training Details:\n{}".format(msg))
        final_loss, average_acc = self.process_network(sample_set, network)
        """
        return True

    @staticmethod
    def save_npy(array, path):
        """Save out adjusted datafile with UNK tokens."""
        np.save(array, path)
        return True

    @staticmethod
    def read_data(fname):
        """Create numpy representation of text from path."""
        log.info("Processing text at path: {}".format(fname))
        if not os.path.isfile(fname):
            log.warn("{} is an invalid path".format(fname))
            return False
        class sample_text: pass
        with open(fname) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [content[i].split() for i in range(len(content))]
        content = np.array(content)
        sample_text.content = np.reshape(content, [-1, ])
        sample_text.len = sample_text.content.shape[0]
        sample_text.sample = sample_text.content[
            np.random.randint(0, sample_text.len)
            ]
        # this should be red if lower than x and green if above y.
        log.debug("Sample text is {} words long.".format(sample_text.len))
        log.info("Sample word from text:\n\t{}".format(sample_text.sample))
        log.info("File Loaded successfully.")
        return sample_text

    def get_text_file(self, file_, trunk=True):
        """Gotta have docstrings."""
        if not os.path.isfile(file_):
            log.warn("{} is an invalid path".format(file_))
            return False
        class sample_text(): pass
        msg = "Text Results:\n"
        with open(file_) as f:
            content = f.readlines()
        sample_text.all_content = content
        content = [x.strip() for x in content]
        print(len(content))
        content = [content[i].split() for i in range(len(content))]
        content = np.array(content)
        # print(content)
        sample_text.content = np.reshape(content, [-1, ])
        print(content.shape[:])
        sample_text.nwords = 0
        sample_text.word_set = []
        sample_text.token_to_vector = {}

        for this_line in sample_text.all_content:
            this_line = this_line.strip()
            words_in_line = this_line.split(' ')
            # TOKEN is the first word in the line
            token = words_in_line[0]
            # VECTOR is the line relitive to the token
            vector = words_in_line[1:] # this line minus the token
            # one hot encoded...
            sample_text.token_to_vector[token] = vector
            for word in words_in_line:
                sample_text.nwords += 1

        del sample_text.all_content # maybe ... save on some rams
        msg += "Num Words: {}\n".format(sample_text.nwords)
        sample_text.uwords = sorted(list(set(sample_text.word_set)))
        msg += "Num Unique Words: {}\n".format(len(sample_text.uwords))
        msg += "Num of Sentences or Unique Vectors: {}\n".format(len(sample_text.token_to_vector))

        log.debug(msg)
        return sample_text

    def build_dataset(self, sample_set):
        """Gotta have docstrings."""
        sample_set.count = collections.Counter(sample_set.content).most_common()
        sample_set.dictionary = dict()
        log.debug("adding word at pos. word[pos]")
        for word, _ in sample_set.count:
            cur_len = len(sample_set.dictionary)
            #log.debug("{} [{}]".format(word, cur_len))
            sample_set.dictionary[word] = cur_len
            sample_set.reverse_dictionary = dict(zip(sample_set.dictionary.values(),
                                                     sample_set.dictionary.keys()))
        sample_set.dict_len = len(sample_set.dictionary)
        log.debug("len of dictionary {}".format(sample_set.dict_len))
        return sample_set

    def build_redis_dataset(self, sample_set):
        """Use redis for managing a dynamic words library."""
        log.info("Accessing redis for text management.")
        start_time = time.time()
        sample_set.count = collections.Counter(
            sample_set.content).most_common()
        sample_set.dict = dict()
        sample_set.rev_dict = dict()
        sample_set.dict['UNK'] = 0
        sample_set.rev_dict[0] = 'UNK'
        sample_set.num_unk = 0
        # unk replacer
        unk_repacler = {'{}'.format(y): '{}'.format(x) for x, y in enumerate(
            sample_set.content)}
        sample_set.Unique_words = 0
        for i, _ in sample_set.count:
            sample_set.Unique_words += 1
        for index, (word, word_instances) in enumerate(sample_set.count):
            mesg = "Popularity Rank: {}, Word: {}: Num References: {}".format(
                index+1, word, word_instances
            )
            # the plan! Stop after 10k words. After that,
            # replace the words in the input text as UNK.
            if index <= self.vocab_size:
                # add each entry to the dict
                sample_set.dict[word] = index
                sample_set.rev_dict[index] = word
            else:
                # This takes time...
                loop_start = time.time()
                print("##################################")
                print("- Looking for {} instances of {}".format(
                    word_instances, word
                ))
                for j in range(word_instances):
                    word_place = unk_repacler[word]
                    sample_set.num_unk += 1
                    print("- Place in data to replace a word: {}".format(word_place))
                    print("-is {} this {}".format(word,
                                                  sample_set.content[
                                                     int(word_place)]))
                    if word in sample_set.content[int(word_place)]:
                        print("-- Yes.")
                        sample_set.content[int(word_place)] = 'UNK'
                        print('-# Changed to UNK')
                        # update the unk_repacler
                        if j+1 < word_instances:
                            print("-! Checking for other instances of word: {}.".format(word))
                            unk_repacler = {'{}'.format(y): '{}'.format(x) for x, y in enumerate(
                                sample_set.content)}
                    else:
                        print("-! Bogus word. {}".format(word))

                loop_end = time.time()
                # if index % 100 == 0:
                elap = loop_end - loop_start
                left = sample_set.Unique_words - (sample_set.num_unk + self.vocab_size)
                print("Word took {} to fix.".format(
                    elapsed(elap)
                ))
                print("######|| Have {} left to fix. should take {} ||######".format(
                    left, elapsed(elap * left)
                ))

                # lookup word in input and replace it with 'UNK'
            #    for i_content, content_word in enumerate(sample_set.content):
            #        if word in content_word:
            #            sample_set.content[i_content] = 'UNK'
            #            print("setting {} as UNK, num_unk: {}".format(word, sample_set.num_unk))
            #            sample_set.num_unk += 1
            # do redis next...
        end_time = time.time()
        print("process took {}secs to complete.".format(
            elapsed(end_time - start_time)
        ))
        print("sample_set.num_unk ", sample_set.num_unk)
        log.debug("Recounting Words in dataset: {}".format(
            len(sample_set.dict)
            ))
        log.info("Finished Creating Dictionaries from texts.")
        """
        try:
            word = float(word)
            word_ = self.p.number_to_words(int(word))
            sample_set.num_converted += 1
            sample_set.converted.append((word, word_))
            word = word_
        except:
            pass

        # FIX ME... SEARCH FOR OLD REFERENCE FIRST!
        # this is broken
        if self.database.read_data(word) is cur_len:
            pass
        else:
            self.database.write_data(str(word), int(cur_len))
            self.rev_dict.write_data(int(cur_len), str(word))
            sample_set.num_to_dict += 1
        #self.database.set_wordposition(str(word), int(cur_len))
        sample_set.dict_len += 1
        """
        # log.debug("len of dictionary {}".format(sample_set.dict_len))
        # log.debug("Num Converted words {}".format(sample_set.num_converted))
        # log.debug("Num  words added to database {}".format(sample_set.num_to_dict))
        # print(sample_set.converted)
        return sample_set

    def new_weights(self, shape):
        """This is standard tf_utils stuff."""
        return tf.Variable(tf.random_normal([self.n_hidden, shape]), name="weights")

    def new_biases(self, shape):
        """This is standard tf_utils stuff."""
        return tf.Variable(tf.random_normal([shape]), name="biases")

    def RNN(self, training_ops, dict_len, num_layers=4):
        """This is standard tf_utils stuff."""
        x = tf.reshape(training_ops.input_word, [-1, self.n_input])
        x = tf.split(x, self.n_input, 1)
        cells = []
        for i in range(num_layers):
            cells.append(tf.contrib.rnn.BasicLSTMCell(self.n_hidden))
        rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
        outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        weight = self.new_weights(dict_len)
        # You can add multiple embeddings. Here we add only one.
        w_embedding = training_ops.config.embeddings.add()
        w_embedding.tensor_name = weight.name
        # tf.summary.histogram('weights', weight)
        # biases...
        bias = self.new_biases(dict_len)
        b_embedding = training_ops.config.embeddings.add()
        b_embedding.tensor_name = bias.name
        # tf.summary.histogram('bias', bias)
        a_tensorflow_layer = tf.matmul(outputs[-1], weight) + bias
        return a_tensorflow_layer

    def build_network(self, sample_set):
        """This is standard tf_utils stuff."""
        class training_ops(): pass
        # RNN output node weights and biases
        # tf Graph input
        #with tf.variable_scope("AlphaGriffin.com") as scope:
        #    tf.summary.audio(self.audio_clip, sample_rate=22050)
        with tf.variable_scope("inputs") as scope:
            training_ops.global_step = tf.Variable(0, trainable=False, name='global_step')
            training_ops.learn_rate = tf.train.exponential_decay( 0.1,
                                                                  training_ops.global_step,
                                                                  .000005,
                                                                  0.87,
                                                                  staircase=True,
                                                                  name="Learn_decay"
                                                                  )
            tf.add_to_collection("global_step", training_ops.global_step)
            tf.add_to_collection("learn_rate", training_ops.learn_rate)
            tf.summary.scalar("global_step", training_ops.global_step)
            tf.summary.scalar("decay_rate", training_ops.learn_rate)
            tf.summary.histogram('decay_rate', training_ops.learn_rate)

            training_ops.input_word = tf.placeholder("float", [None, self.n_input, 1])
            training_ops.input_label = tf.placeholder("float", [None, sample_set.dict_len])
            tf.add_to_collection("input_word", training_ops.input_word)
            tf.add_to_collection("input_label", training_ops.input_label)

        # this is a setup for the tensorboard visualisations... use this when adding scalar histo ... this.
        training_ops.config = projector.ProjectorConfig()
        # tf.add_to_collection("config", training_ops.config)
        # embedding = tf.Variable(tf.pack(mnist.test.images[:FLAGS.max_steps], axis=0),
        #                        trainable=False,
        #                        name='embedding')

        training_ops.final_layer = self.RNN(training_ops, sample_set.dict_len, num_layers=3)
        tf.add_to_collection("final_layer", training_ops.final_layer)
        # Evaluate model
        training_ops.correct_pred = tf.equal(tf.argmax(training_ops.final_layer, 1), tf.argmax(training_ops.input_label, 1))
        training_ops.accuracy = tf.reduce_mean(tf.cast(training_ops.correct_pred, tf.float32))
        tf.summary.scalar("accuracy", training_ops.accuracy)
        tf.summary.histogram('accuracy', training_ops.accuracy)
        tf.add_to_collection("correct_pred", training_ops.correct_pred)
        tf.add_to_collection("accuracy", training_ops.accuracy)

        # Loss and optimizer
        training_ops.cost = tf.reduce_mean( \
                            tf.nn.softmax_cross_entropy_with_logits(logits=training_ops.final_layer,
                                                                    labels=training_ops.input_label))
        tf.summary.scalar("cost", training_ops.cost)
        # tf.summary.histogram('cost', training_ops.cost)
        tf.add_to_collection("cost", training_ops.cost)

        training_ops.optimizer = tf.train.RMSPropOptimizer(learning_rate=training_ops.learn_rate) \
                                                            .minimize(training_ops.cost, global_step=training_ops.global_step)

        tf.add_to_collection("optimizer", training_ops.optimizer)
        training_ops.init_op = tf.global_variables_initializer()
        tf.add_to_collection("init_op", training_ops.init_op)
        training_ops.saver = tf.train.Saver()
        self.saver = training_ops.saver
        # tf.add_to_collection("saver", training_ops.saver)
        training_ops.merged = tf.summary.merge_all()
        tf.add_to_collection("merged", training_ops.merged)
        self.sess = tf.InteractiveSession()
        self.sess.run(training_ops.init_op)
        return training_ops

    def process_network(self, sample_set, network, ):
        """This is standard tf_utils stuff."""

        # DEFINES!!
        training_data = sample_set.content

        # dictionary = sample_set.dictionary
        # reverse_dictionary = sample_set.reverse_dictionary
        n_input = self.n_input
        vocab_size = sample_set.dict_len

        # start here
        start_time = time.time()
        session = self.sess
        #if self.sess:
        #    session = self.sess
        #else:
        #    session = tf.Session()
        #session.run(network.init_op)
        writer = tf.summary.FileWriter(self.logs_path)
        _step = 0
        offset = random.randint(0, n_input + 1)
        end_offset = n_input + 1
        acc_total = 0
        loss_total = 0
        display_step = 10
        pred_msg = ' "{}" *returns* "{}" *vs* "{}"\n'
        msg = "step: {0:}, offset: {1:}, acc_total: {2:.2f}, loss_total: {3:.2f}"
        log.debug("Starting the Train Session:")
        # start by adding the whole graph to the Tboard
        writer.add_graph(session.graph)

        for i in range(self.train_iters):
            # Generate a minibatch. Add some randomness on selection process.
            if offset > (len(training_data) - end_offset):
                offset = random.randint(0, self.n_input + 1)
            symbols_in_keys = []
            for i in range(offset, offset + self.n_input):
                symbols_in_keys.append(self.database.read_data(str(training_data[i])))
            symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            # symbols_out_onehot[dictionary[str(training_data[offset + n_input])]] = 1.0
            one_hot = self.database.read_data(str(training_data[offset + n_input]))
            if one_hot is None:
                one_hot = 0
            symbols_out_onehot[int(one_hot)] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

            feed_dict = {network.input_word: symbols_in_keys,
                         network.input_label: symbols_out_onehot}

            try:
                _, acc, loss, onehot_pred, _step, summary = session.run([network.optimizer,
                                                                     network.accuracy,
                                                                     network.cost,
                                                                     network.final_layer,
                                                                     network.global_step,
                                                                     network.merged
                                                                     ],
                                                                    feed_dict=feed_dict)

                log.debug("###WORKING {}!!####".format(_step))
                # pool data results
                loss_total += loss
                acc_total += acc
                if i % 25 == 0:
                    # acc pool
                    print("###WORKING2!!####")
                    acc_total = (acc_total * 100) / display_step
                    loss_total = loss_total / display_step
                    # gather datas
                    try:
                        symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
                        symbols_out = training_data[offset + n_input]
                        symbols_out_pred = self.rev_dict.read_data(int(tf.argmax(onehot_pred, 1).eval(session=session)))
                        # do save actions
                        log.info("Saving the Train Session:\n{}\n{}".format(msg.format(_step,
                                                                                               offset,
                                                                                               acc_total,
                                                                                               loss_total),
                                                                                                pred_msg.format(symbols_in, symbols_out,
                                                                                                symbols_out_pred)))
                    except Exception as e:
                        log.warn("Bad Things are happening here: {}\n\t{}\n{}".format(elapsed(time.time() - start_time), e))
                        pass
                    # Save Functions
                    self.saver.save(session, self.logs_path + self.filename, global_step=network.global_step)
                    writer.add_summary(summary, global_step=_step)
                    # projector.visualize_embeddings(writer, network.config)
                    # reset the pooling counters
                    acc_total = 0
                    loss_total = 0
                # end of loop increments
                offset += (n_input + 1)
            except Exception as e:
                log.warn("BLowing it DUDE... {}\nError: {}".format(_step, e))
                pass
        # Save Functions
        self.saver.save(session, self.logs_path + self.filename, global_step=network.global_step)
        writer.add_summary(summary, global_step=_step)
        # projector.visualize_embeddings(writer, network.config)
        log.info("Optimization Finished!")
        log.debug("Elapsed time: {}".format(elapsed(time.time() - start_time)))
        return(loss_total, acc_total)
        session.close()

    def load_tf_model(self, folder=None):
        """This is standard tf_utils stuff."""
        if folder is None: folder = self.logs_path
        log.info("Loading Model: {}".format("Model_Name"))
        if self.sess:
            self.sess.close()
        try:
            self.sess = tf.InteractiveSession()
            checkpoint_file = tf.train.latest_checkpoint(folder)
            log.info("trying: {}".format(folder))
            saver = tf.train.import_meta_graph(checkpoint_file + ".meta")
            log.debug("loading modelfile {}".format(checkpoint_file))
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, checkpoint_file)
            log.info("model successfully Loaded: {}".format(checkpoint_file))
            self.saver = saver
            self.model_loaded = True
        except Exception as e:
            log.warn("This folder failed to produce a model {}\n{}".format(folder, e))
            return False
        return True

    def load_model_params(self):
        """This is standard tf_utils stuff."""
        log.info("Loading Model Params")
        class params(object): pass
        params.list_all_ops = [n.name for n in tf.get_default_graph().as_graph_def().node]
        log.debug("Num ops in model: {}".format(len(params.list_all_ops)))
        params.final_layer = tf.get_collection_ref('final_layer')[0]
        #log.debug("Found Final Layer: {}".format(params.final_layer))
        params.input_word = tf.get_collection_ref('input_word')[0]
        #log.debug("Found input tensor: {}".format(params.input_tensor))
        params.input_label = tf.get_collection_ref('input_label')[0]
        #log.debug("Found input label: {}".format(params.input_label))
        params.global_step = tf.get_collection_ref('global_step')[0]
        #log.debug("Found global_step: {}".format(params.global_step))
        params.learn_rate = tf.get_collection_ref('learn_rate')[0]
        #log.debug("Found learn_rate: {}".format(params.learn_rate))
        params.correct_pred = tf.get_collection_ref('correct_pred')[0]
        #log.debug("Found correct_pred op: {}".format(params.correct_pred))
        params.accuracy = tf.get_collection_ref('accuracy')[0]
        #log.debug("Found accuracy op: {}".format(params.accuracy))
        params.cost = tf.get_collection_ref('cost')[0]
        #log.debug("Found cost op: {}".format(params.cost))
        params.optimizer = tf.get_collection_ref('optimizer')[0]
        #log.debug("Found optimizer op: {}".format(params.optimizer))
        params.init_op = tf.get_collection_ref('init_op')[0]
        # log.debug("Found init_op op: {}".format(params.init_op))
        # params.saver = tf.get_collection_ref('saver')[0]
        # log.debug("Found saver op: {}".format(params.saver))
        params.merged = tf.get_collection_ref('merged')[0]
        # log.debug("Found merged op: {}".format(params.merged))
        # params.config = tf.get_collection_ref('config')[0]
        params.test = "okay"
        self.params = params
        return params


if __name__ == '__main__':
    try:
        os.system('clear')
        app = App()
        if app.main(sys.argv):
            sys.exit("PASSED: Thanks A lot for trying Alphagriffin.com")
        log.warn("Alldone! Alphagriffin.com")

    except KeyboardInterrupt:
        os.system('clear')
        sys.exit("AlphaGriffin.com")
