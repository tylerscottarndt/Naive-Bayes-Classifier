import numpy as np
import pickle


class NaiveBayesClassifier:
    def __init__(self, data_files):
        self.train_reviews = data_files['x_train']
        self.train_labels = data_files['y_train']
        self.dev_reviews = data_files['x_dev']
        self.dev_labels = data_files['y_dev']
        self.test_reviews = data_files['x_test']
        self.test_labels = data_files['y_test']

        pickle_in = open("pos_dict.pickle", "rb")
        self.pos_dict = pickle.load(pickle_in)
        pickle_in.close()

        pickle_in = open("neg_dict.pickle", "rb")
        self.neg_dict = pickle.load(pickle_in)
        pickle_in.close()

        self.vocabulary = []

    def generate_vocabulary(self, lower_count_thresh, upper_count_thresh):
        pos_vocab = []
        neg_vocab = []
        for key in self.pos_dict.keys():
            if lower_count_thresh < self.pos_dict[key] < upper_count_thresh:
                pos_vocab.append(key)
        for key in self.neg_dict.keys():
            if lower_count_thresh < self.neg_dict[key] < upper_count_thresh:
                neg_vocab.append(key)

        self.vocabulary = set(pos_vocab + neg_vocab)
