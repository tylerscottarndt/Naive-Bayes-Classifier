import pickle
import math


class NaiveBayesClassifier:
    def __init__(self, data_files):
        self.train_reviews = data_files['x_train']
        self.train_labels = data_files['y_train']
        self.dev_reviews = data_files['x_dev']
        self.dev_labels = data_files['y_dev']
        self.test_reviews = data_files['x_test']
        self.test_labels = data_files['y_test']
        self.probability_of_pos = len([i for i in self.train_labels if i == 1]) / len(self.train_labels)
        self.probability_of_neg = 1 - self.probability_of_pos

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

    def predict(self, data):
        predictions = []
        total_pos_words = self.__total_vocab_words_in_dict(self.pos_dict)
        total_neg_words = self.__total_vocab_words_in_dict(self.neg_dict)
        for line in data:
            pos_probability = math.log(self.probability_of_pos)
            neg_probability = math.log(self.probability_of_neg)
            for word in line:
                pos_occurrence = 0
                neg_occurrence = 0
                if word in self.pos_dict:
                    pos_occurrence = self.pos_dict[word]
                if word in self.neg_dict:
                    neg_occurrence = self.neg_dict[word]
                pos_probability += math.log((pos_occurrence + 1) / (total_pos_words + len(self.vocabulary)))
                neg_probability += math.log((neg_occurrence + 1) / (total_neg_words + len(self.vocabulary)))
            if pos_probability >= neg_probability:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    def __total_vocab_words_in_dict(self, dict):
        count = 0
        for word in self.vocabulary:
            if word in dict:
                count += dict[word]
        return count
