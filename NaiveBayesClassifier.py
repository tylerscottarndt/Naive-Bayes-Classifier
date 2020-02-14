import pickle
import math
import numpy as np
import matplotlib.pyplot as plt


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
        self.confidence_values = []

        pickle_in = open("pos_dict.pickle", "rb")
        self.pos_dict = pickle.load(pickle_in)
        pickle_in.close()
        self.pos_dict_truncated = self.pos_dict.copy()

        pickle_in = open("neg_dict.pickle", "rb")
        self.neg_dict = pickle.load(pickle_in)
        pickle_in.close()
        self.neg_dict_truncated = self.neg_dict.copy()

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
        self.pos_dict_truncated = self.__update_dictionary(self.pos_dict)
        self.neg_dict_truncated = self.__update_dictionary(self.neg_dict)

    def __update_dictionary(self, dict):
        temp_dict = {}
        for word in dict.keys():
            if word in self.vocabulary:
                temp_dict[word] = dict[word]

        return temp_dict

    def predict(self, data):
        self.confidence_values = []
        predictions = []
        total_pos_words = self.__total_vocab_words_in_dict(self.pos_dict)
        total_neg_words = self.__total_vocab_words_in_dict(self.neg_dict)
        for line in data:
            pos_probability = math.log(self.probability_of_pos)
            neg_probability = math.log(self.probability_of_neg)
            for word in line:
                pos_occurrence = 0
                neg_occurrence = 0
                if word in self.pos_dict_truncated:
                    pos_occurrence = self.pos_dict_truncated[word]
                if word in self.neg_dict_truncated:
                    neg_occurrence = self.neg_dict_truncated[word]
                pos_probability += math.log((pos_occurrence + 1) / (total_pos_words + len(self.vocabulary)))
                neg_probability += math.log((neg_occurrence + 1) / (total_neg_words + len(self.vocabulary)))
            if pos_probability >= neg_probability:
                predictions.append(1)
            else:
                predictions.append(0)
            self.confidence_values.append(round(abs(pos_probability - neg_probability), 4))
        return predictions

    def train_on_grid_search(self, low_thresh_vals, high_thresh_vals):
        errors = []
        best_param_combo = None
        thresh_vals = []
        current_best_accuracy = 0
        for low in low_thresh_vals:
            for high in high_thresh_vals:
                thresh_vals.append("({}, {})".format(low, high))
                self.generate_vocabulary(low, high)
                predictions = self.predict(self.dev_reviews)

                correct_predictions = 0
                for expected, actual in zip(predictions, self.dev_labels):
                    if expected == actual:
                        correct_predictions += 1

                errors.append(len(predictions) - correct_predictions)
                curr_accuracy = correct_predictions/len(predictions)

                if curr_accuracy > current_best_accuracy:
                    current_best_accuracy = curr_accuracy
                    best_param_combo = low, high

        self.__plot_grid_search(thresh_vals, errors)
        return best_param_combo

    def __plot_grid_search(self, thresh_vals, errors):
        ypos = np.arange(len(thresh_vals))
        plt.xticks(ypos, thresh_vals)
        plt.title("Naive Bayes Error Rate")
        plt.xlabel("(Low_Threshold, High_Threshold)")
        plt.ylabel("Errors")
        plt.plot(ypos, errors)
        plt.show()

    def __total_vocab_words_in_dict(self, dict):
        count = 0
        for word in self.vocabulary:
            if word in dict:
                count += dict[word]
        return count
