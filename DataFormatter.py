import numpy as np
import sys
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

stop_words = set(stopwords.words('english'))
np.set_printoptions(threshold=sys.maxsize)


class DataFormatter:
    def load_data(self, data_file):
        data_list = open(data_file).read().split("\n")
        return data_list

    def clean_data(self, data_list):
        porter_stemmer = PorterStemmer()
        result = []
        for line in data_list:
            words = line.split(" ")
            cleaned_words = [porter_stemmer.stem(token) for token in words if token not in stop_words]
            cleaned_words[:] = [token for token in cleaned_words if token != "," and token != "." and token != ""]
            result.append(cleaned_words)
        return result

    def split_data(self, class_1_list, class_0_list):
        class_1_labels = [1]*len(class_1_list)
        class_0_labels = [0]*len(class_0_list)

        combined_labels = np.asarray(class_1_labels + class_0_labels)
        combined_list = np.asarray(class_1_list + class_0_list)

        x_train, x_test, y_train, y_test = train_test_split(
            combined_list, combined_labels, test_size=0.3, random_state=1, stratify=combined_labels)

        # evenly split test into test and dev
        # result: 70% train, 15% dev, 15% test
        x_dev = x_test[:len(x_test)//2]
        x_test = x_test[len(x_test)//2:]
        y_dev = y_test[:len(y_test)//2]
        y_test = y_test[len(y_test)//2:]

        return x_train, x_dev, x_test, y_train, y_dev, y_test

    def generate_pos_neg_dict(self, docs, labels):
        pos_dict = {}
        neg_dict = {}
        zipped = zip(docs, labels)
        for line, label in zipped:
            if label == 1:
                self.__place_line_in_dict(line, pos_dict)
            else:
                self.__place_line_in_dict(line, neg_dict)

        return pos_dict, neg_dict

    def __place_line_in_dict(self, line, some_dict):
        for word in line:
            if word in some_dict.keys():
                word_count = some_dict[word]
                some_dict[word] = word_count + 1
            else:
                some_dict[word] = 1

    def save_as_npz(self, x_train, x_dev, x_test, y_train, y_dev, y_test):
        np.savez('train_dev_test_files.npz',
                    x_train=x_train,
                    x_dev=x_dev,
                    x_test=x_test,
                    y_train=y_train,
                    y_dev=y_dev,
                    y_test=y_test)

    def pickle_object(self, object_to_pickle, file_name):
        pickle_out = open(file_name, "wb")
        pickle.dump(object_to_pickle, pickle_out)
        pickle_out.close()


