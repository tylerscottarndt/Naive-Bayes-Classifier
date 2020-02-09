import numpy as np
import sys
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
            combined_list, combined_labels, test_size = 0.3, random_state=1, stratify=combined_labels)

        return x_train, x_test, y_train, y_test

    def save_as_npz(self, x_train, x_dev, x_test, y_train, y_dev, y_test):
        np.savez('train_dev_test_files.npz',
                    x_train=x_train,
                    x_dev=x_dev,
                    x_test=x_test,
                    y_train=y_train,
                    y_dev=y_dev,
                    y_test=y_test)
