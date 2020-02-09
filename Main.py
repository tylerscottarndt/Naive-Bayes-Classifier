import time
import sys
import numpy as np
from DataFormatter import DataFormatter
data_formatter = DataFormatter()


if __name__ == '__main__':
    # load the data from .npz file if it exists
    try:
        data_files = np.load('train_dev_test_files.npz', allow_pickle=True)
    # load, clean, and split data for the first time
    except:
        # turn txt files into list of positive and negative reviews
        print("Loading data...")
        pos_reviews = data_formatter.load_data("rt-polarity.pos.txt")
        neg_reviews = data_formatter.load_data("rt-polarity.neg.txt")

        # clean data by stemming and removing stop words
        print("Cleaning data...")
        pos_reviews = data_formatter.clean_data(pos_reviews)
        neg_reviews = data_formatter.clean_data(neg_reviews)

        # split data into 70% train, 30% test
        print("Splitting data...")
        x_train, x_test, y_train, y_test = data_formatter.split_data(pos_reviews, neg_reviews)

        # evenly split test into test and dev
        # result: 70% train, 15% dev, 15% test
        x_dev = x_test[:len(x_test)//2]
        x_test = x_test[len(x_test)//2:]
        y_dev = y_test[:len(y_test)//2]
        y_test = y_test[len(y_test)//2:]

        print("Saving...")
        data_formatter.save_as_npz(x_train, x_dev, x_test, y_train, y_dev, y_test)
        print("Saved!")
        print("You split the data, please run again for the Naive Bayes classifier.")
        sys.exit()

    train_reviews = data_files['x_train']
    train_labels = data_files['y_train']


