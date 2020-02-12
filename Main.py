import sys
import numpy as np
from DataFormatter import DataFormatter
from NaiveBayesClassifier import NaiveBayesClassifier
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

        # split data into 70% train, 15% dev, 15% test
        print("Splitting data...")
        x_train, x_dev, x_test, y_train, y_dev, y_test = data_formatter.split_data(pos_reviews, neg_reviews)

        # save to npz file
        print("Saving data...")
        data_formatter.save_as_npz(x_train, x_dev, x_test, y_train, y_dev, y_test)

        # generate dictionaries of positive and negative words
        print("Generating Dictionaries...")
        pos_dict, neg_dict = data_formatter.generate_pos_neg_dict(x_train, y_train)

        # pickle the dictionaries
        print("Saving Dictionaries...")
        data_formatter.pickle_object(pos_dict, "pos_dict.pickle")
        data_formatter.pickle_object(neg_dict, "neg_dict.pickle")

        # ask user to run the code again for results
        print("You saved the formatted data, please run again for the Naive Bayes classifier.")
        sys.exit()

    naive_bayes = NaiveBayesClassifier(data_files)
    naive_bayes.generate_vocabulary(0, 1000)
    print(len(naive_bayes.vocabulary))

# generate pos dict
# gnerate neg dict
# generate sum of pos dict values
# generate sum of neg dict values
# generate unique vocabulay of combined lists


