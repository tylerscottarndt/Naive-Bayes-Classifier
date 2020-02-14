import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from DataFormatter import DataFormatter
from NaiveBayesClassifier import NaiveBayesClassifier

plt.style.use('seaborn-whitegrid')
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
        print("You saved the formatted data. Please run again to optimize the hyperparameters.")
        sys.exit()

    # instantiate naive_bayes object
    naive_bayes = NaiveBayesClassifier(data_files)

    try:
        # unpickle optimized hyperparameters
        pickle_in = open("optimal_hyperparams.pickle", "rb")
        optimal_hyperparams = pickle.load(pickle_in)
        pickle_in.close()
    except:
        # grid search for optimal parameters
        low_thresh = [0, 5, 10]
        high_thresh = [50, 250, 1000]
        optimal_hyperparams = naive_bayes.train_on_grid_search(low_thresh, high_thresh)
        data_formatter.pickle_object(optimal_hyperparams, "optimal_hyperparams.pickle")
        print("Hyperparameters optimized. Please run again for final test results.")
        sys.exit()

    # regenerate vocabulary for optimal parameters
    naive_bayes.generate_vocabulary(optimal_hyperparams[0], optimal_hyperparams[1])

    # run Naive Bayes classifier on test data
    predictions = naive_bayes.predict(naive_bayes.test_reviews)
    zipped = zip(predictions, naive_bayes.test_labels)

    # calculate and print the accuracy of test results
    correct = 0
    for expected, actual in zipped:
        if expected == actual:
            correct += 1
    accuracy = correct / len(predictions)*100

    print("Correct Predictions: {}".format(correct))
    print("Total Predictions: {}".format(len(predictions)))
    print("ACCURACY: %{0:.2f}\n".format(accuracy))

    # reviews with highest and lowest confidence values
    confidence_values = naive_bayes.confidence_values
    top_confidence_vals = sorted(zip(confidence_values, naive_bayes.test_reviews, predictions), reverse=True)[:5]
    low_confidence_vals = sorted(zip(confidence_values, naive_bayes.test_reviews, predictions), reverse=False)[:5]

    print("HIGHEST CONFIDENCE PREDICTIONS:")
    print("===============================")
    for val in top_confidence_vals:
        print("Predicted Label: {}".format(val[2]))
        print("Confidence: {}".format(val[0]))
        print("Review: '{}'\n".format(" ".join(val[1])))

    print("LOWEST CONFIDENCE PREDICTIONS:")
    print("===============================")
    for val in low_confidence_vals:
        print("Predicted Label: {}".format(val[2]))
        print("Confidence: {}".format(val[0]))
        print("Review: '{}'\n".format(" ".join(val[1])))

    # find the most important features in each class
    best_pos_feature = ""
    best_neg_feature = ""
    best_pos_feature_count = 0
    best_neg_feature_count = 0

    for key in naive_bayes.pos_dict_truncated.keys():
        curr_key_count = naive_bayes.pos_dict_truncated[key]
        if key in naive_bayes.neg_dict_truncated.keys():
            curr_key_count = curr_key_count - naive_bayes.neg_dict_truncated[key]
        if curr_key_count > best_pos_feature_count:
            best_pos_feature = key
            best_pos_feature_count = curr_key_count

    for key in naive_bayes.neg_dict_truncated.keys():
        curr_key_count = naive_bayes.neg_dict_truncated[key]
        if key in naive_bayes.pos_dict.keys():
            curr_key_count = curr_key_count - naive_bayes.pos_dict_truncated[key]
        if curr_key_count > best_neg_feature_count:
            best_neg_feature = key
            best_neg_feature_count = curr_key_count

    print("MOST IMPORTANT ATTRIBUTES:")
    print("===========================")
    print("Positive Class: {}".format(best_pos_feature))
    print("{} More Appearances\n".format(best_pos_feature_count))
    print("Negative Class: {}".format(best_neg_feature))
    print("{} More Appearances\n".format(best_neg_feature_count))
