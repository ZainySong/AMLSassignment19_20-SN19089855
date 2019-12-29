import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def train(training_images, training_labels):
    clf = svm.SVC(kernel = 'linear', C=1, gamma='auto')
    # train the model
    clf.fit(training_images, training_labels)
    # get the training accuracy
    acc_train = clf.score(training_images, training_labels)
    print("Train Accuracy:", acc_train)
    # get the validation accuracy
    acc_val = cross_val_score(clf, training_images, training_labels, cv=5)
    print("Validation Accuracy:", acc_val.mean())
    # get the learning curve
    train_sizes, train_scores, valid_scores = learning_curve(svm.SVC(kernel='linear'), training_images, training_labels, train_sizes = [0.1, 0.25, 0.5, 0.75, 1], cv=5)
    plt.plot(train_sizes, np.mean(train_scores,axis=1), 'o-', color="r", label="Training")
    plt.plot(train_sizes, np.mean(valid_scores,axis=1), 'o-', color="g", label="Cross-validation")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()
    plt.savefig("emotion.png")
    return clf, acc_train, acc_val

def test (clf, test_images, test_labels):
    acc_test = clf.score(test_images, test_labels)
    print("Test Accuracy:", acc_test)
    return acc_test
