import numpy as np
from sklearn.impute import KNNImputer
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(np.array(matrix).transpose())
    acc = sparse_matrix_evaluate(valid_data, np.array(mat).transpose())
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    K = [1, 6, 11, 16, 21, 26]
    print("part a) Impute by user")
    max_acc = []
    max_k = 1
    print("\n")
    for k in K:
        print(f"for k = {k}")
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        max_acc.append(acc)
        if acc == max(max_acc):
            max_k = k
        print("")

    print(f"best k was {max_k}, so we select it")

    print("This achieves a test accuracy of:")
    knn_impute_by_user(sparse_matrix, test_data, max_k)

    print("part c) Impute by item")
    max_acc = []
    max_k = 1
    print("\n")
    for k in K:
        print(f"for k = {k}")
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        max_acc.append(acc)
        if acc == max(max_acc):
            max_k = k
        print("")

    print(f"best k was {max_k}, so we select it")

    print("This achieves a test accuracy of:")
    knn_impute_by_item(sparse_matrix, test_data, max_k)
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
