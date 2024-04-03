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
    print("Impute by User Accuracy - {}".format(acc))
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
    # Implement the function as described in the docstring.             #
    #####################################################################

    # Core assumption: Questions are similar to all other questions

    matrix = np.transpose(matrix) # Transpose matrix so that rows become columns and vice versa
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    mat = np.transpose(mat) # Tranpose back to make user
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Impute by Item Accuracy - {}".format(acc))
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
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26] # Domain of k stated in the question
    best_k_user = best_k_item = None
    best_acc_user = best_acc_item = 0

    for k in k_values:
        print(f"\nEvaluating k = {k}")
        acc_user = knn_impute_by_user(sparse_matrix, val_data, k)
        if acc_user > best_acc_user:
            best_k_user = k
            best_acc_user = acc_user

        acc_item = knn_impute_by_item(sparse_matrix, val_data, k)
        if acc_item > best_acc_item:
            best_k_item = k
            best_acc_item = acc_item
        
    print('\n',"#"*70, '\n')
    print(f"Best results for user-based collaborative filtering:\nk - {best_k_user}\
          \nvalidation accuracy - {best_acc_user}")
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    
    print(f"\n\nBest results for item-based collaborative filtering:\nk - {best_k_item}\
          \nvalidation accuracy - {best_acc_item}")
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, best_k_item)
    

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
