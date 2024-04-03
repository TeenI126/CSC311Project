from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i, j, is_correct in zip(data["user_id"], data["question_id"], data["is_correct"]):
        log_lklihood += is_correct * ((theta[i]-beta[j]) - np.log(1 + np.exp(theta[i]-beta[j])))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    d_theta = np.zeros_like(theta)
    for i, j, is_correct in zip(data["user_id"], data["question_id"], data["is_correct"]):
        p = sigmoid(theta[i] - beta[j])
        d_theta[i] += is_correct - p
    theta = theta + lr * d_theta

    d_beta = np.zeros_like(beta)
    for i, j, is_correct in zip(data["user_id"], data["question_id"], data["is_correct"]):
        p = sigmoid(theta[i] - beta[j])
        d_beta[j] += p - is_correct
    beta = beta + lr * d_beta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, test_data, lr, iterations, quiet=False):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.ones(len(data["user_id"]))
    beta = np.ones(len(data["question_id"]))

    val_acc_lst, test_acc_lst = [], []
    train_lld_lst, val_lld_lst = [], []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_lld_lst.append(neg_lld)

        val_lld_lst.append(neg_log_likelihood(val_data, theta, beta))

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)

        test_acc_lst.append(evaluate(test_data, theta, beta))
        if not quiet:
            print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, test_acc_lst, train_lld_lst, val_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    import csv
    from tqdm import tqdm
    
    lr_list = [0.01, 0.02, 0.03, 0.04]
    iter_list = [40, 60, 80, 90, 100]
    gridsearch = [["Learning_rate", "Iterations", "val_score", "test_score"]]
    for lr in tqdm(lr_list):
        _, _, val_acc_lst, test_acc_lst, _, _ = irt(train_data, val_data, test_data, lr, max(iter_list), quiet=True)
        for iter in iter_list:
            val_score = val_acc_lst[iter-1]
            test_score = test_acc_lst[iter-1]
            gridsearch.append([lr, iter, val_score, test_score])
    
    with open("IRT_gridsearch.csv", "w", newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(gridsearch)
    
    best_lr, best_iter = 0.02, 90
    theta, beta, val_acc_lst, test_acc_lst, train_lld_lst, val_lld_lst = irt(train_data, val_data, test_data, best_lr, best_iter, quiet=True)

    print(f"Final model hyperparameters:\nLearning Rate - {best_lr}\nIterations - {best_iter}")
    print("Final model validation accuracy: ", val_acc_lst[-1])
    print("Final model test accuracy: ", test_acc_lst[-1])

    with open("IRT_lld.csv", "w", newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow([i+1 for i in range(best_iter)])
        for train_lld, val_lld in zip(train_lld_lst, val_lld_lst):
            csvwriter.writerow([train_lld, val_lld])

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
