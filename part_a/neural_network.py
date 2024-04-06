from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt


def sigmoid(x):  # borrowed from item_response.py
    """ Apply sigmoid function.
    """
    return torch.exp(x) / (1 + torch.exp(x))


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODONE:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.
        inputs = F.sigmoid(self.g(inputs))
        inputs = F.sigmoid(self.h(inputs))

        #####################################################################
        out = inputs
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, plot_over_epoch=False):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODONE: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    valid_acc = 0
    plots = []
    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + (lamb/2)*(model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        plots.append((epoch + 1, train_loss, valid_acc))
    print("Final Valid Acc: {}".format(valid_acc))
    # if plot_over_epoch:
    #     fig, ax = plt.subplots()
    #     epochs = [i for (i, j, k) in plots]
    #     costs = [j for (i, j, k) in plots]
    #     v_accs = [k for (i, j, k) in plots]
    #     ax.set_xlabel("epochs")
    #     ax.set_ylabel("training cost",color="green")
    #     ax.plot(epochs, costs, color="green")
    #     ax2 = ax.twinx()
    #     ax2.set_ylabel("validation accuracy", color="red")
    #     ax2.plot(epochs, v_accs, color="red")
    #     plt.show() # DISABLED FOR SUBMISSION
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODONE:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # (B)
    # Set model hyperparameters.
    K = [10, 50, 100, 200, 500]
    LR = [0.005, 0.01, 0.02]
    for k in K:
        for lr in LR:
            print(f"\nk = {k}, lr = {lr}")
            model = AutoEncoder(train_matrix.size(1), k)
            # Set optimization hyperparameters.
            num_epoch = 50
            lamb = 0  # no regularization

            # training montage!
            train(model, lr, lamb, train_matrix, zero_train_matrix,
                  valid_data, num_epoch)

    # (C) graphing and testing selected model from part b
    model = AutoEncoder(train_matrix.size(1), 50)
    train(model, 0.02, 0, train_matrix, zero_train_matrix, valid_data, 25, plot_over_epoch=True)

    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(f"Final Test Accuracy: {test_acc}")

    # (D) optimising lambda
    k = 50
    lr = 0.02
    epochs = 25
    lambs = [0.001, 0.01, 0.1, 1]
    lamb_model = []
    for lamb in lambs:
        print(f"\n for lambda = {lamb}")
        model = AutoEncoder(train_matrix.size(1), k)
        train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, epochs)
        lamb_model.append(model)

    test_acc = evaluate(lamb_model[0], zero_train_matrix, test_data)
    print(f"Final Test Accuracy for Regularizer: {test_acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
