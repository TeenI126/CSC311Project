import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
import numpy as np

# terminate after iterations as opposed to convergence
def learn_clusters(X, iter=100, k=10):
    tensor_data = torch.from_numpy(X).float()

    # intialize centers
    centers = tensor_data[torch.randperm(tensor_data.size(0))[:10]]

    labels = None
    for i in range(iter):
        distances = torch.cdist(tensor_data, centers)

        _, labels = torch.min(distances, dim=1)

        # update centers
        for j in range(k):
            if torch.sum(labels == j) > 0:
                centers[j] = torch.mean(tensor_data[labels == j], dim=0)
    return labels


def cluster_labels():
    question_meta = utils.load_question_meta_csv("../data")
    num_subjects = 388
    num_questions = 1774
    ques_numpy = np.zeros((num_questions, num_subjects))
    for i, j in zip(question_meta["question_id"], question_meta["subject_id"]):
        k = [int(p) for p in j]
        ques_numpy[i][k] = 1

    labels = learn_clusters(ques_numpy)
    return labels




if __name__ == '__main__':
    cluster_labels()
