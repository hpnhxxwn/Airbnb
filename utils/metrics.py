import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer


def dcg_score(y_true, y_score, k=5):
    """Discounted cumulative gain (DCG) at rank K.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    score : float
    """
    y_predict_rank = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, y_predict_rank[:k])

    gain = 2 ** y_true - 1
    #print("gain is ")
    #print(gain)
    
    discounts = np.log2(np.arange(len(y_true)) + 2)
    #print(discounts)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=5):
    """
    Normalized discounted cumulative gain (NDCG) at rank K.
    Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
    recommendation system based on the graded relevance of the recommended
    entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
    ranking of the entities.
    Parameters
    ----------
    ground_truth : array, shape = [n_samples]
        Ground truth (true labels represended as integers).
    predictions : array, shape = [n_samples, n_classes]
        Predicted probabilities.
    k : The maximum number of entities that can be recommended
    Returns
    -------
    score : float
    Example
    -------
    >>> ground_truth = [1, 0, 2]
    >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    1.0
    >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    0.6666666666
    """
    lb = LabelBinarizer()

    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth) + 1
    #print(T)
    #print(predictions + 1)
    scores = []
    #pred = predictions + 1

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        #print("actual score is ")
        actual = dcg_score(y_true, y_score, k)
        #print(y_true)
        #print(y_score)
        #print("best score is ")
        best = dcg_score(y_true, y_true, k)
        #print(best)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)

def ndcg5_score(preds, dtrain):
    labels = dtrain.get_label()
    top = []

    for i in range(preds.shape[0]):
        top.append(np.argsort(preds[i])[::-1][:5])

    mat = np.reshape(np.repeat(labels,np.shape(top)[1]) == np.array(top).ravel(),np.array(top).shape).astype(int)
    score = np.mean(np.sum(mat/np.log2(np.arange(2, mat.shape[1] + 2)),axis = 1))
    return 'ndcg5', score

# NDCG Scorer function
ndcg_scorer = make_scorer(ndcg_score, needs_proba=True, k=5)
