import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix


def downvote_seen_items(scores: np.ndarray, data: pd.DataFrame, data_description: dict) -> None:
    """
    Downvote relevance scores for seen items.

    Parameters
    ----------
    scores : np.ndarray
        A dense numpy array of scores.
    data : pd.DataFrame
        A pandas DataFrame containing user-item interaction data.
    data_description : dict
        A dictionary containing metadata about the data, including 'items' and 'users' keys.

    Returns
    -------
    None
        The function modifies the input scores array in-place.

    Notes
    -----
    This function assumes that the input scores array is sorted by user index.
    It also assumes that the `users` and `items` keys in the `data_description` dictionary
    correspond to the column names in the `data` DataFrame.
    """
    assert isinstance(scores, np.ndarray), 'Scores must be a dense numpy array!'
    itemid = data_description['items']
    userid = data_description['users']
    # get indices of observed data, corresponding to scores array
    # we need to provide correct mapping of rows in scores array into
    # the corresponding user index (which is assumed to be sorted)
    row_idx, test_users = pd.factorize(data[userid], sort=True)
    assert len(test_users) == scores.shape[0]
    col_idx = data[itemid].values
    # downvote scores at the corresponding positions
    scores[row_idx, col_idx] = scores.min() - 1


def postprocess_scores(scoring_func):
    '''Converts the scores computed with the `scoring_func`
    from a sparse matrix to a dense array if necessary,
    and then applies the `downvote_seen_items` function
    to prevent recommending the items a user has already seen.
    This function is designed to be used as a decorator.
    '''
    def scoring_wrapper(params, testset, testset_description):
        scores = scoring_func(params, testset, testset_description)
        if isspmatrix(scores):
            scores = scores.toarray()
        downvote_seen_items(scores, testset, testset_description)
        return scores
    return scoring_wrapper


def topn_recommendations(scores: np.ndarray, topn: int=10) -> np.ndarray:
    """
    Generate top-N recommendations based on the input scores.

    Parameters
    ----------
    scores : np.ndarray
        A dense numpy array of scores, where each row corresponds to a user and each column to an item.
    topn : int, optional
        The number of top recommendations to generate for each user. Defaults to 10.

    Returns
    -------
    np.ndarray
        A numpy array of shape (n_users, topn) containing the indices of the top-N recommended items for each user.
    """
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations


def topidx(a: np.ndarray, topn: int) -> np.ndarray:
    """
    Returns the indices of the top-N elements in the input array.
    The calculation is based on the argpartition method, which is more efficient for large arrays.

    Parameters
    ----------
    a : np.ndarray
        The input array from which to select the top-N elements.
    topn : int
        The number of top elements to select.

    Returns
    -------
    np.ndarray
        An array of indices corresponding to the top-N elements in the input array.
    """
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]


def model_evaluate(
        recommended_items: np.ndarray,
        holdout: pd.DataFrame,
        holdout_description: dict,
        topn: int = 10
    ) -> tuple:
    """
    Evaluates the performance of a recommender model by comparing the recommended items against the holdout.

    Parameters
    ----------
    recommended_items : np.ndarray
        A numpy array of shape (n_users, topn) containing the indices of the top-N recommended items for each user.
    holdout : pd.DataFrame
        A pandas DataFrame containing the holdout data for evaluation.
    holdout_description : dict
        A dictionary containing the description of the holdout data, including the column names for items.
    topn : int, optional
        The number of top recommendations to consider for evaluation. Defaults to 10.

    Returns
    -------
    tuple
        A tuple containing three metrics: HR (Hit Rate), MRR (Mean Reciprocal Rank), and Coverage.
    """
    itemid = holdout_description['items']
    holdout_items = holdout[itemid].values
    assert recommended_items.shape[0] == len(holdout_items)
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)
    # HR calculation
    hr = np.mean(hits_mask.any(axis=1))
    # MRR calculation
    n_test_users = recommended_items.shape[0]
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users
    # coverage calculation
    n_items = holdout_description['n_items']
    cov = np.unique(recommended_items).size / n_items
    return hr, mrr, cov


def calculate_rmse(
    scores: np.ndarray,
    holdout: pd.DataFrame,
    holdout_description: dict
) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between predicted scores and actual feedback in the holdout data.

    Parameters
    ----------
    scores : np.ndarray
        A numpy array of shape (n_users, n_items) containing the predicted scores for each user-item pair.
    holdout : pd.DataFrame
        A pandas DataFrame containing the holdout data for evaluation.
    holdout_description : dict
        A dictionary containing the description of the holdout data, including the column names for items and feedback.

    Returns
    -------
    float
        The Root Mean Squared Error (RMSE) between the predicted scores and actual feedback.
    """
    user_idx = np.arange(holdout.shape[0])
    item_idx = holdout[holdout_description['items']].values
    feedback = holdout[holdout_description['feedback']].values
    predicted_rating = scores[user_idx, item_idx]
    return np.mean(np.abs(predicted_rating-feedback)**2)