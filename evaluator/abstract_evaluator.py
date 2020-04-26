"""
@author: Zhongchuan Sun
"""


class AbstractEvaluator(object):
    """Base class for all evaluator.
    """

    def __init__(self):
        pass

    def metrics_info(self):
        """Get all metrics information.

        Returns:
            str: A string consist of all metrics information， such as
            `"Precision@10    Precision@20    NDCG@10    NDCG@20"`.
        """
        raise NotImplementedError

    def evaluate(self, model):
        """Evaluate `model`.

        Args:
            model: The model need to be evaluated. This model must have
                a method `predict_for_eval(self, users)`, where the argument
                `users` is a list of users and the return is a 2-D array that
                contains `users` rating/ranking scores on all items.

        Returns:
            str: A string consist of all results, such as
            `"0.18663847    0.11239596    0.35824192    0.21479650"`.
        """
        raise NotImplementedError
