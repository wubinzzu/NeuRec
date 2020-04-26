"""
@author: Zhongchuan Sun
"""
from util import typeassert
from .abstract_evaluator import AbstractEvaluator
from .backend import UniEvaluator
from .grouped_evaluator import GroupedEvaluator


class ProxyEvaluator(AbstractEvaluator):
    """`ProxyEvaluator` is the interface to evaluate models.

    `ProxyEvaluator` contains various evaluation protocols:

    * **First**, evaluation metrics of this class are configurable via the
      argument `metric`. Now there are five configurable metrics: `Precision`,
      `Recall`, `MAP`, `NDCG` and `MRR`.

    * **Second**, this class and its evaluation metrics can automatically fit
      both leave-one-out and fold-out data splitting without specific indication.
      In **leave-one-out** evaluation, 1) `Recall` is equal to `HitRatio`;
      2) The implementation of `NDCG` is compatible with fold-out; 3) `MAP` and
      `MRR` have same numeric values; 4) `Precision` is meaningless.

    * **Furthermore**, the ranking performance of models can be viewed in user
      groups, which are split according to the numbers of users' interactions
      in **training data**. This function can be activated by the argument
      `group_view`. Specifically, if `group_view == None`, the ranking performance
      will be viewed without groups; If `group_view` is a list of integers,
      the ranking performance will be view in groups.
      For example, if `group_view = [10,30,50,100]`, users will be split into
      four groups: `(0, 10]`, `(10, 30]`, `(30, 50]` and `(50, 100]`. And the
      users whose interacted items more than `100` will be discarded.

    * **Finally and importantly**, all the functions mentioned above depend on
      `UniEvaluator`, which is implemented by **python** and **cpp**.
      And both of the two versions are **multi-threaded**.
    """

    @typeassert(user_train_dict=dict, user_test_dict=dict)
    def __init__(self, user_train_dict, user_test_dict, user_neg_test=None, metric=None,
                 group_view=None, top_k=50, batch_size=1024, num_thread=8):
        """Initializes a new `ProxyEvaluator` instance.

        Args:
            user_train_dict (dict): Each key is user ID and the corresponding
                value is the list of **training items**.
            user_test_dict (dict): Each key is user ID and the corresponding
                value is the list of **test items**.
            metric (None or list of str): If `metric == None`, metric will
                be set to `["Precision", "Recall", "MAP", "NDCG", "MRR"]`.
                Otherwise, `metric` must be one or a sublist of metrics
                mentioned above. Defaults to `None`.
            group_view (None or list of int): If `group_view == None`, the ranking
                performance will be viewed without groups. If `group_view` is a
                list of integers, ranking performance will be viewed in groups.
                Defaults to `None`.
            top_k (int or list of int): `top_k` controls the Top-K item ranking
                performance. If `top_k` is an integer, K ranges from `1` to
                `top_k`; If `top_k` is a list of integers, K are only assigned
                these values. Defaults to `50`.
            batch_size (int): An integer to control the test batch size.
                Defaults to `1024`.
            num_thread (int): An integer to control the test thread number.
                Defaults to `8`.

        Raises:
            ValueError: If `metric` or one of its element is not in
                `["Precision", "Recall", "MAP", "NDCG", "MRR"]`.

        TODO:
            * Check the validation of `num_thread` in cpp implementation.
        """
        super(ProxyEvaluator, self).__init__()
        if group_view is not None:
            self.evaluator = GroupedEvaluator(user_train_dict, user_test_dict, user_neg_test,
                                              metric=metric, group_view=group_view,
                                              top_k=top_k, batch_size=batch_size,
                                              num_thread=num_thread)
        else:
            self.evaluator = UniEvaluator(user_train_dict, user_test_dict, user_neg_test,
                                          metric=metric, top_k=top_k,
                                          batch_size=batch_size,
                                          num_thread=num_thread)

    def metrics_info(self):
        """Get all metrics information.

        Returns:
            str: A string consist of all metrics information， such as
                `"Precision@10    Precision@20    NDCG@10    NDCG@20"`.
        """
        return self.evaluator.metrics_info()

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
        return self.evaluator.evaluate(model)
