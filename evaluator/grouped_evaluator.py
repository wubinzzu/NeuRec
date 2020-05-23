"""
@author: Zhongchuan Sun
"""
from util import typeassert
import numpy as np
from collections import OrderedDict
import pandas as pd
from .abstract_evaluator import AbstractEvaluator
from .backend import UniEvaluator


class GroupedEvaluator(AbstractEvaluator):
    """`GroupedEvaluator` evaluates models in user groups.

    This class evaluates the ranking performance of models in user groups,
    which are split according to the numbers of users' interactions in
    **training data**. This function can be activated by the argument
    `group_view`, which must be a list of integers.
    For example, if `group_view = [10,30,50,100]`, users will be split into
    four groups: `(0, 10]`, `(10, 30]`, `(30, 50]` and `(50, 100]`. And the
    users whose interacted items more than `100` will be discard.
    """
    @typeassert(user_train_dict=dict, user_test_dict=dict, group_view=list)
    def __init__(self, user_train_dict, user_test_dict, user_neg_test=None,
                 metric=None, group_view=None, top_k=50, batch_size=1024, num_thread=8):
        """Initializes a new `GroupedEvaluator` instance.

        Args:
            user_train_dict (dict): Each key is user ID and the corresponding
                value is the list of **training items**.
            user_test_dict (dict): Each key is user ID and the corresponding
                value is the list of **test items**.
            metric (None or list of str): If `metric == None`, metric will
                be set to `["Precision", "Recall", "MAP", "NDCG", "MRR"]`.
                Otherwise, `metric` must be one or a sublist of metrics
                mentioned above. Defaults to `None`.
            group_view (list of int): A list of integers.
            top_k (int or list of int): `top_k` controls the Top-K item ranking
                performance. If `top_k` is an integer, K ranges from `1` to
                `top_k`; If `top_k` is a list of integers, K are only assigned
                these values. Defaults to `50`.
            batch_size (int): An integer to control the test batch size.
                Defaults to `1024`.
            num_thread (int): An integer to control the test thread number.
                Defaults to `8`.

        Raises:
             TypeError: If `group_view` is not a list.
             ValueError: If user splitting with `group_view` is not suitable.
        """
        super(GroupedEvaluator, self).__init__()

        if not isinstance(group_view, list):
            raise TypeError("The type of 'group_view' must be `list`!")

        self.evaluator = UniEvaluator(user_train_dict, user_test_dict, user_neg_test,
                                      metric=metric, top_k=top_k,
                                      batch_size=batch_size,
                                      num_thread=num_thread)
        self.user_pos_train = user_train_dict
        self.user_pos_test = user_test_dict

        group_list = [0] + group_view
        group_info = [("(%d,%d]:" % (g_l, g_h)).ljust(12)
                      for g_l, g_h in zip(group_list[:-1], group_list[1:])]

        all_test_user = list(self.user_pos_test.keys())
        num_interaction = [len(self.user_pos_train[u]) for u in all_test_user]
        group_idx = np.searchsorted(group_list[1:], num_interaction)
        user_group = pd.DataFrame(list(zip(all_test_user, group_idx)),
                                  columns=["user", "group"])
        grouped = user_group.groupby(by=["group"])

        self.grouped_user = OrderedDict()
        for idx, users in grouped:
            if idx < len(group_info):
                self.grouped_user[group_info[idx]] = users["user"].tolist()

        if not self.grouped_user:
            raise ValueError("The splitting of user groups is not suitable!")

    def metrics_info(self):
        """Get all metrics information.

        Returns:
            str: A string consist of all metrics information， such as
            `"Precision@10    Precision@20    NDCG@10    NDCG@20"`.
        """
        return self.evaluator.metrics_info()

    def evaluate(self, model):
        """Evaluate `model` in user groups.

        Args:
            model: The model need to be evaluated. This model must have
                a method `predict_for_eval(self, users)`, where the argument
                `users` is a list of users and the return is a 2-D array that
                contains `users` rating/ranking scores on all items.

        Returns:
            str: A multi-line string consist of all results of groups, such as:
                `"(0,10]:   0.00648002   0.00421617   0.00301847   0.00261693\n
                (10,30]:  0.00686600   0.00442968   0.00310077   0.00249169\n
                (30,50]:  0.00653595   0.00326797   0.00217865   0.00163399\n
                (50,100]: 0.00423729   0.00211864   0.00141243   0.00105932"`
        """
        result_to_show = ""
        for group, users in self.grouped_user.items():
            tmp_result = self.evaluator.evaluate(model, users)
            result_to_show = "%s\n%s\t%s" % (result_to_show, group, tmp_result)

        return result_to_show
