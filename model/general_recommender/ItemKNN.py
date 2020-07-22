"""
reference: https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation
"""
import numpy as np
import scipy.sparse as sps
from model.AbstractRecommender import AbstractRecommender
from util import timer
import sys
import time

class Compute_Similarity_Euclidean:


    def __init__(self, dataMatrix, topK=100, shrink = 0, normalize=False, normalize_avg_row=False,
                 similarity_from_distance_mode ="lin", row_weights = None, **args):
        """
        Computes the euclidean similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param normalize
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param similarity_from_distance_mode:       "exp"        euclidean_similarity = 1/(e ^ euclidean_distance)
                                                    "lin"        euclidean_similarity = 1/(1 + euclidean_distance)
                                                    "log"        euclidean_similarity = 1/(1 + euclidean_distance)
        :param args:                accepts other parameters not needed by the current object
        """

        super(Compute_Similarity_Euclidean, self).__init__()

        self.shrink = shrink
        self.normalize = normalize
        self.normalize_avg_row = normalize_avg_row

        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topK, self.n_columns)

        self.dataMatrix = dataMatrix.copy()

        self.similarity_is_exp = False
        self.similarity_is_lin = False
        self.similarity_is_log = False

        if similarity_from_distance_mode == "exp":
            self.similarity_is_exp = True
        elif similarity_from_distance_mode == "lin":
            self.similarity_is_lin = True
        elif similarity_from_distance_mode == "log":
            self.similarity_is_log = True
        else:
            raise ValueError("Compute_Similarity_Euclidean: value for parameter 'mode' not recognized."
                             " Allowed values are: 'exp', 'lin', 'log'."
                             " Passed value was '{}'".format(similarity_from_distance_mode))



        self.use_row_weights = False

    def compute_similarity(self, start_col=None, end_col=None, block_size = 100):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """
        values = []
        rows = []
        cols = []

        start_time = time.time()
        start_time_print_batch = start_time
        processedItems = 0


        #self.dataMatrix = self.dataMatrix.toarray()

        start_col_local = 0
        end_col_local = self.n_columns

        if start_col is not None and start_col>0 and start_col<self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col>start_col_local and end_col<self.n_columns:
            end_col_local = end_col

        # Compute sum of squared values
        item_distance_initial = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()
        sumOfSquared = np.sqrt(item_distance_initial)

        start_col_block = start_col_local

        this_block_size = 0

        # Compute all similarities for each item using vectorization
        while start_col_block < end_col_local:

            # Add previous block size
            processedItems += this_block_size

            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block-start_col_block

            if time.time() - start_time_print_batch >= 30 or end_col_block==end_col_local:
                columnPerSec = processedItems / (time.time() - start_time + 1e-9)

                print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / (end_col_local - start_col_local) * 100, columnPerSec, (time.time() - start_time)/ 60))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()


            # All data points for a given item
            item_data = self.dataMatrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()

            # If only 1 feature avoid last dimension to disappear
            if item_data.ndim == 1:
                item_data = np.atleast_2d(item_data)

            if self.use_row_weights:
                this_block_weights = self.dataMatrix_weighted.T.dot(item_data)

            else:
                # Compute item similarities
                this_block_weights = self.dataMatrix.T.dot(item_data)



            for col_index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_column_weights = this_block_weights
                else:
                    this_column_weights = this_block_weights[:,col_index_in_block]


                columnIndex = col_index_in_block + start_col_block

                # item_data = self.dataMatrix[:,columnIndex]

                # (a-b)^2 = a^2 + b^2 - 2ab
                item_distance = item_distance_initial.copy()
                item_distance += item_distance_initial[columnIndex]

                # item_distance -= 2*item_data.T.dot(self.dataMatrix).toarray().ravel()
                item_distance -= 2 * this_column_weights

                item_distance[columnIndex] = 0.0


                if self.use_row_weights:
                    item_distance = np.multiply(item_distance, self.row_weights)


                if self.normalize:
                    item_distance = item_distance / (sumOfSquared[columnIndex] * sumOfSquared)

                if self.normalize_avg_row:
                    item_distance = item_distance / self.n_rows

                item_distance = np.sqrt(item_distance)

                if self.similarity_is_exp:
                    item_similarity = 1/(np.exp(item_distance) + self.shrink + 1e-9)

                elif self.similarity_is_lin:
                    item_similarity = 1/(item_distance + self.shrink + 1e-9)

                elif self.similarity_is_log:
                    item_similarity = 1/(np.log(item_distance+1) + self.shrink + 1e-9)

                else:
                    assert False


                item_similarity[columnIndex] = 0.0

                this_column_weights = item_similarity



                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.TopK-1)[0:self.TopK]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_column_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_column_weights[top_k_idx][notZerosMask])
                rows.extend(top_k_idx[notZerosMask])
                cols.extend(np.ones(numNotZeros) * columnIndex)


            start_col_block += block_size


        # End while on columns

        W_sparse = sps.csr_matrix((values, (rows, cols)),
                                  shape=(self.n_columns, self.n_columns),
                                  dtype=np.float32)

        return W_sparse

class Compute_Similarity:
    def   __init__(self, dataMatrix, similarity = None, **args):
        """
        Interface object that will call the appropriate similarity implementation
        :param dataMatrix:
        :param use_implementation:      "density" will choose the most efficient implementation automatically
                                        "cython" will use the cython implementation, if available. Most efficient for sparse matrix
                                        "python" will use the python implementation. Most efficent for dense matrix
        :param similarity:              the type of similarity to use, see SimilarityFunction enum
        :param args:                    other args required by the specific similarity implementation
        """
        if similarity == "euclidean":
            pass
            # This is only available here
            self.compute_similarity_object = Compute_Similarity_Euclidean(dataMatrix, **args)
        else:
            if similarity is not None:
                args["similarity"] = similarity
            self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)

    def compute_similarity(self,  **args):

        return self.compute_similarity_object.compute_similarity(**args)

class Compute_Similarity_Python:

    def __init__(self, dataMatrix, topK=100, shrink=0, normalize=True,
                 asymmetric_alpha=0.5, tversky_alpha=1.0, tversky_beta=1.0,
                 similarity="cosine"):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param shrink:
        :param normalize:           If True divide the dot product by the product of the norms
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param asymmetric_alpha     Coefficient alpha for the asymmetric cosine
        :param similarity:  "cosine"        computes Cosine similarity
                            "adjusted"      computes Adjusted Cosine, removing the average of the users
                            "asymmetric"    computes Asymmetric Cosine
                            "pearson"       computes Pearson Correlation, removing the average of the items
                            "jaccard"       computes Jaccard similarity for binary interactions using Tanimoto
                            "dice"          computes Dice similarity for binary interactions
                            "tversky"       computes Tversky similarity for binary interactions
                            "tanimoto"      computes Tanimoto coefficient for binary interactions
        """
        """
        Asymmetric Cosine as described in: 
        Aiolli, F. (2013, October). Efficient top-n recommendation for very large scale binary rated datasets. In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.
        """

        super(Compute_Similarity_Python, self).__init__()

        self.shrink = shrink
        self.normalize = normalize

        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topK, self.n_columns)

        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.dataMatrix = dataMatrix.copy()

        self.adjusted_cosine = False
        self.asymmetric_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False
        self.dice_coefficient = False
        self.tversky_coefficient = False

        if similarity == "adjusted":
            self.adjusted_cosine = True
        elif similarity == "asymmetric":
            self.asymmetric_cosine = True
        elif similarity == "pearson":
            self.pearson_correlation = True
        elif similarity == "jaccard" or similarity == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif similarity == "dice":
            self.dice_coefficient = True
            self.normalize = False

        elif similarity == "tversky":
            self.tversky_coefficient = True
            self.normalize = False

        elif similarity == "cosine":
            pass
        else:
            raise ValueError("Cosine_Similarity: value for parameter 'mode' not recognized."
                             " Allowed values are: 'cosine', 'pearson', 'adjusted', 'asymmetric', 'jaccard', 'tanimoto',"
                             "dice, tversky."
                             " Passed value was '{}'".format(similarity))

        self.use_row_weights = False

    def applyAdjustedCosine(self):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """
        self.dataMatrix = self.dataMatrix.tocsr()
        # in csr matrix, indptr is the index of the first element of matrix
        interactionsPerRow = np.diff(self.dataMatrix.indptr)
        # calculate each row's element number
        nonzeroRows = interactionsPerRow > 0
        # calculate the indices of nonZero Rows
        sumPerRow = np.asarray(self.dataMatrix.sum(axis=1)).ravel()
        # calculate each row's sum
        rowAverage = np.zeros_like(sumPerRow)
        # all zero array shape: row
        rowAverage[nonzeroRows] = sumPerRow[nonzeroRows] / interactionsPerRow[nonzeroRows]
        # each row's average rating
        # Split in blocks to avoid duplicating the whole data structure
        start_row = 0
        end_row = 0

        blockSize = 1000

        while end_row < self.n_rows:
            end_row = min(self.n_rows, end_row + blockSize)
            # average every row
            self.dataMatrix.data[self.dataMatrix.indptr[start_row]:self.dataMatrix.indptr[end_row]] -= \
                np.repeat(rowAverage[start_row:end_row], interactionsPerRow[start_row:end_row])

            start_row += blockSize

    def applyPearsonCorrelation(self):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """
        # manage as column
        self.dataMatrix = self.dataMatrix.tocsc()

        interactionsPerCol = np.diff(self.dataMatrix.indptr)

        nonzeroCols = interactionsPerCol > 0
        sumPerCol = np.asarray(self.dataMatrix.sum(axis=0)).ravel()

        colAverage = np.zeros_like(sumPerCol)
        colAverage[nonzeroCols] = sumPerCol[nonzeroCols] / interactionsPerCol[nonzeroCols]

        # Split in blocks to avoid duplicating the whole data structure
        start_col = 0
        end_col = 0

        blockSize = 1000

        while end_col < self.n_columns:
            end_col = min(self.n_columns, end_col + blockSize)
            # average = 0 every column
            self.dataMatrix.data[self.dataMatrix.indptr[start_col]:self.dataMatrix.indptr[end_col]] -= \
                np.repeat(colAverage[start_col:end_col], interactionsPerCol[start_col:end_col])

            start_col += blockSize

    def useOnlyBooleanInteractions(self):
        # convert the rating to boolean
        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos = 0

        blockSize = 1000

        while end_pos < len(self.dataMatrix.data):
            end_pos = min(len(self.dataMatrix.data), end_pos + blockSize)

            self.dataMatrix.data[start_pos:end_pos] = np.ones(end_pos - start_pos)

            start_pos += blockSize

    def compute_similarity(self, start_col=None, end_col=None, block_size=100):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """

        values = []
        rows = []
        cols = []

        start_time = time.time()
        start_time_print_batch = start_time
        processedItems = 0

        if self.adjusted_cosine:
            self.applyAdjustedCosine()

        elif self.pearson_correlation:
            self.applyPearsonCorrelation()

        elif self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient:
            self.useOnlyBooleanInteractions()

        # We explore the matrix column-wise
        self.dataMatrix = self.dataMatrix.tocsc()

        # Compute sum of squared values to be used in normalization 对每一列求平方和组成array
        sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()

        # Tanimoto does not require the square root to be applied
        if not (self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient):
            sumOfSquared = np.sqrt(sumOfSquared)  # 取模

        if self.asymmetric_cosine:
            sumOfSquared_to_1_minus_alpha = np.power(sumOfSquared, 2 * (1 - self.asymmetric_alpha))
            sumOfSquared_to_alpha = np.power(sumOfSquared, 2 * self.asymmetric_alpha)

        self.dataMatrix = self.dataMatrix.tocsc()

        start_col_local = 0
        end_col_local = self.n_columns

        if start_col is not None and start_col > 0 and start_col < self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col > start_col_local and end_col < self.n_columns:
            end_col_local = end_col

        start_col_block = start_col_local

        this_block_size = 0

        # Compute all similarities for each item using vectorization
        while start_col_block < end_col_local:

            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block - start_col_block

            # All data points for a given item
            item_data = self.dataMatrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()

            # If only 1 feature avoid last dimension to disappear # 防止只取到了一列，扩展成二维
            if item_data.ndim == 1:
                item_data = np.atleast_2d(item_data)

            this_block_weights = self.dataMatrix.T.dot(item_data)  # (I, U)*(U, col)

            for col_index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_column_weights = this_block_weights
                else:
                    this_column_weights = this_block_weights[:, col_index_in_block]  # 取一列数据

                columnIndex = col_index_in_block + start_col_block
                this_column_weights[columnIndex] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:
                    if self.asymmetric_cosine:
                        denominator = sumOfSquared_to_alpha[
                                          columnIndex] * sumOfSquared_to_1_minus_alpha + self.shrink + 1e-6
                    else:
                        denominator = sumOfSquared[columnIndex] * sumOfSquared + self.shrink + 1e-6

                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)


                # Apply the specific denominator for Tanimoto
                elif self.tanimoto_coefficient:
                    denominator = sumOfSquared[columnIndex] + sumOfSquared - this_column_weights + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.dice_coefficient:
                    denominator = sumOfSquared[columnIndex] + sumOfSquared + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.tversky_coefficient:
                    denominator = this_column_weights + \
                                  (sumOfSquared[columnIndex] - this_column_weights) * self.tversky_alpha + \
                                  (sumOfSquared - this_column_weights) * self.tversky_beta + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                # If no normalization or tanimoto is selected, apply only shrink
                elif self.shrink != 0:
                    this_column_weights = this_column_weights / self.shrink

                # this_column_weights = this_column_weights.toarray().ravel()

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.TopK - 1)[0:self.TopK]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_column_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_column_weights[top_k_idx][notZerosMask])
                rows.extend(top_k_idx[notZerosMask])
                cols.extend(np.ones(numNotZeros) * columnIndex)

            # Add previous block size
            processedItems += this_block_size

            if time.time() - start_time_print_batch >= 30 or end_col_block == end_col_local:
                columnPerSec = processedItems / (time.time() - start_time + 1e-9)

                print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / (end_col_local - start_col_local) * 100, columnPerSec,
                                    (time.time() - start_time) / 60))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()

            start_col_block += block_size

        # End while on columns

        W_sparse = sps.csr_matrix((values, (rows, cols)),
                                  shape=(self.n_columns, self.n_columns),
                                  dtype=np.float32)
        return W_sparse

class ItemKNN(AbstractRecommender):
    """
    Using several methods to measure the similarity between each item
    """

    def __init__(self, sess, dataset, conf):
        super(ItemKNN, self).__init__(dataset, conf)
        self.verbose = conf['verbose']
        self.topK = conf['neighbor']
        self.shrink = conf['shrink']
        self.dataset = dataset
        self.train_matrix = self.dataset.train_matrix.tocsc()
        self.similarity = conf['similarity']
        self.asymmetric_alpha = conf['asymmetric_alpha']
        self.tversky_alpha = conf['tversky_alpha']
        self.tversky_beta = conf['tversky_beta']
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

    def build_graph(self):
        similarity = Compute_Similarity(self.train_matrix, shrink=self.shrink, topK=self.topK, normalize=True,
                                        similarity=self.similarity, asymmetric_alpha=self.asymmetric_alpha, tversky_alpha=self.tversky_alpha, tversky_beta=self.tversky_beta,)
        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = self.W_sparse.tocsr()
        self.ratings = self.train_matrix.dot(self.W_sparse).toarray()

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        self.logger.info(self.evaluate())

    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items=None):
        if candidate_items is None:
            return self.ratings[user_ids]
        else:
            ratings = None
            """waiting to complete"""
        return ratings
