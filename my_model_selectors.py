import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        n_state_scores = []
        # Iterate over the given range of hidden states to find the best model
        for n_states in range(self.min_n_components, self.max_n_components + 1):

            try:
                model = self.base_model(n_states)
                logL = model.score(self.X, self.lengths)
                N = len(self.X)

                # According to a Udacity slack discussion
                # (https://ai-nd.slack.com/archives/C3TSZ56U8/p1501322009967996)
                # p = n^2 + 2*d*n - 1
                # where n = n_states
                # and d = len(self.X[0])
                p = n_states**2 + 2 * len(self.X[0]) * n_states - 1

                BIC = -2 * logL + p * np.log(N)
                n_state_scores.append((BIC, model))
            except Exception as e:
                print(e)
                pass

        if len(n_state_scores)==0:
            return None

        return max(n_state_scores, key=lambda x: x[0])[1]




class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        split_method = KFold(shuffle=True, random_state=self.random_state)

        n_state_scores = []
        # Iterate over the given range of hidden states to find the best model
        for n_states in range(self.min_n_components, self.max_n_components+1):

            try:
                cv_scores = []

                if len(self.sequences) < 3:

                    model = self.base_model(n_states)
                    n_state_scores.append((model.score(self.X, self.lengths), model))

                else:
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                        self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        model = self.base_model(n_states)
                        cv_scores.append(model.score(X_test, lengths_test))

                n_state_scores.append((np.mean(cv_scores), model))

            except Exception as e:
                print(e)
                pass

        if len(n_state_scores)==0:
            return None

        return max(n_state_scores, key=lambda x: x[0])[1]
