from sklearn_crfsuite import CRF as CRFModel
import sys
import os
from sklearn_crfsuite.utils import flatten

ROOT_PATH_FOLDER = os.path.dirname(os.getcwd())
sys.path.append(ROOT_PATH_FOLDER + '/packages/CharacterIdentificationModule/')
from .BaseModel import BaseModel


class ConditionalRandomFields(BaseModel):
    def __init__(self,
                 feature_encoding,
                 crf_hyperparameters={
                     'algorithm': 'lbfgs',
                     'c1': 0.01,
                     'c2': 0.1,
                     'max_iteration': 80,
                 }
                 ):
        self.CRF = CRFModel(
            algorithm=crf_hyperparameters['algorithm'],
            c1=float(crf_hyperparameters['c1']),
            c2=float(crf_hyperparameters['c2']),
            max_iterations=int(crf_hyperparameters['max_iteration']),
            epsilon=1e-5,
            all_possible_states=True,
            all_possible_transitions=True
        )
        BaseModel.__init__(self,
                           model_clf=self.CRF,
                           feature_encoding=feature_encoding
                           )

    def fit(self, data_train_df):
        self.SEQ_DATA_TRAIN = ConditionalRandomFields.dataframe2sequential(
            data_train_df)
        self.X_TRAIN = [ConditionalRandomFields.sentence2features(
            sentence, self.FEATURE_ENCODING) for sentence in self.SEQ_DATA_TRAIN]
        self.Y_TRAIN = [ConditionalRandomFields.sentence2labels(
            sentence) for sentence in self.SEQ_DATA_TRAIN]
        # train CRF
        try:
            self.CRF.fit(self.X_TRAIN, self.Y_TRAIN)
        except AttributeError:
            raise AttributeError

        return self

    def predict(self, data_test_df):
        # prepare data input for training and testing data
        self.SEQ_DATA_TEST = ConditionalRandomFields.dataframe2sequential(
            data_test_df)
        self.X_TEST = [ConditionalRandomFields.sentence2features(
            sentence, self.FEATURE_ENCODING) for sentence in self.SEQ_DATA_TEST]
        self.Y_TEST = [ConditionalRandomFields.sentence2labels(
            sentence) for sentence in self.SEQ_DATA_TEST]

        # predict
        self.y_pred_train = self.CRF.predict(self.X_TRAIN)
        self.y_pred_test = self.CRF.predict(self.X_TEST)
        return self.y_pred_test

    def predict_sentence(self, sentence):
        token_sentence, X_features = super()._prepare_input_sentence(sentence)

        # predict
        y_pred = flatten(self.CRF.predict(X_features))
        token_with_predicted_tag = self.token_with_predicted_tags(
            token_sentence, y_pred)
        return y_pred, token_with_predicted_tag
    
    