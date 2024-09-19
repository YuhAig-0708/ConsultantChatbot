from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from transformer.feature_transformer import FeatureTransformer
from sklearn.neighbors import KNeighborsClassifier


class KNeighborsClassifier_Model(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()), # Word Segmentation - ViTokenizer + ViPostagger
            ("vect", CountVectorizer()), # Data preprocessing - BoW
            ("tfidf", TfidfTransformer()), # Data preprocessing - TFIDF
            ("clf", KNeighborsClassifier()) # kNN model configuration
        ])

        return pipe_line