from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from transformer.feature_transformer import FeatureTransformer
from sklearn.svm import SVC


class SVM_Model(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()), # Word Segmentation - ViTokenizer + ViPostagger
            ("vect", CountVectorizer()), # Data preprocessing - BoW
            ("tfidf", TfidfTransformer()), # Data preprocessing - TFIDF
            ("clf", SVC(kernel='sigmoid', C=5000.0, gamma=0.001, probability=True, class_weight='balanced')) # SVM model configuration
        ])

        return pipe_line