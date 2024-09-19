from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from transformer.feature_transformer import FeatureTransformer


class NaiveBayes_Model(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()), # Word Segmentation - ViTokenizer + ViPostagger
            ("vect", CountVectorizer()), # Data preprocessing - BoW
            ("tfidf", TfidfTransformer()), # Data preprocessing - TFIDF
            ("clf", MultinomialNB(alpha=0.1)) # NBs model configuration
        ])

        return pipe_line