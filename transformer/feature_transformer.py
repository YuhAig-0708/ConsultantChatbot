from pyvi import ViTokenizer, ViPosTagger
from sklearn.base import TransformerMixin, BaseEstimator


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self): # hàm khởi tạo
        self.tokenizer = ViTokenizer # call ViTokenizer for data preprocessing 
        self.pos_tagger = ViPosTagger # call ViPosTagger for data preprocessing 

    # override fit() and transform() of FeatureTransformer class
    def fit(self, *_):
        return self

    def transform(self, X, y=None, **fit_params):
        result = X.apply(lambda text: self.tokenizer.tokenize(text))
        return result
