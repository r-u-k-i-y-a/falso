# Important
import pickle
import os
from Falso.custom_transformers import TextSelector, NumberSelector

# SK Learn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# XGBoost separate package
from xgboost import XGBClassifier


class classifier:
    @staticmethod
    def train_classify(df):
        # Tfidf to convert words to a numeric format based on term frequency
        tfidf = TfidfVectorizer(ngram_range=(1, 3))

        # tsvd for dimensionality reduction
        tsvd = TruncatedSVD(algorithm='randomized', n_components=300)

        # XGBoost classifier
        clf = XGBClassifier(max_depth=8, n_estimators=300, learning_rate=0.1, subsample=0.8)

        # Generate pipeline with combined features
        clsf = classifier.get_pipeline(tfidf, tsvd, clf)

        # Train classifier
        clsf.fit(df[['REVIEW_TEXT', 'VERIFIED_PURCHASE', 'PRODUCT_CATEGORY', 'RATING']], df[['LABEL']].values)

        return clsf

    @staticmethod
    def get_pipeline(tfidf, tsvd, clf):
        pipe = Pipeline([
            ('features', FeatureUnion([
                ('review_text', Pipeline([
                    ('review', TextSelector('REVIEW_TEXT')),
                    ('tfidf', tfidf),
                    ('svd', tsvd),
                ])),
                ('category', Pipeline([
                    ('product', NumberSelector('PRODUCT_CATEGORY')),
                ])),
                ('verified', Pipeline([
                    ('purchase', NumberSelector('VERIFIED_PURCHASE')),
                ])),
                ('rating', Pipeline([
                    ('score', NumberSelector('RATING')),
                ])),
            ])),
            ('clf', clf),
        ])

        return pipe

    @staticmethod
    def save_model(name, client, model):
        # Create file model save path if not exists
        if client == "default":
            model_path = "/FinalModels/CommonDefault"
        else:
            model_path = "/FinalModels/" + client

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save models as pickle files
        pickle.dump(model, open(model_path + '/' + name + '.pickle', 'wb'))

    @staticmethod
    def read_model(path):
        clf = pickle.load(open(path, 'rb'))
        return clf
