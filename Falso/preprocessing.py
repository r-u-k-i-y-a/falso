# Important
import numpy as np
from collections import defaultdict
from Falso.classifier import classifier as clsf

# NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# SK Learn
from sklearn.preprocessing import LabelEncoder

class preprocessing:
    @staticmethod
    def drop_unused_columns(df):
        # Drop unused columns
        df = df.drop(['DOC_ID', 'PRODUCT_ID', 'PRODUCT_TITLE', 'REVIEW_TITLE'], axis=1)
        return df

    @staticmethod
    def process_reviews(df):
        # Remove empty rows if exist
        df['REVIEW_TEXT'].dropna(inplace=True)

        # Convert to lower case
        df['REVIEW_TEXT'] = [review.lower() for review in df['REVIEW_TEXT']]

        # Tokenize words
        df['REVIEW_TEXT'] = [word_tokenize(review) for review in df['REVIEW_TEXT']]

        return preprocessing.perform_text_cleaning(df)

    @staticmethod
    def perform_text_cleaning(df):
        # POS tags for word lemmatizer to understand Verbs, Adjectives, Adverbs and Nouns
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        # Loop through all reviews one by one
        for index, entry in enumerate(df['REVIEW_TEXT']):
            # List to store the final words
            final_words = []

            # Word lemmatizer
            word_lemmatized = WordNetLemmatizer()

            # Tag words as Verbs, Adjectives, Adverbs or Nouns
            for word, tag in pos_tag(entry):
                # Check for stop words and alpha numeric characters
                if word not in stopwords.words('english') and word.isalpha():
                    word_final = word_lemmatized.lemmatize(word, tag_map[tag[0]])
                    final_words.append(word_final)

            # Final words saved back in df
            df.loc[index, 'REVIEW_TEXT'] = str(final_words)

        return df

    @staticmethod
    def encode_numeric(df, mode):
        if mode == "train":
            # Manually encode labels to avoid confusion when classifying
            df['LABEL'] = np.where(df['LABEL'] == 'fake', 0, df['LABEL'])
            df['LABEL'] = np.where(df['LABEL'] == 'real', 1, df['LABEL'])

            # Product Category
            encoder1 = LabelEncoder()
            encoder1.fit(df['PRODUCT_CATEGORY'])
            clsf.save_model("encoder1", "default", encoder1)

            # Verified Purchase
            encoder2 = LabelEncoder()
            encoder2.fit(df['VERIFIED_PURCHASE'])
            clsf.save_model("encoder2", "default", encoder2)
        else:
            encoder1 = clsf.read_model("/FinalModels/CommonDefault/encoder1.pickle")
            encoder2 = clsf.read_model("/FinalModels/CommonDefault/encoder2.pickle")

        df['PRODUCT_CATEGORY'] = encoder1.transform(df['PRODUCT_CATEGORY'])
        df['VERIFIED_PURCHASE'] = encoder2.transform(df['VERIFIED_PURCHASE'])

        return df
