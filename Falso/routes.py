# Important
import numpy as np
import pandas as pd
import os
from Falso import app
from Falso.preprocessing import preprocessing
from Falso.classifier import classifier as clsf

# Flask
from flask import Flask, jsonify, request


# API test method
@app.route("/falso/test")
def test():
    return jsonify('Falso API Status: Running'), 200


# Generate default models to be used
@app.route("/falso/generate_default")
def generate_default():
    # Check if model already exists
    exists = os.path.isfile('/FinalModels/CommonDefault/xgboost_default_pipeline.pickle')

    if exists:
        # File exists
        return jsonify('Models already exist'), 200
    else:
        # Create files
        # Get path to csv from query
        path_to_csv = request.args.get('path')

        # Get data from default training data CSV
        df = pd.read_csv(path_to_csv, encoding='latin')

        df = preprocessing.drop_unused_columns(df)
        df = preprocessing.process_reviews(df)
        df = preprocessing.encode_numeric(df, "train")
        classifier = clsf.train_classify(df)

        # Save classifier
        clsf.save_model('xgboost_default_pipeline', "default", classifier)

        # Return success message
        return jsonify('Default models created'), 200


# Classify given review
@app.route("/falso/classify", methods=['POST'])
def classify():
    # Get client name
    client = request.args.get('client')

    # Get json from request
    df = pd.DataFrame(request.json)

    # Process data
    df = preprocessing.process_reviews(df)
    df = preprocessing.encode_numeric(df, "")

    # Get saved default classifier
    if client is None or client == "default":
        classifier = clsf.read_model('/FinalModels/CommonDefault/xgboost_default_pipeline.pickle')
    else:
        classifier = clsf.read_model('/FinalModels/' + client + '/custom_model.pickle')

    # Calculate prediction
    prediction = classifier.predict(df[['REVIEW_TEXT', 'VERIFIED_PURCHASE', 'PRODUCT_CATEGORY', 'RATING']])

    # Calculate probability score
    probability = classifier.predict_proba(df[['REVIEW_TEXT', 'VERIFIED_PURCHASE', 'PRODUCT_CATEGORY', 'RATING']])[0]

    # Prediction as text
    if prediction[0] == 0:
        prediction_text = "fake"
    else:
        prediction_text = "real"

    return jsonify(
        {"Category": prediction_text, "Confidence": {"Fake": probability[0] * 100, "Real": probability[1] * 100}})


# Generate custom model based on customer
@app.route("/falso/custom_model", methods=['GET'])
def create_custom_model():
    # Get client name
    client = request.args.get('client')

    # Get csv path from query
    path_to_csv = request.args.get('path')

    # Get data from default training data CSV
    df = pd.read_csv(path_to_csv, encoding='latin')

    # Process data
    ndf = df.copy(deep=True)
    ndf = preprocessing.process_reviews(ndf)
    ndf = preprocessing.encode_numeric(ndf, "")

    # Get saved default classifier
    classifier = clsf.read_model('/FinalModels/CommonDefault/xgboost_default_pipeline.pickle')

    # Calculate prediction
    prediction = classifier.predict(ndf[['REVIEW_TEXT', 'VERIFIED_PURCHASE', 'PRODUCT_CATEGORY', 'RATING']])
    probability = classifier.predict_proba(ndf[['REVIEW_TEXT', 'VERIFIED_PURCHASE', 'PRODUCT_CATEGORY', 'RATING']])

    df['LABEL'] = prediction
    ndf['LABEL'] = prediction

    classifier_new = clsf.train_classify(ndf)
    clsf.save_model("custom_model", client, classifier_new)

    df['LABEL'] = np.where(df['LABEL'] == 0, 'fake', df['LABEL'])
    df['LABEL'] = np.where(df['LABEL'] == 1, 'real', df['LABEL'])

    df['FAKE_SCORE'] = probability[:, 0] * 100
    df['REAL_SCORE'] = probability[:, 1] * 100

    return df.to_json(orient='records')
