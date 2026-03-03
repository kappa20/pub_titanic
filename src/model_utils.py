"""
Model utilities for loading the trained model and making predictions.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sklearn.compose._column_transformer as ct

from src.feature_engineering import engineer_features
from config.app_config import ALL_FEATURES


# Compatibility shim for scikit-learn versions with different internal classes
if not hasattr(ct, '_RemainderColsList'):
    class _RemainderColsList(list):
        """Compatibility class for older sklearn pickles."""
        pass
    ct._RemainderColsList = _RemainderColsList

# Add _Scorer compatibility if missing
try:
    import sklearn.metrics._scorer as scorer_module
    if not hasattr(scorer_module, '_Scorer'):
        class _Scorer:
            """Compatibility class for sklearn metrics._scorer."""
            pass
        scorer_module._Scorer = _Scorer
except (ImportError, AttributeError):
    pass


def load_model(model_path):
    """
    Load the trained model from disk.

    The saved model is a GridSearchCV object, so we need to extract
    the best_estimator_ to get the actual Pipeline.

    Args:
        model_path: Path to the saved model file

    Returns:
        tuple: (pipeline, preprocessor, classifier)
    """
    try:
        # Load the GridSearchCV object
        grid_search = joblib.load(model_path)

        # Extract the best pipeline
        pipeline = grid_search.best_estimator_

        # Extract components
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']

        return pipeline, preprocessor, classifier

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def get_feature_importance(pipeline, preprocessor, classifier):
    """
    Extract feature importance from Logistic Regression coefficients.

    Args:
        pipeline: The full sklearn Pipeline
        preprocessor: The preprocessor from the pipeline
        classifier: The trained Logistic Regression classifier

    Returns:
        DataFrame with feature names, coefficients, and importance percentages
    """
    try:
        # Get coefficients from Logistic Regression
        coefficients = classifier.coef_[0]

        # Get numerical feature names (first 9 features)
        num_features = preprocessor.transformers_[0][2]  # Column names from numeric transformer

        # Get categorical feature names (one-hot encoded)
        cat_transformer = preprocessor.transformers_[1][1]
        cat_features_raw = preprocessor.transformers_[1][2]  # Original categorical column names

        # Get one-hot encoded feature names
        cat_feature_names = cat_transformer.named_steps['onehot'].get_feature_names_out(cat_features_raw)

        # Combine all feature names
        all_feature_names = list(num_features) + list(cat_feature_names)

        # Ensure we have the right number of features
        n_features = min(len(all_feature_names), len(coefficients))
        all_feature_names = all_feature_names[:n_features]
        coefficients = coefficients[:n_features]

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'coefficient': coefficients,
            'abs_importance': np.abs(coefficients)
        })

        # Calculate percentage importance
        total_importance = importance_df['abs_importance'].sum()
        if total_importance > 0:
            importance_df['importance_pct'] = (importance_df['abs_importance'] / total_importance) * 100
        else:
            importance_df['importance_pct'] = 0

        # Sort by absolute importance
        importance_df = importance_df.sort_values('abs_importance', ascending=False)

        return importance_df

    except Exception as e:
        raise RuntimeError(f"Failed to extract feature importance: {str(e)}")


def predict_survival(pipeline, passenger_data_dict):
    """
    Make a survival prediction for a passenger.

    Args:
        pipeline: The trained model pipeline
        passenger_data_dict: Dictionary with passenger information

    Returns:
        tuple: (prediction, probability, confidence)
            - prediction: 0 (did not survive) or 1 (survived)
            - probability: float between 0 and 1 (probability of survival)
            - confidence: float between 0 and 1 (confidence in prediction)
    """
    try:
        # Create DataFrame from input dictionary
        passenger_df = pd.DataFrame([passenger_data_dict])

        # Apply feature engineering
        passenger_df = engineer_features(passenger_df)

        # Select only the features needed for the model
        X = passenger_df[ALL_FEATURES]

        # Make prediction
        prediction = pipeline.predict(X)[0]

        # Get probability estimates
        probabilities = pipeline.predict_proba(X)[0]

        # Probability of survival (class 1)
        survival_probability = probabilities[1]

        # Confidence is the max probability
        confidence = max(probabilities)

        return int(prediction), float(survival_probability), float(confidence)

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")


def get_feature_explanation(feature_name, coefficient):
    """
    Generate plain-English explanation for a feature's impact.

    Args:
        feature_name: Name of the feature
        coefficient: Coefficient value from Logistic Regression

    Returns:
        String explanation
    """
    explanations = {
        'Pclass': "Lower class number (1st class) increases survival chances",
        'Sex_male': "Being male significantly decreased survival chances",
        'Sex_female': "Being female significantly increased survival chances",
        'Age': "Younger passengers (children) had higher survival rates",
        'SibSp': "Number of siblings/spouses aboard affected survival",
        'Parch': "Number of parents/children aboard affected survival",
        'Fare': "Higher fare (better accommodations) increased survival",
        'FamilySize': "Moderate family size (2-4 members) had best survival rates",
        'IsAlone': "Traveling alone decreased survival chances",
        'HasCabin': "Having cabin information indicates higher class and survival",
        'FarePerPerson': "Individual fare per person affected survival",
        'Title_Mr': "Adult males (Mr.) had lowest survival rate",
        'Title_Mrs': "Married women (Mrs.) had high survival priority",
        'Title_Miss': "Unmarried women (Miss) had high survival rates",
        'Title_Master': "Young boys (Master) had high survival priority",
        'Title_Rare': "Rare titles had varied survival rates",
        'Embarked_C': "Embarking from Cherbourg affected survival",
        'Embarked_Q': "Embarking from Queenstown affected survival",
        'Embarked_S': "Embarking from Southampton affected survival",
    }

    # Try to find exact match
    if feature_name in explanations:
        base_explanation = explanations[feature_name]
    # Try to find partial match for Deck features
    elif feature_name.startswith('Deck_'):
        deck_letter = feature_name.split('_')[1]
        base_explanation = f"Being on Deck {deck_letter} affected survival chances"
    else:
        base_explanation = f"Feature {feature_name} influenced the prediction"

    # Add direction indicator
    if coefficient > 0:
        return f"✓ {base_explanation} (positive impact)"
    else:
        return f"✗ {base_explanation} (negative impact)"


def get_prediction_summary(prediction, probability):
    """
    Generate a summary message for the prediction.

    Args:
        prediction: 0 or 1
        probability: Survival probability

    Returns:
        tuple: (outcome_text, emoji, color)
    """
    if prediction == 1:
        outcome = "SURVIVED"
        emoji = "✓"
        color = "success"
        confidence_text = f"{probability * 100:.1f}% chance of survival"
    else:
        outcome = "DID NOT SURVIVE"
        emoji = "✗"
        color = "error"
        confidence_text = f"{(1 - probability) * 100:.1f}% chance of not surviving"

    return outcome, emoji, color, confidence_text
