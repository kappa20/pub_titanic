"""
Feature engineering functions for the Titanic dataset.
This module replicates the exact feature engineering from the notebook (cell 16).
"""
import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Create new features for improved prediction.

    This function must match the exact feature engineering used during model training.

    Args:
        df: DataFrame with raw passenger data

    Returns:
        DataFrame with engineered features added
    """
    df_copy = df.copy()

    # 1. Family Size = SibSp + Parch + 1
    df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1

    # 2. Is Alone = 1 if traveling alone, 0 otherwise
    df_copy['IsAlone'] = (df_copy['FamilySize'] == 1).astype(int)

    # 3. Title extraction from Name
    df_copy['Title'] = df_copy['Name'].str.extract(' ([A-Za-z]+)\.')[0]
    # Group rare titles (those appearing less than 10 times in training)
    title_counts = df_copy['Title'].value_counts()
    rare_titles = title_counts[title_counts < 10].index
    df_copy['Title'] = df_copy['Title'].replace(rare_titles, 'Rare')

    # 4. Cabin Deck - Extract first letter of cabin number
    df_copy['Deck'] = df_copy['Cabin'].str[0]

    # 5. Has Cabin indicator - 1 if cabin info available, 0 otherwise
    df_copy['HasCabin'] = df_copy['Cabin'].notna().astype(int)

    # 6. Fare per person - divide fare by family size
    df_copy['FarePerPerson'] = df_copy['Fare'] / (df_copy['SibSp'] + df_copy['Parch'] + 1)

    return df_copy


def validate_input(passenger_data):
    """
    Validate user input data.

    Args:
        passenger_data: Dictionary with passenger information

    Returns:
        tuple: (is_valid, error_message)
    """
    errors = []

    # Check required fields
    required_fields = ['Name', 'Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']
    for field in required_fields:
        if field not in passenger_data or passenger_data[field] is None:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, "; ".join(errors)

    # Name validation removed - title is selected directly by user
    # The Name field is auto-generated with the correct format

    # Validate numeric ranges
    age = passenger_data['Age']
    if age < 0 or age > 100:
        errors.append("Age must be between 0 and 100")

    sibsp = passenger_data['SibSp']
    if sibsp < 0 or sibsp > 8:
        errors.append("Number of siblings/spouses must be between 0 and 8")

    parch = passenger_data['Parch']
    if parch < 0 or parch > 6:
        errors.append("Number of parents/children must be between 0 and 6")

    fare = passenger_data['Fare']
    if fare < 0:
        errors.append("Fare must be non-negative")

    # Validate categorical values
    if passenger_data['Sex'] not in ['male', 'female']:
        errors.append("Sex must be 'male' or 'female'")

    if passenger_data['Pclass'] not in [1, 2, 3]:
        errors.append("Passenger class must be 1, 2, or 3")

    if passenger_data['Embarked'] not in ['S', 'C', 'Q']:
        errors.append("Embarked must be 'S', 'C', or 'Q'")

    if errors:
        return False, "; ".join(errors)

    return True, ""


def create_passenger_dataframe(passenger_dict):
    """
    Convert passenger dictionary to DataFrame format expected by the model.

    Args:
        passenger_dict: Dictionary with passenger information

    Returns:
        DataFrame with single row containing passenger data
    """
    # Create DataFrame with all required columns
    df = pd.DataFrame([passenger_dict])

    # Ensure proper data types
    df['Pclass'] = df['Pclass'].astype(int)
    df['Age'] = df['Age'].astype(float)
    df['SibSp'] = df['SibSp'].astype(int)
    df['Parch'] = df['Parch'].astype(int)
    df['Fare'] = df['Fare'].astype(float)

    # Handle empty cabin as NaN
    if 'Cabin' in df.columns and df['Cabin'].iloc[0] == '':
        df['Cabin'] = np.nan

    return df


def get_title_from_name(name):
    """
    Extract title from passenger name.

    Args:
        name: Full passenger name (e.g., "Smith, Mr. John")

    Returns:
        Title string (e.g., "Mr")
    """
    match = pd.Series([name]).str.extract(' ([A-Za-z]+)\.')[0]
    if pd.notna(match.iloc[0]):
        return match.iloc[0]
    return "Unknown"
