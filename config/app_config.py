"""
Configuration constants for the Titanic Survival Predictor app.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'best_model.pkl'
TRAIN_PATH = BASE_DIR / 'titanic' / 'train.csv'

# Color Scheme (Titanic-themed)
COLOR_PRIMARY = '#1E3A5F'  # Deep ocean blue
COLOR_SUCCESS = '#28A745'  # Green for survival
COLOR_DANGER = '#DC3545'   # Red for death
COLOR_ACCENT = '#FFD700'   # Gold
COLOR_BACKGROUND = '#F8F9FA'  # Light gray

# Model Configuration
RANDOM_STATE = 42
OVERALL_SURVIVAL_RATE = 0.384  # 38.4% from training data

# Embarkation Port Mappings
EMBARKATION_MAPPING = {
    'Southampton': 'S',
    'Cherbourg': 'C',
    'Queenstown': 'Q'
}

EMBARKATION_REVERSE = {
    'S': 'Southampton',
    'C': 'Cherbourg',
    'Q': 'Queenstown'
}

# Passenger Class Mappings
CLASS_MAPPING = {
    '1st Class': 1,
    '2nd Class': 2,
    '3rd Class': 3
}

CLASS_REVERSE = {
    1: '1st Class',
    2: '2nd Class',
    3: '3rd Class'
}

# Feature Descriptions (for tooltips and explanations)
FEATURE_DESCRIPTIONS = {
    'Pclass': 'Passenger class: 1st (upper), 2nd (middle), 3rd (lower)',
    'Sex_male': 'Being male significantly decreased survival chances',
    'Age': 'Children and young adults had higher survival rates',
    'SibSp': 'Number of siblings/spouses aboard',
    'Parch': 'Number of parents/children aboard',
    'Fare': 'Higher fare indicates better accommodations and survival chances',
    'Embarked': 'Port of embarkation (Southampton, Cherbourg, Queenstown)',
    'FamilySize': 'Total family members aboard (optimal size: 2-4)',
    'IsAlone': 'Traveling alone decreased survival chances',
    'Title_Mr': 'Adult males (Mr.) had the lowest survival rate',
    'Title_Mrs': 'Married women (Mrs.) had higher survival priority',
    'Title_Miss': 'Unmarried women (Miss) had high survival rates',
    'Title_Master': 'Young boys (Master) had high survival priority',
    'Title_Rare': 'Rare titles (Dr, Rev, Col, etc.) had varied survival rates',
    'Deck': 'Cabin deck location (A-F), higher decks closer to lifeboats',
    'HasCabin': 'Having cabin information indicates higher class',
    'FarePerPerson': 'Individual fare paid per family member'
}

# Default Input Values
DEFAULT_VALUES = {
    'name': 'Smith, Mr. John',
    'sex': 'male',
    'age': 30,
    'pclass': '3rd Class',
    'sibsp': 0,
    'parch': 0,
    'fare': 15.0,
    'cabin': '',
    'embarked': 'Southampton'
}

# Validation Ranges (from training data)
VALIDATION_RANGES = {
    'age': (0, 100),
    'sibsp': (0, 8),
    'parch': (0, 6),
    'fare': (0, 600),
}

# Fare ranges per passenger class (derived from training data)
# Format: (min, max, default, step, help_hint)
FARE_RANGES_BY_CLASS = {
    1: (0.0, 513.0, 60.0, 1.0, "1st Class typical range: £26 – £233"),
    2: (0.0,  74.0, 14.0, 0.5, "2nd Class typical range: £11 – £41"),
    3: (0.0,  70.0,  8.0, 0.5, "3rd Class typical range: £7 – £40"),
}

# Feature names after preprocessing
NUMERICAL_FEATURES = [
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
    'FamilySize', 'IsAlone', 'HasCabin', 'FarePerPerson'
]

CATEGORICAL_FEATURES = ['Sex', 'Embarked', 'Title', 'Deck']

# All features required for model input
ALL_FEATURES = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked', 'FamilySize', 'IsAlone', 'Title', 'Deck',
    'HasCabin', 'FarePerPerson'
]
