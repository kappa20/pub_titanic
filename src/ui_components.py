"""
UI components for the Streamlit app.
"""
import streamlit as st
import pandas as pd

from config.app_config import (
    DEFAULT_VALUES, CLASS_MAPPING, EMBARKATION_MAPPING,
    COLOR_SUCCESS, COLOR_DANGER, VALIDATION_RANGES, FARE_RANGES_BY_CLASS
)
from src.model_utils import get_feature_explanation


def render_input_form():
    """
    Render the input form for passenger information.

    Returns:
        Dictionary with all passenger inputs
    """
    passenger_data = {}

    # Personal Information Section
    st.subheader("Personal Information")

    passenger_data['Sex'] = st.radio(
        "Sex",
        options=['male', 'female'],
        index=0 if DEFAULT_VALUES['sex'] == 'male' else 1,
        horizontal=True
    )

    # Title options filtered by sex
    if passenger_data['Sex'] == 'male':
        title_options = ['Mr', 'Master', 'Dr', 'Rev', 'Col', 'Major', 'Rare']
    else:
        title_options = ['Mrs', 'Miss', 'Dr', 'Rare']

    title = st.selectbox(
        "Title",
        options=title_options,
        index=0,
        help="Social title - affects survival prediction"
    )
    # Create a dummy name with the selected title for feature engineering
    passenger_data['Name'] = f"Passenger, {title}. Test"

    passenger_data['Age'] = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=DEFAULT_VALUES['age'],
        step=1,
        help="Age in years"
    )

    pclass_display = st.selectbox(
        "Passenger Class",
        options=list(CLASS_MAPPING.keys()),
        index=2,  # Default to 3rd Class
        help="1st Class: Upper class, 2nd Class: Middle class, 3rd Class: Lower class"
    )
    passenger_data['Pclass'] = CLASS_MAPPING[pclass_display]

    fare_min, fare_max, fare_default, fare_step, fare_hint = FARE_RANGES_BY_CLASS[passenger_data['Pclass']]
    passenger_data['Fare'] = st.number_input(
        "Ticket Fare (£)",
        min_value=fare_min,
        max_value=fare_max,
        value=fare_default,
        step=fare_step,
        help=fare_hint
    )
    st.caption(f"Range for {pclass_display}: £{fare_min:.0f} – £{fare_max:.0f}")

    # Family Information Section
    st.subheader("Family Information")

    passenger_data['SibSp'] = st.slider(
        "Number of Siblings/Spouses Aboard",
        min_value=0,
        max_value=8,
        value=DEFAULT_VALUES['sibsp'],
        help="Number of siblings or spouses traveling with you"
    )

    passenger_data['Parch'] = st.slider(
        "Number of Parents/Children Aboard",
        min_value=0,
        max_value=6,
        value=DEFAULT_VALUES['parch'],
        help="Number of parents or children traveling with you"
    )

    # Show calculated family size
    family_size = passenger_data['SibSp'] + passenger_data['Parch'] + 1
    st.info(f"Total family size: {family_size} person(s)")

    # Travel Details Section
    st.subheader("Travel Details")

    passenger_data['Cabin'] = st.text_input(
        "Cabin (optional)",
        value=DEFAULT_VALUES['cabin'],
        help="Cabin number if known (e.g., C85, E46). Leave empty if unknown."
    )

    embarked_display = st.selectbox(
        "Port of Embarkation",
        options=list(EMBARKATION_MAPPING.keys()),
        index=0,  # Default to Southampton
        help="Port where passenger boarded the Titanic"
    )
    passenger_data['Embarked'] = EMBARKATION_MAPPING[embarked_display]

    return passenger_data


def display_prediction_result(prediction, probability):
    """
    Display the prediction result in a prominent card.

    Args:
        prediction: 0 or 1
        probability: Float between 0 and 1
    """
    if prediction == 1:
        outcome = "SURVIVED"
        emoji = "✅"
        color = COLOR_SUCCESS
        prob_text = f"{probability * 100:.1f}%"
    else:
        outcome = "DID NOT SURVIVE"
        emoji = "❌"
        color = COLOR_DANGER
        prob_text = f"{(1 - probability) * 100:.1f}%"

    # Create a styled card using markdown
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22 0%, {color}44 100%);
            border-left: 5px solid {color};
            border-radius: 10px;
            padding: 30px;
            margin: 20px 0;
            text-align: center;
        ">
            <h1 style="color: {color}; margin: 0; font-size: 3em;">{emoji}</h1>
            <h2 style="color: #333; margin: 10px 0;">{outcome}</h2>
            <p style="font-size: 1.2em; color: #666; margin: 5px 0;">
                Survival Probability: <strong>{prob_text}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_feature_importance(importance_df, top_n=7):
    """
    Display feature importance with explanations.

    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to display
    """
    st.subheader("Key Factors Influencing Prediction")

    # Get top features
    top_features = importance_df.head(top_n)

    # Display each feature with explanation
    for idx, row in top_features.iterrows():
        feature_name = row['feature']
        importance = row['importance_pct']
        coefficient = row['coefficient']

        # Get explanation
        explanation = get_feature_explanation(feature_name, coefficient)

        # Determine color
        color = COLOR_SUCCESS if coefficient > 0 else COLOR_DANGER

        # Display feature
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{feature_name}**")
            st.caption(explanation)
        with col2:
            st.metric("Impact", f"{importance:.1f}%")


def display_passenger_profile(profile_data):
    """
    Display passenger profile information.

    Args:
        profile_data: Dictionary with passenger profile sections
    """
    st.subheader("Passenger Profile")

    # Display each section
    for section_name, section_data in profile_data.items():
        with st.expander(section_name, expanded=False):
            for key, value in section_data.items():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{key}:**")
                with col2:
                    st.markdown(f"{value}")


def display_comparison_metrics(comparisons):
    """
    Display comparison metrics in columns.

    Args:
        comparisons: Dictionary with comparison data
    """
    if not comparisons:
        return

    st.subheader("Comparison with Historical Data")

    cols = st.columns(len(comparisons))

    for idx, (metric_name, metric_data) in enumerate(comparisons.items()):
        with cols[idx]:
            value = metric_data['value']
            delta = metric_data.get('delta', None)
            delta_text = metric_data.get('delta_text', '')

            if metric_name == 'Age':
                st.metric(
                    "Your Age",
                    f"{value:.0f} years",
                    delta=delta_text
                )
            elif metric_name == 'Fare':
                st.metric(
                    "Your Fare",
                    f"£{value:.2f}",
                    delta=delta_text
                )
            elif metric_name == 'Class Survival Rate':
                st.metric(
                    "Your Class Survival Rate",
                    f"{value:.1f}%"
                )


def display_model_info():
    """
    Display information about the model in an expander.
    """
    with st.expander("About the Model"):
        st.markdown("""
        ### Model Information

        **Model Type:** Logistic Regression

        **Accuracy:** 82.68% on validation set

        **AUC-ROC:** 0.8679

        **Training Data:** 891 Titanic passengers with known survival outcomes

        **Features Used:**
        - Passenger Class, Sex, Age
        - Family Size (Siblings/Spouses, Parents/Children)
        - Ticket Fare
        - Cabin Information
        - Port of Embarkation
        - Engineered features (Title, Deck, etc.)

        **How it works:**
        The model analyzes patterns from historical Titanic passenger data to predict
        survival probability. It considers factors like passenger class, sex, age,
        and family relationships that historically influenced survival rates.

        **Note:** This is an educational demonstration based on historical data.
        """)


def display_about_section():
    """
    Display about section in the sidebar.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About

    This app uses machine learning to predict passenger survival
    on the RMS Titanic based on historical data.

    **Data Source:** Kaggle Titanic Dataset

    **Purpose:** Educational demonstration of ML model deployment

    **Accuracy:** 82.68%
    """)


def display_instructions():
    """
    Display usage instructions.
    """
    st.info("""
    **How to use:**
    1. Fill in the passenger information in the sidebar
    2. Click 'Predict Survival' to see the prediction
    3. View the survival probability and key influencing factors
    """)
