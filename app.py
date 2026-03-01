"""
Titanic Survival Predictor - Streamlit Web Application

This app uses a trained Logistic Regression model to predict passenger survival
on the Titanic based on various features.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.app_config import MODEL_PATH, TRAIN_PATH, COLOR_PRIMARY
from src.model_utils import (
    load_model, get_feature_importance, predict_survival, get_prediction_summary
)
from src.feature_engineering import validate_input, engineer_features
from src.visualization import (
    create_probability_gauge, create_feature_importance_chart,
    create_passenger_profile_card, create_comparison_metrics
)
from src.ui_components import (
    render_input_form, display_prediction_result, display_feature_importance,
    display_passenger_profile, display_comparison_metrics, display_model_info,
    display_about_section, display_instructions
)

# Page Configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_cached():
    """Load the trained model (cached for performance)."""
    try:
        pipeline, preprocessor, classifier = load_model(MODEL_PATH)
        return pipeline, preprocessor, classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


@st.cache_data
def load_train_data():
    """Load training data for comparisons (cached for performance)."""
    try:
        return pd.read_csv(TRAIN_PATH)
    except Exception as e:
        st.warning(f"Could not load training data for comparisons: {str(e)}")
        return None


def main():
    """Main application function."""

    # Header
    st.title("🚢 Titanic Survival Predictor")
    st.markdown(
        "Predict passenger survival on the RMS Titanic using Machine Learning"
    )

    # Display instructions
    display_instructions()

    # Load model and data
    with st.spinner("Loading model..."):
        pipeline, preprocessor, classifier = load_model_cached()
        train_data = load_train_data()

    # Sidebar - Input Collection
    with st.sidebar:
        st.header("Passenger Information")
        st.markdown("Fill in the details below:")

        # Render input form
        passenger_inputs = render_input_form()

        # Predict button
        st.markdown("---")
        predict_button = st.button("🔮 Predict Survival", type="primary", use_container_width=True)

        # About section
        display_about_section()

    # Main Area - Results Display
    if predict_button:
        # Validate inputs
        is_valid, error_message = validate_input(passenger_inputs)

        if not is_valid:
            st.error(f"Invalid input: {error_message}")
            st.stop()

        try:
            # Make prediction
            with st.spinner("Making prediction..."):
                prediction, probability, confidence = predict_survival(pipeline, passenger_inputs)

            # Display prediction result
            st.markdown("## Prediction Result")
            display_prediction_result(prediction, probability)

            # Create columns for gauge and metrics
            col1, col2 = st.columns([1, 1])

            with col1:
                # Display probability gauge
                st.markdown("### Survival Probability")
                gauge_fig = create_probability_gauge(probability)
                st.plotly_chart(gauge_fig, use_container_width=True)

            with col2:
                # Display comparison metrics if training data is available
                if train_data is not None:
                    display_comparison_metrics(
                        create_comparison_metrics(passenger_inputs, train_data)
                    )

            # Get feature importance
            st.markdown("---")
            st.markdown("## What Influenced This Prediction?")

            try:
                importance_df = get_feature_importance(pipeline, preprocessor, classifier)

                # Create two columns
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Display feature importance chart
                    importance_fig = create_feature_importance_chart(importance_df, top_n=7)
                    st.plotly_chart(importance_fig, use_container_width=True)

                with col2:
                    # Display top features with explanations
                    display_feature_importance(importance_df, top_n=7)

            except Exception as e:
                st.warning(f"Could not display feature importance: {str(e)}")

            # Display passenger profile
            st.markdown("---")
            try:
                # Apply feature engineering to get engineered features
                passenger_df = pd.DataFrame([passenger_inputs])
                engineered_df = engineer_features(passenger_df)

                # Extract engineered feature values
                engineered_features = {
                    'FamilySize': engineered_df['FamilySize'].iloc[0],
                    'IsAlone': engineered_df['IsAlone'].iloc[0],
                    'Title': engineered_df['Title'].iloc[0],
                    'HasCabin': engineered_df['HasCabin'].iloc[0],
                    'FarePerPerson': engineered_df['FarePerPerson'].iloc[0]
                }

                profile_data = create_passenger_profile_card(passenger_inputs, engineered_features)
                display_passenger_profile(profile_data)

            except Exception as e:
                st.warning(f"Could not display passenger profile: {str(e)}")

            # Model information
            st.markdown("---")
            display_model_info()

            # Success message
            st.success("Prediction completed successfully!")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)

    else:
        # Show welcome message when no prediction has been made yet
        st.markdown("---")
        st.markdown("""
        ### Welcome!

        This application uses a machine learning model trained on historical Titanic passenger data
        to predict survival probability.

        **To get started:**
        1. Enter passenger information in the sidebar
        2. Click the "Predict Survival" button
        3. View the prediction and explore the factors that influenced it

        **Features:**
        - Survival probability with confidence gauge
        - Feature importance analysis
        - Comparison with historical statistics
        - Detailed passenger profile
        """)

        # Display model information upfront
        display_model_info()

        # Show some statistics about the Titanic
        if train_data is not None:
            st.markdown("---")
            st.markdown("### Historical Titanic Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                survival_rate = train_data['Survived'].mean() * 100
                st.metric("Overall Survival Rate", f"{survival_rate:.1f}%")

            with col2:
                male_survival = train_data[train_data['Sex'] == 'male']['Survived'].mean() * 100
                st.metric("Male Survival Rate", f"{male_survival:.1f}%")

            with col3:
                female_survival = train_data[train_data['Sex'] == 'female']['Survived'].mean() * 100
                st.metric("Female Survival Rate", f"{female_survival:.1f}%")

            with col4:
                first_class_survival = train_data[train_data['Pclass'] == 1]['Survived'].mean() * 100
                st.metric("1st Class Survival Rate", f"{first_class_survival:.1f}%")


if __name__ == "__main__":
    main()
