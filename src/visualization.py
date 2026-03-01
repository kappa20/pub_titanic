"""
Visualization functions for the Streamlit app using Plotly.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from config.app_config import COLOR_SUCCESS, COLOR_DANGER, OVERALL_SURVIVAL_RATE


def create_probability_gauge(probability):
    """
    Create a gauge chart showing survival probability.

    Args:
        probability: Float between 0 and 1

    Returns:
        Plotly figure object
    """
    # Convert to percentage
    probability_pct = probability * 100

    # Determine color based on probability
    if probability_pct < 30:
        gauge_color = COLOR_DANGER
    elif probability_pct < 60:
        gauge_color = "#FFC107"  # Warning yellow
    else:
        gauge_color = COLOR_SUCCESS

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability_pct,
        title={'text': "Survival Probability", 'font': {'size': 24}},
        delta={
            'reference': OVERALL_SURVIVAL_RATE * 100,
            'valueformat': '.1f',
            'suffix': '%',
            'increasing': {'color': COLOR_SUCCESS},
            'decreasing': {'color': COLOR_DANGER}
        },
        number={'suffix': '%', 'font': {'size': 50}},
        gauge={
            'axis': {'range': [None, 100], 'ticksuffix': '%'},
            'bar': {'color': gauge_color, 'thickness': 0.75},
            'steps': [
                {'range': [0, 30], 'color': "rgba(220, 53, 69, 0.2)"},
                {'range': [30, 60], 'color': "rgba(255, 193, 7, 0.2)"},
                {'range': [60, 100], 'color': "rgba(40, 167, 69, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': OVERALL_SURVIVAL_RATE * 100
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial, sans-serif'}
    )

    return fig


def create_feature_importance_chart(importance_df, top_n=7):
    """
    Create a horizontal bar chart showing feature importance.

    Args:
        importance_df: DataFrame with 'feature', 'coefficient', 'importance_pct' columns
        top_n: Number of top features to display

    Returns:
        Plotly figure object
    """
    # Get top N features
    top_features = importance_df.head(top_n).copy()

    # Assign colors based on coefficient sign
    colors = [COLOR_SUCCESS if coef > 0 else COLOR_DANGER for coef in top_features['coefficient']]

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_features['importance_pct'],
        y=top_features['feature'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f"{val:.1f}%" for val in top_features['importance_pct']],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.1f}%<br>Coefficient: %{customdata:.3f}<extra></extra>',
        customdata=top_features['coefficient']
    ))

    fig.update_layout(
        title={
            'text': f'Top {top_n} Factors Influencing Prediction',
            'font': {'size': 20, 'family': 'Arial, sans-serif'}
        },
        xaxis_title='Importance (%)',
        yaxis_title='',
        height=max(300, top_n * 50),
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial, sans-serif'},
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(autorange="reversed")
    )

    return fig


def create_passenger_profile_card(passenger_data, engineered_features):
    """
    Create a formatted display of passenger information.

    Args:
        passenger_data: Dictionary with original passenger inputs
        engineered_features: Dictionary with engineered feature values

    Returns:
        Formatted HTML string for display
    """
    # Basic information
    sex = passenger_data.get('Sex', 'Unknown')
    age = passenger_data.get('Age', 'Unknown')
    pclass = passenger_data.get('Pclass', 'Unknown')

    # Travel information
    fare = passenger_data.get('Fare', 0)
    cabin = passenger_data.get('Cabin', 'Not specified')
    embarked = passenger_data.get('Embarked', 'Unknown')

    # Family information
    sibsp = passenger_data.get('SibSp', 0)
    parch = passenger_data.get('Parch', 0)
    family_size = engineered_features.get('FamilySize', sibsp + parch + 1)
    is_alone = engineered_features.get('IsAlone', 1 if family_size == 1 else 0)

    # Engineered features
    title = engineered_features.get('Title', 'Unknown')
    fare_per_person = engineered_features.get('FarePerPerson', fare)
    has_cabin = engineered_features.get('HasCabin', 0)

    # Build profile data structure
    profile = {
        'Basic Information': {
            'Title': title,
            'Sex': sex.capitalize(),
            'Age': f"{age} years",
            'Class': f"{pclass}"
        },
        'Travel Details': {
            'Ticket Fare': f"£{fare:.2f}",
            'Fare per Person': f"£{fare_per_person:.2f}",
            'Cabin': cabin if cabin else 'Not specified',
            'Embarked': embarked
        },
        'Family Information': {
            'Siblings/Spouses': sibsp,
            'Parents/Children': parch,
            'Family Size': family_size,
            'Traveling Alone': 'Yes' if is_alone == 1 else 'No'
        }
    }

    return profile


def create_comparison_metrics(passenger_data, train_data):
    """
    Create comparison metrics between passenger and training data statistics.

    Args:
        passenger_data: Dictionary with passenger information
        train_data: DataFrame with training data

    Returns:
        Dictionary with comparison metrics
    """
    comparisons = {}

    # Age comparison
    passenger_age = passenger_data.get('Age', None)
    if passenger_age is not None:
        mean_age = train_data['Age'].mean()
        age_delta = passenger_age - mean_age
        comparisons['Age'] = {
            'value': passenger_age,
            'delta': age_delta,
            'delta_text': f"{age_delta:+.1f} vs avg"
        }

    # Fare comparison
    passenger_fare = passenger_data.get('Fare', None)
    if passenger_fare is not None:
        mean_fare = train_data['Fare'].mean()
        fare_delta = passenger_fare - mean_fare
        comparisons['Fare'] = {
            'value': passenger_fare,
            'delta': fare_delta,
            'delta_text': f"{fare_delta:+.2f} vs avg"
        }

    # Class survival rate
    passenger_class = passenger_data.get('Pclass', None)
    if passenger_class is not None:
        class_survival_rate = train_data[train_data['Pclass'] == passenger_class]['Survived'].mean()
        comparisons['Class Survival Rate'] = {
            'value': class_survival_rate * 100,
            'delta': None,
            'delta_text': f"{class_survival_rate * 100:.1f}%"
        }

    return comparisons
