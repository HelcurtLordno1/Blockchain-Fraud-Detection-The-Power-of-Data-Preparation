"""
UI Components Module

Reusable UI components for the Ethereum Phishing Detection dashboard.

Author: Fraud Detection Research Team
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


def render_metric_card(
    title: str, 
    value: str, 
    delta: Optional[str] = None,
    card_type: str = "normal"
) -> None:
    """
    Render a styled metric card.
    
    Args:
        title: Card title
        value: Main metric value
        delta: Optional delta/change indicator
        card_type: 'normal', 'phisher', or 'warning'
        
    Example:
        >>> render_metric_card("Accuracy", "94.66%", "+2.3%", "normal")
    """
    card_class = f"metric-card {card_type}"
    
    delta_html = f"<p style='color: #10b981; margin: 0;'>{delta}</p>" if delta else ""
    
    st.markdown(f"""
    <div class="{card_class}">
        <h4 style='color: #cbd5e1; margin: 0; font-size: 0.9rem;'>{title}</h4>
        <h2 style='color: #f1f5f9; margin: 0.5rem 0;'>{value}</h2>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(label: str, status_type: str = "normal") -> None:
    """
    Render a status badge.
    
    Args:
        label: Badge text
        status_type: 'normal', 'phisher', or 'uncertain'
        
    Example:
        >>> render_status_badge("✅ NORMAL ACCOUNT", "normal")
    """
    st.markdown(f"""
    <div class='status-badge status-{status_type}'>
        {label}
    </div>
    """, unsafe_allow_html=True)


def render_info_box(title: str, content: str, icon: str = "ℹ️") -> None:
    """
    Render an information box.
    
    Args:
        title: Box title
        content: Box content (supports HTML)
        icon: Icon to display
        
    Example:
        >>> render_info_box("Note", "This is important information", "⚠️")
    """
    st.markdown(f"""
    <div class='info-box'>
        <h4>{icon} {title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)


def render_feature_box(feature_name: str, feature_value: float, unit: str = "") -> None:
    """
    Render a feature display box.
    
    Args:
        feature_name: Name of the feature
        feature_value: Feature value
        unit: Optional unit string
        
    Example:
        >>> render_feature_box("Transaction Count", 142, "txs")
    """
    st.markdown(f"""
    <div class='feature-box'>
        <strong>{feature_name}</strong>: {feature_value:.4f} {unit}
    </div>
    """, unsafe_allow_html=True)


def render_probability_gauge(probability: float, title: str = "Phishing Probability") -> None:
    """
    Render a gauge chart showing probability.
    
    Args:
        probability: Probability value (0-1)
        title: Chart title
        
    Example:
        >>> render_probability_gauge(0.75, "Phishing Risk Score")
    """
    # Determine color based on probability
    if probability < 0.3:
        color = "#10b981"  # Green
    elif probability < 0.7:
        color = "#f59e0b"  # Yellow
    else:
        color = "#ef4444"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#f1f5f9'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#f1f5f9'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
            'bar': {'color': color},
            'bgcolor': "#1e293b",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#00d4d4", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        font={'color': '#f1f5f9'},
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance_chart(importance_dict: Dict[str, float], title: str = "Feature Importance") -> None:
    """
    Render horizontal bar chart of feature importance.
    
    Args:
        importance_dict: Dictionary of {feature_name: importance_score}
        title: Chart title
        
    Example:
        >>> importance = {'degree': 0.15, 'amount': 0.12, 'frequency': 0.10}
        >>> render_feature_importance_chart(importance)
    """
    # Sort by importance
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    fig = go.Figure(go.Bar(
        y=features,
        x=scores,
        orientation='h',
        marker=dict(
            color=scores,
            colorscale='Teal',
            showscale=False
        ),
        text=[f'{score:.4f}' for score in scores],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        font={'color': '#f1f5f9'},
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix(
    true_positives: int,
    true_negatives: int,
    false_positives: int,
    false_negatives: int
) -> None:
    """
    Render confusion matrix heatmap.
    
    Args:
        true_positives: TP count
        true_negatives: TN count
        false_positives: FP count
        false_negatives: FN count
        
    Example:
        >>> render_confusion_matrix(145, 1930, 43, 74)
    """
    confusion_data = np.array([
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ])
    
    fig = go.Figure(data=go.Heatmap(
        z=confusion_data,
        x=['Predicted Normal', 'Predicted Phisher'],
        y=['Actual Normal', 'Actual Phisher'],
        text=confusion_data,
        texttemplate='%{text}',
        textfont={"size": 16, "color": "#f1f5f9"},
        colorscale='Teal',
        showscale=True
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        font={'color': '#f1f5f9'},
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_performance_metrics_table(metrics: Dict[str, float]) -> None:
    """
    Render performance metrics as a formatted table.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Example:
        >>> metrics = {'Accuracy': 0.9466, 'Precision': 0.7713, 'Recall': 0.6621}
        >>> render_performance_metrics_table(metrics)
    """
    df = pd.DataFrame([
        {"Metric": name, "Value": f"{value:.2%}"} 
        for name, value in metrics.items()
    ])
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


def render_timeline_chart(dates: List[str], values: List[float], title: str = "Activity Over Time") -> None:
    """
    Render timeline/line chart.
    
    Args:
        dates: List of date strings
        values: List of corresponding values
        title: Chart title
        
    Example:
        >>> dates = ['2024-01', '2024-02', '2024-03']
        >>> values = [10, 15, 12]
        >>> render_timeline_chart(dates, values, "Monthly Transactions")
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        line=dict(color='#00d4d4', width=3),
        marker=dict(size=8, color='#00d4d4'),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 212, 0.2)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        font={'color': '#f1f5f9'},
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_distribution_histogram(
    values: List[float], 
    bins: int = 30, 
    title: str = "Distribution"
) -> None:
    """
    Render histogram showing distribution.
    
    Args:
        values: List of values to plot
        bins: Number of histogram bins
        title: Chart title
        
    Example:
        >>> amounts = [1.2, 3.4, 2.1, 5.6, ...]
        >>> render_distribution_histogram(amounts, bins=50, title="Transaction Amounts")
    """
    fig = go.Figure(data=[go.Histogram(
        x=values,
        nbinsx=bins,
        marker=dict(
            color='#00d4d4',
            line=dict(color='#0f172a', width=1)
        )
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Frequency",
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        font={'color': '#f1f5f9'},
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_section_header(title: str, icon: str = "", description: str = "") -> None:
    """
    Render a styled section header.
    
    Args:
        title: Section title
        icon: Optional emoji icon
        description: Optional description text
        
    Example:
        >>> render_section_header("Analysis Results", "📊", "Model predictions and insights")
    """
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%); 
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                border: 1px solid rgba(0, 212, 212, 0.3);'>
        <h2 style='color: white; margin: 0;'>{icon} {title}</h2>
        {f"<p style='color: #cbd5e1; margin: 0.5rem 0 0 0;'>{description}</p>" if description else ""}
    </div>
    """, unsafe_allow_html=True)


def render_loading_spinner(message: str = "Loading...") -> None:
    """
    Display a loading spinner with message.
    
    Args:
        message: Loading message to display
        
    Example:
        >>> with st.spinner():
        ...     render_loading_spinner("Training model...")
        ...     train_model()
    """
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem;'>
        <div class='loader'></div>
        <p style='color: #cbd5e1; margin-top: 1rem;'>{message}</p>
    </div>
    """, unsafe_allow_html=True)
