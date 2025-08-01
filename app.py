import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Set Streamlit page config first thing
st.set_page_config(
    page_title="EV Adoption Forecaster",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define forecast horizon globally
forecast_horizon = 36

# === Load model with caching ===
@st.cache_resource
def load_model():
    return joblib.load('forecasting_ev_model.pkl')

model = load_model()

# === Enhanced Dark Professional Styling ===
st.markdown("""
<style>
/* Main app styling - Pitch Dark */
.stApp {
    background-color: #0E1117;
    color: #f0f2f6;
    margin:0;
    padding:0;
}
/* Custom container styling */
.main-container {
    margin: 0rem 0 !important;
}
/* Header styling */
.app-header {
    text-align: center;
}
.app-title {
    font-size: 4.5rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #7D50F3, #00C2FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-subtitle {
    font-size: 1.2rem;
    color: #9aa8b9;
    font-weight: 400;
    margin-bottom: 1rem;
}
.app-description {
    font-size: 1rem;
    color: #b0bac5;
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.6;
}
/* Section headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #ffffff;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #7D50F3;
}
/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d111b 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(125, 80, 243, 0.3);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: #00C2FF;
}
.metric-label {
    font-size: 0.9rem;
    opacity: 0.8;
    color: #9aa8b9;
}
/* Custom table styling for multi-county dashboard */
.stTable > div {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d111b 100%);
    border: 1px solid rgba(125, 80, 243, 0.3);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
}
.stTable th {
    background: #1a1f2e;
    color: #00C2FF;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 1rem;
    text-align: center;
    border-bottom: 1px solid rgba(125, 80, 243, 0.2);
}
.stTable td {
    color: #b0bac5;
    font-size: 1rem;
    padding: 1rem;
    text-align: center;
    border-bottom: 1px solid rgba(125, 80, 243, 0.2);
}
.stTable tr:hover {
    background: rgba(125, 80, 243, 0.1);
}
.stTable .rank {
    color: #7D50F3;
    font-weight: 700;
}
.stTable .growth-positive {
    color: #51E3A4;
}
.stTable .growth-negative {
    color: #FF6B6B;
}
/* Leaderboard styling */
.leaderboard {
    margin: 1rem 0;
    padding: 1rem;
    background: linear-gradient(135deg, #1a1f2e 0%, #0d111b 100%);
    border: 1px solid rgba(125, 80, 243, 0.3);
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
}
.leaderboard h3 {
    color: #00C2FF;
    margin-bottom: 0.5rem;
}
.leaderboard p {
    color: #9aa8b9;
    margin: 0.25rem 0;
}
/* Success message styling */
.success-message {
    background: rgba(29, 78, 137, 0.3);
    color: #9dc6ff;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    font-weight: 500;
    border-left: 4px solid #7D50F3;
}
/* Input styling */
.stSelectbox > div > div {
    background: #1a1f2e !important;
    border: 1px solid #2d3746 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
.stMultiSelect > div > div {
    background: #1a1f2e !important;
    border: 1px solid #2d3746 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
/* Image container */
.image-container {
    border-radius: 12px;
    overflow: hidden;
    margin: 2rem 0;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #7D50F3, transparent);
    margin: 3rem 0;
    border: none;
}
/* Footer */
.footer {
    text-align: center;
    color: #6b7c93;
    font-size: 1.0rem;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
/* Plot styling */
div.stPlotlyChart {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
/* Hide streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="app-header">
<div class="app-title">EV Adoption Forecaster</div>
<div class="app-subtitle">Washington State County Analysis</div>
<div class="app-description">
Analyze and predict electric vehicle adoption trends across Washington State counties
using advanced machine learning models. Get insights into future EV growth patterns
and compare adoption rates between different regions.
</div>
</div>
""", unsafe_allow_html=True)

# Image with custom styling
st.markdown('<div class="image-container">', unsafe_allow_html=True)
st.image("./assets/ev-car-factory.jpg", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# === Load data ===
@st.cache_data
def load_data():
    df = pd.read_csv("./data/preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# === Cached function to get combined historical and forecast data ===
@st.cache_data
def get_combined_data(county):
    county_df = df[df['County'] == county].sort_values("Date")
    if county not in df['County'].unique():
        return None
    
    county_code = county_df['county_encoded'].iloc[0]
    historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()
    future_rows = []

    # Check if there are enough historical data points
    if len(historical_ev) < 3:
        return None

    for i in range(1, forecast_horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1
        lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        recent_cumulative = cumulative_ev[-6:]
        ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

        new_row = {
            'months_since_start': months_since_start,
            'county_encoded': county_code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }

        pred = model.predict(pd.DataFrame([new_row]))[0]
        future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
        historical_ev.append(pred)
        if len(historical_ev) > 6:
            historical_ev.pop(0)
        cumulative_ev.append(cumulative_ev[-1] + pred)
        if len(cumulative_ev) > 6:
            cumulative_ev.pop(0)

    historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
    historical_cum['Source'] = 'Historical'
    historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()
    forecast_df = pd.DataFrame(future_rows)
    forecast_df['Source'] = 'Forecast'
    forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

    combined = pd.concat([
        historical_cum[['Date', 'Cumulative EV', 'Source']],
        forecast_df[['Date', 'Cumulative EV', 'Source']]
    ], ignore_index=True)
    return combined

# === County Selection Section ===
st.markdown('<div class="section-header">County Selection</div>', unsafe_allow_html=True)
county_list = sorted(df['County'].dropna().unique().tolist())
county = st.selectbox(
    "Choose a county to analyze",
    county_list,
    help="Select a Washington State county to view EV adoption forecasts"
)
if st.button("Generate County Forecast"):
    combined = get_combined_data(county)
    if combined is None:
        st.error(f"County '{county}' not found in dataset or has insufficient historical data.")
        st.stop()

    # === Results Section ===
    st.markdown('<div class="section-header">Forecast Results</div>', unsafe_allow_html=True)

    # Display key metrics (original card layout)
    historical_total = combined[combined['Source'] == 'Historical']['Cumulative EV'].iloc[-1]
    forecasted_total = combined[combined['Source'] == 'Forecast']['Cumulative EV'].iloc[-1]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-value">{historical_total:.1f}</div>
        <div class="metric-label">Current EVs</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-value">{forecasted_total:.1f}</div>
        <div class="metric-label">Projected EVs (3 Years)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        if historical_total > 0:
            forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
            trend_symbol = "↗" if forecast_growth_pct > 0 else "↘"
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-value">{trend_symbol} {forecast_growth_pct:.1f}%</div>
            <div class="metric-label">Growth Rate</div>
            </div>
            """, unsafe_allow_html=True)

    # === Enhanced Plot ===
    st.markdown(f'<div class="section-header">Cumulative EV Trend - {county} County</div>', unsafe_allow_html=True)

    # Set matplotlib style for dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    # Plot with minimal and sleek trend lines
    colors = ['#00C2FF', '#7D50F3']  # Subtle, professional colors
    for i, (label, data) in enumerate(combined.groupby('Source')):
        ax.plot(data['Date'], data['Cumulative EV'],
                label=label,
                linestyle='-' if i == 0 else '--',
                linewidth=2,
                color=colors[i],
                alpha=0.9,
                marker=None)  # Removed markers for minimalism

    # Professional plot styling
    ax.set_title(f"EV Adoption Forecast: {county} County",
                fontsize=18,
                fontweight='bold',
                color='white',
                pad=20)
    ax.set_xlabel("Timeline", fontsize=14, color='#b0bac5', fontweight='500')
    ax.set_ylabel("Cumulative EV Count", fontsize=14, color='#b0bac5', fontweight='500')

    # Grid and styling
    ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5, color='#2d3746')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2d3746')
    ax.spines['bottom'].set_color('#2d3746')

    # Legend styling
    legend = ax.legend(frameon=True, framealpha=0.1, fancybox=True, shadow=False,
                    loc='upper left', fontsize=12, facecolor='#0E1117')
    legend.get_frame().set_edgecolor('#2d3746')

    # Tick styling
    ax.tick_params(colors='#9aa8b9', labelsize=11)
    plt.tight_layout()
    st.pyplot(fig)

    # Growth summary
    if historical_total > 0:
        forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
        trend = "growth" if forecast_growth_pct > 0 else "decline"
        st.markdown(f"""
        <div class="success-message">
        <strong>Forecast Summary:</strong> EV adoption in {county} County is projected to show
        a {trend} of {abs(forecast_growth_pct):.1f}% over the next 3 years,
        reaching approximately {forecasted_total:.1f} total electric vehicles.
        </div>
        """, unsafe_allow_html=True)

# === County Comparison Section ===
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Multi-County Comparison</div>', unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom: 1.5rem; color: #9aa8b9; font-size: 0.95rem;">
Compare EV adoption trends across multiple counties to identify regional patterns and opportunities.
</div>
""", unsafe_allow_html=True)

multi_counties = st.multiselect(
    "Select counties for comparison (up to 10)",
    county_list,
    max_selections=10,
    help="Choose multiple counties to compare their EV adoption trends"
)
if st.button("Generate Multi-County Comparison"):
    if len(multi_counties) < 2:
        st.error("Please select minimum two county to generate the comparison.")
    else:
        comparison_data = []
        for cty in multi_counties:
            combined_cty = get_combined_data(cty)
            if combined_cty is not None:
                combined_cty['County'] = cty
                comparison_data.append(combined_cty)

        if comparison_data:
            comp_df = pd.concat(comparison_data, ignore_index=True)

            # Enhanced comparison plot
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(16, 9))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')

            comparison_colors = ['#00C2FF', '#7D50F3', '#FF6B6B', '#51E3A4']
            line_styles = ['-', '--', '-.']
            for i, (cty, group) in enumerate(comp_df.groupby('County')):
                ax.plot(group['Date'], group['Cumulative EV'],
                        label=cty,
                        linestyle=line_styles[i % len(line_styles)],
                        linewidth=2,
                        color=comparison_colors[i % len(comparison_colors)],
                        alpha=0.9,
                        marker=None) 

            ax.set_title("EV Adoption Comparison: Multi-County Analysis",
                        fontsize=20,
                        fontweight='bold',
                        color='white',
                        pad=25)
            ax.set_xlabel("Timeline", fontsize=14, color='#b0bac5', fontweight='500')
            ax.set_ylabel("Cumulative EV Count", fontsize=14, color='#b0bac5', fontweight='500')
            ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5, color='#2d3746')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#2d3746')
            ax.spines['bottom'].set_color('#2d3746')
            legend = ax.legend(title="County", frameon=True, framealpha=0.1, fancybox=True, shadow=False,
                            loc='upper left', fontsize=12, facecolor='#0E1117')
            legend.get_frame().set_edgecolor('#2d3746')
            plt.setp(legend.get_title(), color='white')
            ax.tick_params(colors='#9aa8b9', labelsize=11)
            plt.tight_layout()
            st.pyplot(fig)

            # Comparative Dashboard
            st.markdown('<div class="section-header">Comparative Dashboard</div>', unsafe_allow_html=True)
            metrics = {}
            for cty in multi_counties:
                cty_df = comp_df[comp_df['County'] == cty].reset_index(drop=True)
                # Check if there’s enough data to avoid IndexError
                if len(cty_df) > forecast_horizon:
                    historical_total = cty_df['Cumulative EV'].iloc[len(cty_df) - forecast_horizon - 1]
                    forecasted_total = cty_df['Cumulative EV'].iloc[-1]
                    if historical_total > 0:
                        growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
                        metrics[cty] = {
                            'current_evs': historical_total,
                            'projected_evs': forecasted_total,
                            'growth_rate': growth_pct
                        }
                else:
                    st.warning(f"Insufficient data for {cty} to calculate metrics.")

            # Display as a table using st.table, rename index to Rank and start from 1
            if metrics:
                table_data = []
                sorted_metrics = dict(sorted(metrics.items(), key=lambda x: x[1]['projected_evs'], reverse=True))
                for rank, (cty, data) in enumerate(sorted_metrics.items(), 1):
                    trend_symbol = "↗" if data['growth_rate'] > 0 else "↘"
                    growth_class = "growth-positive" if data['growth_rate'] > 0 else "growth-negative"
                    trend_desc = "Strong growth trend" if data['growth_rate'] > 10 else "Moderate growth" if data['growth_rate'] > 0 else "Declining or stable"
                    top_diff = data['projected_evs'] - max(metrics.values(), key=lambda x: x['projected_evs'])['projected_evs']
                    diff_text = f"({abs(top_diff):.1f} behind leader)" if top_diff < 0 else "(Leader)"
                    table_data.append({
                        'County': cty,
                        'Current EVs': f"{data['current_evs']:.1f}",
                        'Projected EVs (3 Years)': f"{data['projected_evs']:.1f}",
                        'Growth Rate': f"{trend_symbol} {data['growth_rate']:.1f}%",
                        'Trend Description': f"{trend_desc} {diff_text}"
                    })

                df_table = pd.DataFrame(table_data).reset_index(drop=True)  # Remove separate Rank column
                df_table.index = df_table.index + 1  # Start index from 1
                df_table.index.name = 'Rank'  # Rename index to Rank
                st.table(df_table.style.set_properties(**{
                    'background-color': 'transparent',
                    'border': '1px solid rgba(125, 80, 243, 0.2)',
                    'text-align': 'center',
                    'color': '#b0bac5'
                }).set_table_styles([
                    {'selector': 'th', 'props': [
                        ('background-color', '#1a1f2e'),
                        ('color', '#00C2FF'),
                        ('font-weight', '600'),
                        ('font-size', '1.1rem'),
                        ('padding', '1rem'),
                        ('border-bottom', '1px solid rgba(125, 80, 243, 0.2)')
                    ]},
                    {'selector': 'td', 'props': [
                        ('padding', '1rem'),
                        ('font-size', '1rem')
                    ]},
                    {'selector': '.rank', 'props': [
                        ('color', '#7D50F3'),
                        ('font-weight', '700')
                    ]},
                    {'selector': '.growth-positive', 'props': [('color', '#51E3A4')]},
                    {'selector': '.growth-negative', 'props': [('color', '#FF6B6B')]}
                ]).set_table_attributes('class="stTable"'))

                # Leaderboard Highlights (Enhanced and Simple with more details)
                top_projected = max(metrics.values(), key=lambda x: x['projected_evs'])
                top_growth = max(metrics.values(), key=lambda x: x['growth_rate'])
                leaderboard_html = """
                <div class="leaderboard">
                    <h3>Key Insights</h3>
                    <p><strong>Top EV Leader:</strong> {} leads with {:.0f} EVs projected in 3 years, driven by strong historical adoption rates.</p>
                    <p><strong>Fastest Growth:</strong> {} shows a growth rate of {:.0f}%, indicating rapid EV adoption momentum.</p>
                    <p><strong>Average Projection:</strong> The average projected EV count across all counties is {:.0f}, reflecting regional trends.</p>
                </div>
                """.format(
                    max(metrics, key=lambda x: metrics[x]['projected_evs']),
                    top_projected['projected_evs'],
                    max(metrics, key=lambda x: metrics[x]['growth_rate']),
                    top_growth['growth_rate'],
                    sum(m['projected_evs'] for m in metrics.values()) / len(metrics)
                )
                st.markdown(leaderboard_html, unsafe_allow_html=True)

                # Comparative Analysis (Enhanced and Simple)
                min_projected = min(metrics.values(), key=lambda x: x['projected_evs'])
                max_growth = max(metrics.values(), key=lambda x: x['growth_rate'])
                min_growth = min(metrics.values(), key=lambda x: x['growth_rate'])
                comparison_text = f"""
                <div class="success-message">
                <strong>Quick Comparison:</strong><br>
                - {max(metrics, key=lambda x: metrics[x]['projected_evs'])} is ahead with {top_projected['projected_evs']:.0f} EVs, a gap of {abs(min_projected['projected_evs'] - top_projected['projected_evs']):.0f} from the lowest.<br>
                - {max(metrics, key=lambda x: metrics[x]['growth_rate'])} grows fastest at {max_growth['growth_rate']:.0f}%, compared to {min_growth['growth_rate']:.0f}% for the slowest.
                </div>
                """
                st.markdown(comparison_text, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
<strong>EV Adoption Forecaster</strong> | Prepared for AICTE Internship Cycle 2 by S4F<br>
Made by Akash with 💌
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Close main container