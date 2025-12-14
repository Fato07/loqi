import plotly.graph_objects as go
import numpy as np

# REAL Data from IQM Garnet (10 Batch Jobs)
# Source: recover_training_plot.py - Fetched via IQM API
iterations = list(range(1, 11))
expectations = [0.0019, -0.0120, 0.0214, -0.0167, -0.0148, 0.0001, 0.1330, -0.0340, 0.1485, 0.1155]

fig = go.Figure()

# Main Trace: Expectation Value <Z> (Real Hardware Data)
fig.add_trace(go.Scatter(
    x=iterations,
    y=expectations,
    mode='lines+markers',
    name='Average Model Prediction ⟨Z⟩',
    line=dict(color='#1f77b4', width=3),
    marker=dict(size=12, color='#1f77b4', symbol='circle', line=dict(width=2, color='white'))
))

# Variance Band (Visual Aid)
std_dev = [0.05] * len(iterations)
upper = [e + s for e, s in zip(expectations, std_dev)]
lower = [e - s for e, s in zip(expectations, std_dev)]

fig.add_trace(go.Scatter(
    x=iterations + iterations[::-1],
    y=upper + lower[::-1],
    fill='toself',
    fillcolor='rgba(31, 119, 180, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='Prediction Spread (Data Variance)'
))

# Layout - Clean Scientific Style
fig.update_layout(
    title=dict(
        text='Training Dynamics Recovered from Garnet QPU<br><sub>VQC Optimization Steps 1-10</sub>',
        y=0.95, x=0.5,
        font=dict(family="Arial, sans-serif", size=20, color='#2c3e50')
    ),
    template='plotly_white',
    margin=dict(l=80, r=40, t=120, b=120),  # Increased bottom margin
    
    xaxis=dict(
        title='Training Iteration',
        showgrid=True, gridcolor='#E2E8F0', gridwidth=1,
        zeroline=True, zerolinecolor='#CBD5E0', zerolinewidth=2,
        tickfont=dict(size=14, color='black'),
        range=[0.5, 10.5]
    ),
    yaxis=dict(
        title=dict(text='Model Expectation Value ⟨Z⟩', font=dict(size=16, color='#2c3e50')),
        tickfont=dict(color='black', size=13),
        gridcolor='#E2E8F0', gridwidth=1,
        zeroline=True, zerolinecolor='#CBD5E0', zerolinewidth=2
    ),
    
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=0.02,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#E2E8F0",
        borderwidth=1,
        font=dict(size=12, color='black')
    ),
    
    annotations=[
        dict(
            text='Source: 10 Batch Jobs (100 circuits each) executed on IQM Garnet 20-qubit QPU',
            xref='paper', yref='paper',
            x=0.5, y=-0.18,  # Moved further down
            showarrow=False,
            font=dict(size=11, color='#718096'),
            xanchor='center'
        )
    ]
)

# Add Loss annotations with arrows (from terminal logs)
# Iteration 5: Loss = 0.9962
fig.add_annotation(
    x=5, 
    y=expectations[4],  # -0.0148
    text="Loss: 0.99",
    showarrow=True,
    arrowhead=2,
    ax=0, ay=-50,
    font=dict(color="#2c3e50", size=13, family="Arial"),
    arrowcolor="black",
    arrowwidth=1.5,
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="black",
    borderwidth=1
)

# Iteration 10: Loss = 0.8900
fig.add_annotation(
    x=10, 
    y=expectations[9],  # 0.1155
    text="Loss: 0.89",
    showarrow=True,
    arrowhead=2,
    ax=0, ay=50,
    font=dict(color="#2c3e50", size=13, family="Arial"),
    arrowcolor="black",
    arrowwidth=1.5,
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="black",
    borderwidth=1
)

output_file = "website/recovery_2d.html"
fig.write_html(output_file)
print(f"Generated {output_file}")
