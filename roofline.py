import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define the hardware specifications
peak_memory_bandwidth = 600  # GB/s
peak_flops = 35000  # GFLOPS

# Define the kernel data
kernels = {
    'Naive': {
        'DRAM_Bytes': 32746880,
        'Execution_Time': 10.78,  # ms
        'FLOPS': 2148532224,
        'Arithmetic_Intensity': 65.61,
        'Performance': 199.37
    },
    'GMEM Coalescing': {
        'DRAM_Bytes': 6596352,
        'Execution_Time': 1.40,
        'FLOPS': 2148532224,
        'Arithmetic_Intensity': 325.72,
        'Performance': 1537.97
    },
    'SMEM Cache-Blocking': {
        'DRAM_Bytes': 513920,
        'Execution_Time': 1.01,
        'FLOPS': 2148532224,
        'Arithmetic_Intensity': 4180.67,
        'Performance': 2134.87
    },
    '1D Blocktiling': {
        'DRAM_Bytes': 22970112,
        'Execution_Time': 0.43,
        'FLOPS': 2147483648,
        'Arithmetic_Intensity': 93.49,
        'Performance': 4987.28
    },
    '2D Blocktiling': {
        'DRAM_Bytes': 19443072,
        'Execution_Time': 0.29,
        'FLOPS': 2147483648,
        'Arithmetic_Intensity': 110.45,
        'Performance': 7284.15
    },
    'Vectorizd Mem Access': {
        'DRAM_Bytes': 17231616,
        'Execution_Time': 0.26,
        'FLOPS': 2147483648,
        'Arithmetic_Intensity': 124.62,
        'Performance': 8172.05
    },
    'Warptiling': {
        'DRAM_Bytes': 15022464,
        'Execution_Time': 0.22,
        'FLOPS': 2147483648,
        'Arithmetic_Intensity': 142.95,
        'Performance': 9591.09
    },
    'cuBLAS (ampere_sgemm_1)': {
        'DRAM_Bytes': 35172992,
        'Execution_Time': 0.15,
        'FLOPS': 2194669568,
        'Arithmetic_Intensity': 62.40,
        'Performance': 14695.40
    }
}

# Create a DataFrame for easier plotting
df = pd.DataFrame.from_dict(kernels, orient='index')

# Calculate the ridge point (where memory bound meets compute bound)
ridge_point = peak_flops / peak_memory_bandwidth

# Create interactive plot with Plotly
fig = make_subplots()

# Set up the log scales for x and y with standardized tick marks
fig.update_xaxes(
    type="log", 
    range=[1, 4],  # log10(10) to log10(10000)
    tickvals=[10, 100, 1000, 10000],
    ticktext=["10", "100", "1,000", "10,000"],
    tickmode="array"
)
fig.update_yaxes(
    type="log", 
    range=[2, 5],  # log10(100) to log10(100000)
    tickvals=[100, 1000, 10000, 100000],
    ticktext=["100", "1,000", "10,000", "100,000"],
    tickmode="array"
)
# Add the memory-bound roofline (diagonal line)
x_mem = np.logspace(1, np.log10(ridge_point), 100)
y_mem = x_mem * peak_memory_bandwidth

fig.add_trace(
    go.Scatter(
        x=x_mem,
        y=y_mem,
        mode='lines',
        line=dict(color='red', width=3),
        name='Memory Bound'
    )
)

# Add the compute-bound roofline (horizontal line)
x_compute = np.logspace(np.log10(ridge_point), 4, 100)
y_compute = np.ones_like(x_compute) * peak_flops

fig.add_trace(
    go.Scatter(
        x=x_compute,
        y=y_compute,
        mode='lines',
        line=dict(color='red', width=3),
        name='Compute Bound'
    )
)

# Add ridge point marker
fig.add_trace(
    go.Scatter(
        x=[ridge_point],
        y=[peak_flops],
        mode='markers',
        marker=dict(color='red', size=12),
        name=f'Ridge Point ({ridge_point:.2f})',
        hoverinfo='name'
    )
)

# Add each kernel as a point
for kernel, data in kernels.items():
    fig.add_trace(
        go.Scatter(
            x=[data['Arithmetic_Intensity']],
            y=[data['Performance']],
            mode='markers+text',
            marker=dict(size=12),
            text=kernel,
            textposition="top center",
            name=kernel,
            hovertemplate=
            '<b>%{kernel}</b><br><br>' +
            'Arithmetic Intensity: %{x:.2f} FLOPS/byte<br>' +
            'Performance: %{y:.2f} GFLOPS/s<br>' +
            '<extra></extra>',
        )
    )

# Add theoretical peak line
fig.add_shape(
    type="line",
    x0=10,
    y0=peak_flops,
    x1=10000,
    y1=peak_flops,
    line=dict(
        color="gray",
        width=2,
        dash="dash",
    ),
    opacity=0.5
)

# Configure the layout
fig.update_layout(
    title='GPU Kernel Performance Roofline Model<br>Peak Compute: 35 TFLOPS, Peak Memory Bandwidth: 600 GB/s',
    xaxis_title='Arithmetic Intensity (FLOPS/byte)',
    yaxis_title='Performance (GFLOPS/s)',
    height=800,
    width=1200,
    showlegend=True,
    legend=dict(
        x=1.05,
        y=1,
        xanchor='left',
    ),
    plot_bgcolor='rgba(240, 240, 240, 0.5)'
)

# Add annotation for theoretical peak
fig.add_annotation(
    x=10,
    y=peak_flops * 1.05,
    text=f'Theoretical Peak: {peak_flops} GFLOPS',
    showarrow=False,
    font=dict(color="gray"),
    xanchor="left"
)

# Save as interactive HTML file
fig.write_html('interactive_roofline_plot.html')

# Show plot in notebook or browser
fig.show()
