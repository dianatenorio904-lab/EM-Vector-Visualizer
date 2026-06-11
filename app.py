import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="EM Vector Tool", layout="wide")
st.title("⚡ EE Project: Interactive 2D & 3D Vector Calculus Visualizer")
st.write("An interactive interface to plot and map custom Electromagnetics vector fields.")

# Sidebar Controls
st.sidebar.header("Input Settings")
view_mode = st.sidebar.radio("Select View Mode:", ["2D Plane (X-Y)", "3D Space (X-Y-Z)"])
op = st.sidebar.selectbox("Select Operation Display:", ["Gradient Field", "Divergence Field", "Curl Field"])

st.sidebar.subheader("Vector Field Components")
st.sidebar.caption("Use standard python expressions like: x, y, z, x**2, np.sin(x), np.cos(y)")

# Default inputs safe for both 2D and 3D
if view_mode == "2D Plane (X-Y)":
    eq_x = st.sidebar.text_input("Fx (Horizontal component) =", "-y")
    eq_y = st.sidebar.text_input("Fy (Vertical component) =", "x")
    eq_z = "0"
else:
    eq_x = st.sidebar.text_input("Fx =", "-y")
    eq_y = st.sidebar.text_input("Fy =", "x")
    eq_z = st.sidebar.text_input("Fz =", "z")

st.sidebar.subheader("Point Evaluation Marker")
val_x = st.sidebar.number_input("Marker X position:", value=1.0)
val_y = st.sidebar.number_input("Marker Y position:", value=1.0)
val_z = st.sidebar.number_input("Marker Z position:", value=1.0) if view_mode == "3D Space (X-Y-Z)" else 0.0

arrows = st.sidebar.slider("Number of Arrows", 4, 15, 8)

# Map Window Creation
st.header("🌐 Interactive Visualizer Map Window")
limit = 4

# Run plotting inside a safe container to completely block red error screens
try:
    if view_mode == "3D Space (X-Y-Z)":
        x_space = np.linspace(-limit, limit, arrows)
        y_space = np.linspace(-limit, limit, arrows)
        z_space = np.linspace(-limit, limit, arrows)
        X, Y, Z = np.meshgrid(x_space, y_space, z_space)
        
        # Safely evaluate user string expressions using local grid variables
        U = eval(eq_x, {"x": X, "y": Y, "z": Z, "np": np})
        V = eval(eq_y, {"x": X, "y": Y, "z": Z, "np": np})
        W = eval(eq_z, {"x": X, "y": Y, "z": Z, "np": np})
        
        # Handle constants (if user enters just a number like "0" or "5")
        if not isinstance(U, np.ndarray): U = np.full_like(X, float(U))
        if not isinstance(V, np.ndarray): V = np.full_like(Y, float(V))
        if not isinstance(W, np.ndarray): W = np.full_like(Z, float(W))

        x_flat, y_flat, z_flat = X.flatten(), Y.flatten(), Z.flatten()
        u_flat, v_flat, w_flat = U.flatten(), V.flatten(), W.flatten()
        
        my_plot = go.Figure(data=go.Cone(
            x=x_flat, y=y_flat, z=z_flat,
            u=u_flat, v=v_flat, w=w_flat,
            colorscale='Viridis', sizemode='scaled', sizeref=1.2,
            colorbar=dict(title="Field Intensity")
        ))
        
        my_plot.add_trace(go.Scatter3d(
            x=[val_x], y=[val_y], z=[val_z], mode='markers',
            marker=dict(size=7, color='red'), name="Eval Point"
        ))
        
        my_plot.update_layout(
            scene=dict(xaxis=dict(title='X Axis'), yaxis=dict(title='Y Axis'), zaxis=dict(title='Z Axis'), aspectmode='cube'),
            margin=dict(l=0, r=0, b=0, t=10), height=600
        )
        st.plotly_chart(my_plot, use_container_width=True)
        
    else:
        x_space = np.linspace(-limit, limit, arrows * 2)
        y_space = np.linspace(-limit, limit, arrows * 2)
        X, Y = np.meshgrid(x_space, y_space)
        Z = np.zeros_like(X)
        
        U = eval(eq_x, {"x": X, "y": Y, "z": Z, "np": np})
        V = eval(eq_y, {"x": X, "y": Y, "z": Z, "np": np})
        
        if not isinstance(U, np.ndarray): U = np.full_like(X, float(U))
        if not isinstance(V, np.ndarray): V = np.full_like(Y, float(V))
        
        uf, vf = U.flatten(), V.flatten()
        xf, yf = X.flatten(), Y.flatten()
        
        mags = np.sqrt(uf**2 + vf**2)
        max_mag = np.max(mags) if np.max(mags) > 0 else 1.0
        
        my_plot = go.Figure()
        
        # Build 2D arrows cleanly without cluttering
        for xi, yi, ui, vi, m in zip(xf, yf, uf, vf, mags):
            if m == 0: continue
            scale = 0.4 / max_mag
            my_plot.add_trace(go.Scatter(
                x=[xi, xi + ui * scale], y=[yi, yi + vi * scale],
                mode='lines', line=dict(color='teal', width=2), showlegend=False
            ))
            
        my_plot.add_trace(go.Scatter(
            x=[val_x], y=[val_y], mode='markers',
            marker=dict(size=12, color='red', symbol='circle'), name='Eval Point'
        ))
        
        my_plot.update_layout(
            xaxis=dict(title='X Axis', range=[-limit-0.5, limit+0.5]),
            yaxis=dict(title='Y Axis', range=[-limit-0.5, limit+0.5]),
            height=600, margin=dict(l=40, r=40, b=40, t=10)
        )
        st.plotly_chart(my_plot, use_container_width=True)

except Exception as user_input_error:
    st.error("⚠️ Waiting for a valid mathematical equation setup. Please check your variables (use x, y, z) and syntax (use '*' for multiplication, e.g., '2*x').")
