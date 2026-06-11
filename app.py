import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp

st.set_page_config(page_title="EM Vector Tool", layout="wide")
st.title("⚡ EE Project: Advanced Interactive Vector Calculus Tool")
st.write("Solve and map Gradient, Divergence, and Curl equations with step-by-step solutions and 3D Surface Mesh plots.")

# Sidebar setup for user input variables
st.sidebar.header("Input Settings")

op = st.sidebar.selectbox(
    "Select Math Operator:",
    ["Gradient (∇f)", "Divergence (∇ · F)", "Curl (∇ × F)"]
)

st.sidebar.subheader("Equations (Use x, y, z)")

# Textbook input fields
if op == "Gradient (∇f)":
    user_eq = st.sidebar.text_input("f(x, y) or f(x, y, z) =", "7*x*y / sp.exp(x**2 + y**2)")
else:
    eq_x = st.sidebar.text_input("Fx =", "-y")
    eq_y = st.sidebar.text_input("Fy =", "x")
    eq_z = st.sidebar.text_input("Fz =", "0")

st.sidebar.subheader("Point Evaluation Marker")
val_x = st.sidebar.number_input("Marker X position:", value=1.0)
val_y = st.sidebar.number_input("Marker Y position:", value=1.0)
val_z = st.sidebar.number_input("Marker Z position:", value=0.0)

# Gridlines mapping slider (matches the 'Number of Gridlines' input in your photo)
gridlines = st.sidebar.slider("Number of Gridlines", 10, 50, 31)

# Initialize variables for the symbolic calculus engine
x, y, z = sp.symbols('x y z')

st.header("📖 Analytical Step-by-Step Solution")

try:
    if op == "Gradient (∇f)":
        my_function = sp.sympify(user_eq)
        
        # Calculate partial derivatives across all axes
        g_x = sp.diff(my_function, x)
        g_y = sp.diff(my_function, y)
        g_z = sp.diff(my_function, z)
        
        # Display LaTeX steps
        st.latex(rf"\text{{Given Scalar Field: }} f = {sp.latex(my_function)}")
        st.write("**Step 1: Compute partial derivatives for each unit axis:**")
        st.latex(rf"\frac{{\partial f}}{{\partial x}} = {sp.latex(g_x)}")
        st.latex(rf"\frac{{\partial f}}{{\partial y}} = {sp.latex(g_y)}")
        st.latex(rf"\frac{{\partial f}}{{\partial z}} = {sp.latex(g_z)}")
        
        st.write("**Step 2: Assemble the analytical Gradient expression:**")
        st.latex(rf"\nabla f = \left({sp.latex(g_x)}\right)\hat{{i}} + \left({sp.latex(g_y)}\right)\hat{{j}} + \left({sp.latex(g_z)}\right)\hat{{k}}")
        
        ans_x = g_x.subs({x: val_x, y: val_y, z: val_z})
        ans_y = g_y.subs({x: val_x, y: val_y, z: val_z})
        ans_z = g_z.subs({x: val_x, y: val_y, z: val_z})
        
        st.write(f"**Step 3: Evaluate vector values at coordinate point ({val_x}, {val_y}, {val_z}):**")
        st.latex(rf"\nabla f|_{{point}} = \mathbf{{\left[ {ans_x},\, {ans_y},\, {ans_z} \right]}}")

        # FIXED: Substitute z variable first before mapping the 2D grid plane coordinates
        flattened_function = my_function.subs(z, val_z)
        z_mesh_func = sp.lambdify((x, y), flattened_function, 'numpy')

    elif op == "Divergence (∇ · F)":
        f_x = sp.sympify(eq_x)
        f_y = sp.sympify(eq_y)
        f_z = sp.sympify(eq_z)
        
        d_x = sp.diff(f_x, x)
        d_y = sp.diff(f_y, y)
        d_z = sp.diff(f_z, z)
        total_div = d_x + d_y + d_z
        
        st.latex(rf"\text{{Given Vector Field F: }} \mathbf{{F}} = \left[ {sp.latex(f_x)},\, {sp.latex(f_y)},\, {sp.latex(f_z)} \right]")
        st.write("**Step 1: Compute corresponding directional partial derivatives:**")
        st.latex(rf"\frac{{\partial F_x}}{{\partial x}} = {sp.latex(d_x)}, \quad \frac{{\partial F_y}}{{\partial y}} = {sp.latex(d_y)}, \quad \frac{{\partial F_z}}{{\partial z}} = {sp.latex(d_z)}")
        st.write("**Step 2: Sum the derivatives to find total Divergence:**")
        st.latex(rf"\nabla \cdot \mathbf{{F}} = {sp.latex(total_div)}")
        
        div_ans = total_div.subs({x: val_x, y: val_y, z: val_z})
        st.write(f"**Step 3: Evaluate scalar scale magnitude at point ({val_x}, {val_y}, {val_z}):**")
        st.latex(rf"\nabla \cdot \mathbf{{F}}|_{{point}} = \mathbf{{{div_ans}}}")
        
        # FIXED: Substitute z variable first to lock execution parameters
        flattened_function = total_div.subs(z, val_z)
        z_mesh_func = sp.lambdify((x, y), flattened_function, 'numpy')

    elif op == "Curl (∇ × F)":
        f_x = sp.sympify(eq_x)
        f_y = sp.sympify(eq_y)
        f_z = sp.sympify(eq_z)
        
        c_x = sp.diff(f_z, y) - sp.diff(f_y, z)
        c_y = sp.diff(f_x, z) - sp.diff(f_z, x)
        c_z = sp.diff(f_y, x) - sp.diff(f_x, y)
        
        st.latex(rf"\text{{Given Vector Field F: }} \mathbf{{F}} = \left[ {sp.latex(f_x)},\, {sp.latex(f_y)},\, {sp.latex(f_z)} \right]")
        st.write("**Step 1: Expand the cross-product matrix determinant components:**")
        st.latex(rf"i = {sp.latex(c_x)}, \quad j = {sp.latex(c_y)}, \quad k = {sp.latex(c_z)}")
        st.write("**Step 2: Assemble the analytical Curl vector expression:**")
        st.latex(rf"\nabla \times \mathbf{{F}} = \left[ {sp.latex(c_x)},\, {sp.latex(c_y)},\, {sp.latex(c_z)} \right]")
        
        ans_x = c_x.subs({x: val_x, y: val_y, z: val_z})
        ans_y = c_y.subs({x: val_x, y: val_y, z: val_z})
        ans_z = c_z.subs({x: val_x, y: val_y, z: val_z})
        
        st.write(f"**Step 3: Evaluate rotation vectors at point ({val_x}, {val_y}, {val_z}):**")
        st.latex(rf"\nabla \times \mathbf{{F}}|_{{point}} = \mathbf{{\left[ {ans_x},\, {ans_y},\, {ans_z} \right]}}")
        
        # FIXED: Substitute z variable first to prevent matrix compilation crashes
        flattened_function = c_z.subs(z, val_z)
        z_mesh_func = sp.lambdify((x, y), flattened_function, 'numpy')

    # --- 3D SURFACE MESH GRAPH ENGINE ---
    st.header("🌐 Interactive 3D Surface Visualization")
    
    limit = 3.0
    x_range = np.linspace(-limit, limit, gridlines)
    y_range = np.linspace(-limit, limit, gridlines)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Compute numerical matrices safely
    Z = np.array(z_mesh_func(X, Y), dtype=float)
    if Z.ndim == 0: 
        Z = np.full_like(X, Z)

    # Build the 3D grid surface mesh plot trace
    my_plot = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Geyser', 
        contours=dict(
            x=dict(show=True, color="black", width=1, project=dict(x=False)),
            y=dict(show=True, color="black", width=1, project=dict(y=False))
        ),
        lighting=dict(ambient=0.7, roughness=0.1) 
    )])
    
    # Place evaluation marker dot floating inside the mesh grid ecosystem
    eval_z_val = float(z_mesh_func(val_x, val_y))
    my_plot.add_trace(go.Scatter3d(
        x=[val_x], y=[val_y], z=[eval_z_val], mode='markers',
        marker=dict(size=8, color='red', symbol='circle'), name="Eval Point"
    ))
    
    my_plot.update_layout(
        scene=dict(
            xaxis=dict(title='X Axis', backgroundcolor="white", gridcolor="lightgray"),
            yaxis=dict(title='Y Axis', backgroundcolor="white", gridcolor="lightgray"),
            zaxis=dict(title='Z Axis', backgroundcolor="white", gridcolor="lightgray"),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=10), height=700
    )
    st.plotly_chart(my_plot, use_container_width=True)

except Exception as math_parse_error:
    st.error(f"❌ Input Equation Syntax Error. Remember to use 'sp.exp()' for exponents, '*' for multiplication (e.g. 2*x) and '**' for powers (e.g. x**2). Technical Details: {math_parse_error}")
