import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp

st.set_page_config(page_title="EM Vector Tool", layout="wide")
st.title("Electromagnetics: Vector Calculus Visualizer")
st.write("Solve and map Gradient, Divergence, and Curl equations with step-by-step solutions in 2D or 3D.")

# Sidebar setup for user input variables
st.sidebar.header("Input Settings")

# 2D or 3D viewport selector
view_mode = st.sidebar.radio("Select View Mode:", ["2D Plane (X-Y)", "3D Space (X-Y-Z)"])

op = st.sidebar.selectbox(
    "Select Math Operator:",
    ["Gradient (∇f)", "Divergence (∇ · F)", "Curl (∇ × F)"]
)

st.sidebar.subheader("Equations (Use x, y, z)")

# Dynamic string input rendering based on selected parameters
if op == "Gradient (∇f)":
    if view_mode == "2D Plane (X-Y)":
        user_eq = st.sidebar.text_input("f(x, y) =", "x**2 - y**2")
    else:
        user_eq = st.sidebar.text_input("f(x, y, z) =", "x**2 + y**2 - z")
else:
    if view_mode == "2D Plane (X-Y)":
        eq_x = st.sidebar.text_input("Fx =", "-y")
        eq_y = st.sidebar.text_input("Fy =", "x")
    else:
        eq_x = st.sidebar.text_input("Fx =", "-y")
        eq_y = st.sidebar.text_input("Fy =", "x")
        eq_z = st.sidebar.text_input("Fz =", "z")

st.sidebar.subheader("Point Evaluation Marker")
val_x = st.sidebar.number_input("Marker X position:", value=1.0)
val_y = st.sidebar.number_input("Marker Y position:", value=1.0)
if view_mode == "3D Space (X-Y-Z)":
    val_z = st.sidebar.number_input("Marker Z position:", value=1.0)
else:
    val_z = 0.0  # Constant baseline altitude for 2D cross sections

arrows = st.sidebar.slider("Number of Arrows", 4, 15, 8)

# Initialize variables for the symbolic calculus engine
x, y, z = sp.symbols('x y z')

st.header("📖 Analytical Step-by-Step Solution")

# --- SYMPY SYMBOLIC MATH LAYER ---
try:
    if op == "Gradient (∇f)":
        my_function = sp.sympify(user_eq)
        
        # Calculate partial derivatives across all axes
        g_x = sp.diff(my_function, x)
        g_y = sp.diff(my_function, y)
        g_z = sp.diff(my_function, z) if view_mode == "3D Space (X-Y-Z)" else sp.sympify(0)
        
        # Display LaTeX steps
        st.latex(rf"\text{{Given Scalar Field: }} f = {sp.latex(my_function)}")
        st.write("**Step 1: Compute partial derivatives for each unit axis:**")
        st.latex(rf"\frac{{\partial f}}{{\partial x}} = {sp.latex(g_x)}")
        st.latex(rf"\frac{{\partial f}}{{\partial y}} = {sp.latex(g_y)}")
        
        if view_mode == "3D Space (X-Y-Z)":
            st.latex(rf"\frac{{\partial f}}{{\partial z}} = {sp.latex(g_z)}")
            st.write("**Step 2: Assemble the analytical Gradient expression:**")
            st.latex(rf"\nabla f = \left({sp.latex(g_x)}\right)\hat{{i}} + \left({sp.latex(g_y)}\right)\hat{{j}} + \left({sp.latex(g_z)}\right)\hat{{k}}")
            ans_x = g_x.subs({x: val_x, y: val_y, z: val_z})
            ans_y = g_y.subs({x: val_x, y: val_y, z: val_z})
            ans_z = g_z.subs({x: val_x, y: val_y, z: val_z})
            st.write(f"**Step 3: Evaluate vector values at coordinate point ({val_x}, {val_y}, {val_z}):**")
            st.latex(rf"\nabla f|_{{point}} = \mathbf{{\left[ {ans_x},\, {ans_y},\, {ans_z} \right]}}")
        else:
            st.write("**Step 2: Assemble the analytical Gradient expression:**")
            st.latex(rf"\nabla f = \left({sp.latex(g_x)}\right)\hat{{i}} + \left({sp.latex(g_y)}\right)\hat{{j}}")
            ans_x = g_x.subs({x: val_x, y: val_y, z: val_z})
            ans_y = g_y.subs({x: val_x, y: val_y, z: val_z})
            st.write(f"**Step 3: Evaluate vector values at coordinate point ({val_x}, {val_y}):**")
            st.latex(rf"\nabla f|_{{point}} = \mathbf{{\left[ {ans_x},\, {ans_y} \right]}}")

        # Compile symbolic formulas into stable functions accepting (x, y, z) inputs
        u_calc = sp.lambdify((x, y, z), g_x, 'numpy')
        v_calc = sp.lambdify((x, y, z), g_y, 'numpy')
        w_calc = sp.lambdify((x, y, z), g_z, 'numpy')

    elif op == "Divergence (∇ · F)":
        f_x = sp.sympify(eq_x)
        f_y = sp.sympify(eq_y)
        f_z = sp.sympify(eq_z) if view_mode == "3D Space (X-Y-Z)" else sp.sympify(0)
        
        d_x = sp.diff(f_x, x)
        d_y = sp.diff(f_y, y)
        d_z = sp.diff(f_z, z)
        total_div = d_x + d_y + d_z
        
        if view_mode == "3D Space (X-Y-Z)":
            st.latex(rf"\text{{Given Vector Field F: }} \mathbf{{F}} = \left[ {sp.latex(f_x)},\, {sp.latex(f_y)},\, {sp.latex(f_z)} \right]")
            st.write("**Step 1: Compute corresponding directional partial derivatives:**")
            st.latex(rf"\frac{{\partial F_x}}{{\partial x}} = {sp.latex(d_x)}, \quad \frac{{\partial F_y}}{{\partial y}} = {sp.latex(d_y)}, \quad \frac{{\partial F_z}}{{\partial z}} = {sp.latex(d_z)}")
            st.write("**Step 2: Sum the derivatives to find total Divergence:**")
            st.latex(rf"\nabla \cdot \mathbf{{F}} = {sp.latex(total_div)}")
            div_ans = total_div.subs({x: val_x, y: val_y, z: val_z})
            st.write(f"**Step 3: Evaluate scalar scale magnitude at point ({val_x}, {val_y}, {val_z}):**")
        else:
            st.latex(rf"\text{{Given 2D Vector Field F: }} \mathbf{{F}} = \left[ {sp.latex(f_x)},\, {sp.latex(f_y)} \right]")
            st.write("**Step 1: Compute corresponding directional partial derivatives:**")
            st.latex(rf"\frac{{\partial F_x}}{{\partial x}} = {sp.latex(d_x)}, \quad \frac{{\partial F_y}}{{\partial y}} = {sp.latex(d_y)}")
            st.write("**Step 2: Sum the derivatives to find 2D Divergence:**")
            st.latex(rf"\nabla \cdot \mathbf{{F}} = {sp.latex(d_x + d_y)}")
            div_ans = (d_x + d_y).subs({x: val_x, y: val_y, z: val_z})
            st.write(f"**Step 3: Evaluate scalar scale magnitude at point ({val_x}, {val_y}):**")
            
        st.latex(rf"\nabla \cdot \mathbf{{F}}|_{{point}} = \mathbf{{{div_ans}}}")
        u_calc = sp.lambdify((x, y, z), f_x, 'numpy')
        v_calc = sp.lambdify((x, y, z), f_y, 'numpy')
        w_calc = sp.lambdify((x, y, z), f_z, 'numpy')

    elif op == "Curl (∇ × F)":
        f_x = sp.sympify(eq_x)
        f_y = sp.sympify(eq_y)
        
        if view_mode == "3D Space (X-Y-Z)":
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
            u_calc = sp.lambdify((x, y, z), c_x, 'numpy')
            v_calc = sp.lambdify((x, y, z), c_y, 'numpy')
            w_calc = sp.lambdify((x, y, z), c_z, 'numpy')
        else:
            # 2D field curl yields rotation purely orthogonal to the layout surface plane (z component)
            c_z = sp.diff(f_y, x) - sp.diff(f_x, y)
            st.latex(rf"\text{{Given 2D Vector Field F: }} \mathbf{{F}} = \left[ {sp.latex(f_x)},\, {sp.latex(f_y)} \right]")
            st.write("**Step 1: Calculate the out-of-plane planar rotation component (Z-axis curl):**")
            st.latex(rf"\text{{Curl}}_z = \frac{{\partial F_y}}{{\partial x}} - \frac{{\partial F_x}}{{\partial y}} = {sp.latex(c_z)}")
            ans_z = c_z.subs({x: val_x, y: val_y, z: val_z})
            st.write(f"**Step 2: Solve magnitude of vector rotation at point ({val_x}, {val_y}):**")
            st.latex(rf"\text{{Curl}}_z|_{{point}} = \mathbf{{{ans_z}}}")
            u_calc = sp.lambdify((x, y, z), f_x, 'numpy') # Display background vector paths for orientation context
            v_calc = sp.lambdify((x, y, z), f_y, 'numpy')
            w_calc = sp.lambdify((x, y, z), sp.sympify(0), 'numpy')

    # --- GRAPH RENDERING LAYOUT ENGINE ---
    st.header("Interactive Resultant Visualization")
    limit = 4
    
    if view_mode == "3D Space (X-Y-Z)":
        x_space = np.linspace(-limit, limit, arrows)
        y_space = np.linspace(-limit, limit, arrows)
        z_space = np.linspace(-limit, limit, arrows)
        X, Y, Z = np.meshgrid(x_space, y_space, z_space)
    else:
        # 2D cross sections utilize double mesh lines mapped on a flat Z=0 plane grid
        x_space = np.linspace(-limit, limit, arrows * 2)
        y_space = np.linspace(-limit, limit, arrows * 2)
        X, Y = np.meshgrid(x_space, y_space)
        Z = np.zeros_like(X)

    # Convert symbolic representations to numeric matrix arrays safely
    U = np.array(u_calc(X, Y, Z), dtype=float)
    V = np.array(v_calc(X, Y, Z), dtype=float)
    W = np.array(w_calc(X, Y, Z), dtype=float)
    
    # Broadcast values if inputs returned basic numeric constants instead of arrays
    if U.ndim == 0: U = np.full_like(X, U)
    if V.ndim == 0: V = np.full_like(Y, V)
    if W.ndim == 0: W = np.full_like(Z, W)

    # Flatten coordinates for interactive Plotly layout compatibility
    x_flat, y_flat, z_flat = X.flatten(), Y.flatten(), Z.flatten()
    u_flat, v_flat, w_flat = U.flatten(), V.flatten(), W.flatten()
