import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp

st.set_page_config(page_title="EM Vector Tool", layout="wide")
st.title("Electromagnetics: Interactive Vector Calculus Visualizer")
st.write("Solve and map Gradient, Divergence, and Curl formulas using interactive 2D or 3D viewports.")

# Sidebar setup for user input variables
st.sidebar.header("Input Settings")

# 2D or 3D selection block requested by professor
view_mode = st.sidebar.radio("Select View Mode:", ["2D Plane (X-Y)", "3D Space (X-Y-Z)"])

op = st.sidebar.selectbox(
    "Select Operation:",
    ["Gradient (∇f)", "Divergence (∇ · F)", "Curl (∇ × F)"]
)

st.sidebar.subheader("Equations")
if op == "Gradient":
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
        eq_z = st.sidebar.text_input("Fz =", "0")

st.sidebar.subheader("Point Evaluation")
val_x = st.sidebar.number_input("x position:", value=1.0)
val_y = st.sidebar.number_input("y position:", value=1.0)
if view_mode == "3D Space (X-Y-Z)":
    val_z = st.sidebar.number_input("z position:", value=1.0)

arrows = st.sidebar.slider("Number of Arrows", 4, 15, 8)

# Setup variables for calculus equations
x, y, z = sp.symbols('x y z')

st.header("Step-by-Step Solution Breakdown")

try:
    if op == "Gradient":
        my_function = sp.sympify(user_eq)
        
        # calculate partial derivatives
        g_x = sp.diff(my_function, x)
        g_y = sp.diff(my_function, y)
        
        st.latex(rf"\text{{Given function: }} f = {sp.latex(my_function)}")
        st.write("**Step 1: Find partial derivatives for each axis**")
        st.latex(rf"\frac{{\partial f}}{{\partial x}} = {sp.latex(g_x)}")
        st.latex(rf"\frac{{\partial f}}{{\partial y}} = {sp.latex(g_y)}")
        
        if view_mode == "3D Space (X-Y-Z)":
            g_z = sp.diff(my_function, z)
            st.latex(rf"\frac{{\partial f}}{{\partial z}} = {sp.latex(g_z)}")
            st.write("**Step 2: Write final gradient expression**")
            st.latex(rf"\nabla f = ({sp.latex(g_x)})\hat{{i}} + ({sp.latex(g_y)})\hat{{j}} + ({sp.latex(g_z)})\hat{{k}}")
            ans_x = g_x.subs({x: val_x, y: val_y, z: val_z})
            ans_y = g_y.subs({x: val_x, y: val_y, z: val_z})
            ans_z = g_z.subs({x: val_x, y: val_y, z: val_z})
            st.write(f"**Step 3: Plug in coordinates ({val_x}, {val_y}, {val_z})**")
            st.latex(rf"\nabla f|_{{point}} = \mathbf{{\left[ {ans_x},\, {ans_y},\, {ans_z} \right]}}")
            u_calc = sp.lambdify((x, y, z), g_x, 'numpy')
            v_calc = sp.lambdify((x, y, z), g_y, 'numpy')
            w_calc = sp.lambdify((x, y, z), g_z, 'numpy')
        else:
            st.write("**Step 2: Write final gradient expression**")
            st.latex(rf"\nabla f = ({sp.latex(g_x)})\hat{{i}} + ({sp.latex(g_y)})\hat{{j}}")
            ans_x = g_x.subs({x: val_x, y: val_y})
            ans_y = g_y.subs({x: val_x, y: val_y})
            st.write(f"**Step 3: Plug in coordinates ({val_x}, {val_y})**")
            st.latex(rf"\nabla f|_{{point}} = \mathbf{{\left[ {ans_x},\, {ans_y} \right]}}")
            u_calc = sp.lambdify((x, y), g_x, 'numpy')
            v_calc = sp.lambdify((x, y), g_y, 'numpy')

    elif op == "Divergence":
        f_x = sp.sympify(eq_x)
        f_y = sp.sympify(eq_y)
        
        d_x = sp.diff(f_x, x)
        d_y = sp.diff(f_y, y)
        
        if view_mode == "3D Space (X-Y-Z)":
            f_z = sp.sympify(eq_z)
            d_z = sp.diff(f_z, z)
            total_div = d_x + d_y + d_z
            st.latex(rf"\text{{Vector Field: }} \mathbf{{F}} = \left[ {sp.latex(f_x)},\, {sp.latex(f_y)},\, {sp.latex(f_z)} \right]")
            st.write("**Step 1: Differentiate each component by its axis**")
            st.latex(rf"\frac{{\partial F_x}}{{\partial x}} = {sp.latex(d_x)}, \quad \frac{{\partial F_y}}{{\partial y}} = {sp.latex(d_y)}, \quad \frac{{\partial F_z}}{{\partial z}} = {sp.latex(d_z)}")
            st.write("**Step 2: Sum them up for total Divergence**")
            st.latex(rf"\nabla \cdot \mathbf{{F}} = {sp.latex(total_div)}")
            div_ans = total_div.subs({x: val_x, y: val_y, z: val_z})
            st.write(f"**Step 3: Evaluate value at point ({val_x}, {val_y}, {val_z})**")
            u_calc = sp.lambdify((x, y, z), f_x, 'numpy')
            v_calc = sp.lambdify((x, y, z), f_y, 'numpy')
            w_calc = sp.lambdify((x, y, z), f_z, 'numpy')
        else:
            total_div = d_x + d_y
            st.latex(rf"\text{{2D Vector Field: }} \mathbf{{F}} = \left[ {sp.latex(f_x)},\, {sp.latex(f_y)} \right]")
            st.write("**Step 1: Differentiate each component by its axis**")
            st.latex(rf"\frac{{\partial F_x}}{{\partial x}} = {sp.latex(d_x)}, \quad \frac{{\partial F_y}}{{\partial y}} = {sp.latex(d_y)}")
            st.write("**Step 2: Sum them up for 2D Divergence**")
            st.latex(rf"\nabla \cdot \mathbf{{F}} = {sp.latex(total_div)}")
            div_ans = total_div.subs({x: val_x, y: val_y})
            st.write(f"**Step 3: Evaluate value at point ({val_x}, {val_y})**")
            u_calc = sp.lambdify((x, y), f_x, 'numpy')
            v_calc = sp.lambdify((x, y), f_y, 'numpy')
            
        st.latex(rf"\nabla \cdot \mathbf{{F}}|_{{point}} = \mathbf{{{div_ans}}}")

    elif op == "Curl":
        f_x = sp.sympify(eq_x)
        f_y = sp.sympify(eq_y)
        
        if view_mode == "3D Space (X-Y-Z)":
            f_z = sp.sympify(eq_z)
            c_x = sp.diff(f_z, y) - sp.diff(f_y, z)
            c_y = sp.diff(f_x, z) - sp.diff(f_z, x)
            c_z = sp.diff(f_y, x) - sp.diff(f_x, y)
            
            st.latex(rf"\text{{Vector Field: }} \mathbf{{F}} = \left[ {sp.latex(f_x)},\, {sp.latex(f_y)},\, {sp.latex(f_z)} \right]")
            st.write("**Step 1: Solve cross product matrix components**")
            st.latex(rf"i = {sp.latex(c_x)}, \quad j = {sp.latex(c_y)}, \quad k = {sp.latex(c_z)}")
            st.write("**Step 2: Collect into a Curl vector**")
            st.latex(rf"\nabla \times \mathbf{{F}} = \left[ {sp.latex(c_x)},\, {sp.latex(c_y)},\, {sp.latex(c_z)} \right]")
            ans_x = c_x.subs({x: val_x, y: val_y, z: val_z})
            ans_y = c_y.subs({x: val_x, y: val_y, z: val_z})
            ans_z = c_z.subs({x: val_x, y: val_y, z: val_z})
            st.write(f"**Step 3: Solve orientation at point ({val_x}, {val_y}, {val_z})**")
            st.latex(rf"\nabla \times \mathbf{{F}}|_{{point}} = \mathbf{{\left[ {ans_x},\, {ans_y},\, {ans_z} \right]}}")
            u_calc = sp.lambdify((x, y, z), c_x, 'numpy')
            v_calc = sp.lambdify((x, y, z), c_y, 'numpy')
            w_calc = sp.lambdify((x, y, z), c_z, 'numpy')
        else:
            c_z = sp.diff(f_y, x) - sp.diff(f_x, y)
            st.latex(rf"\text{{2D Vector Field: }} \mathbf{{F}} = \left[ {sp.latex(f_x)},\, {sp.latex(f_y)} \right]")
            st.write("**Step 1: Calculate the out-of-plane planar rotation component (Z-axis rotation)**")
            st.latex(rf"\text{{Curl}}_z = \frac{{\partial F_y}}{{\partial x}} - \frac{{\partial F_x}}{{\partial y}} = {sp.latex(c_z)}")
            ans_z = c_z.subs({x: val_x, y: val_y})
            st.write(f"**Step 2: Solve magnitude of rotation at point ({val_x}, {val_y})**")
            st.latex(rf"\text{{Curl}}_z|_{{point}} = \mathbf{{{ans_z}}}")
            u_calc = sp.lambdify((x, y), f_x, 'numpy') 
            v_calc = sp.lambdify((x, y), f_y, 'numpy')

    # Graph plotting starts here
    st.header("Interactive Visualizer")
    limit = 4
    
    if view_mode == "3D Space (X-Y-Z)":
        x_space = np.linspace(-limit, limit, arrows)
        y_space = np.linspace(-limit, limit, arrows)
        z_space = np.linspace(-limit, limit, arrows)
        X, Y, Z = np.meshgrid(x_space, y_space, z_space)
        
        U = np.array(u_calc(X, Y, Z), dtype=float)
        V = np.array(v_calc(X, Y, Z), dtype=float)
        W = np.array(w_calc(X, Y, Z), dtype=float)
        
        if U.ndim == 0: U = np.full_like(X, U)
        if V.ndim == 0: V = np.full_like(Y, V)
        if W.ndim == 0: W = np.full_like(Z, W)

        x_flat, y_flat, z_flat = X.flatten(), Y.flatten(), Z.flatten()
        u_flat, v_flat, w_flat = U.flatten(), V.flatten(), W.flatten()
        
        my_plot = go.Figure(data=go.Cone(
            x=x_flat, y=y_flat, z=z_flat,
            u=u_flat, v=v_flat, w=w_flat,
            colorscale='Viridis', sizemode='scaled', sizeref=1.2,
            colorbar=dict(title="Intensity")
        ))
        my_plot.add_trace(go.Scatter3d(
            x=[val_x], y=[val_y], z=[val_z], mode='markers',
            marker=dict(size=7, color='red'), name="Target Point"
        ))
        my_plot.update_layout(
            scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z'), aspectmode='cube'),
            margin=dict(l=0, r=0, b=0, t=10), height=600
        )
        st.plotly_chart(my_plot, use_container_width=True)
        
    else:
        # 2D stable layout using a pure Graph Objects line-based vector mapping
        x_space = np.linspace(-limit, limit, arrows * 2)
        y_space = np.linspace(-limit, limit, arrows * 2)
        X, Y = np.meshgrid(x_space, y_space)
        
        U = np.array(u_calc(X, Y), dtype=float)
        V = np.array(v_calc(X, Y), dtype=float)
        
        if U.ndim == 0: U = np.full_like(X, U)
        if V.ndim == 0: V = np.full_like(Y, V)
        
        # Flatten out grid layout coordinates
        xf, yf = X.flatten(), Y.flatten()
        uf, vf = U.flatten(), V.flatten()
