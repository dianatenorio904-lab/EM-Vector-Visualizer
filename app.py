import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp

st.set_page_config(page_title="EM Vector Tool", layout="wide")
st.title("⚡ EE Project: Vector Calculus Visualizer")
st.write("An interactive tool to solve and map Gradient, Divergence, and Curl equations for Electromagnetics.")

# Sidebar setup for user input variables
st.sidebar.header("Input Settings")

op = st.sidebar.selectbox(
    "Select Operation:",
    ["Gradient", "Divergence", "Curl"]
)

st.sidebar.subheader("Equations")
if op == "Gradient":
    user_eq = st.sidebar.text_input("f(x, y, z) =", "x**2 + y**2 - z")
else:
    eq_x = st.sidebar.text_input("Fx =", "-y")
    eq_y = st.sidebar.text_input("Fy =", "x")
    eq_z = st.sidebar.text_input("Fz =", "0")

st.sidebar.subheader("Point Evaluation")
val_x = st.sidebar.number_input("x position:", value=1.0)
val_y = st.sidebar.number_input("y position:", value=1.0)
val_z = st.sidebar.number_input("z position:", value=1.0)

arrows = st.sidebar.slider("Number of Arrows", 4, 10, 6)

# Setup variables for calculus equations
x, y, z = sp.symbols('x y z')

st.header("Step-by-Step Solution Breakdown")

try:
    if op == "Gradient":
        my_function = sp.sympify(user_eq)
        
        # calculate partial derivatives
        g_x = sp.diff(my_function, x)
        g_y = sp.diff(my_function, y)
        g_z = sp.diff(my_function, z)
        
        # printing the latex breakdown onto the web screen
        st.latex(rf"\text{{Given function: }} f = {sp.latex(my_function)}")
        st.write("**Step 1: Find partial derivatives for each axis**")
        st.latex(rf"\frac{{\partial f}}{{\partial x}} = {sp.latex(g_x)}")
        st.latex(rf"\frac{{\partial f}}{{\partial y}} = {sp.latex(g_y)}")
        st.latex(rf"\frac{{\partial f}}{{\partial z}} = {sp.latex(g_z)}")
        
        st.write("**Step 2: Write final gradient expression**")
        st.latex(rf"\nabla f = ({sp.latex(g_x)})\hat{{i}} + ({sp.latex(g_y)})\hat{{j}} + ({sp.latex(g_z)})\hat{{k}}")
        
        # evaluate at the selected points
        ans_x = g_x.subs({x: val_x, y: val_y, z: val_z})
        ans_y = g_y.subs({x: val_x, y: val_y, z: val_z})
        ans_z = g_z.subs({x: val_x, y: val_y, z: val_z})
        
        st.write(f"**Step 3: Plug in coordinates ({val_x}, {val_y}, {val_z})**")
        st.latex(rf"\nabla f|_{{point}} = \mathbf{{\left[ {ans_x},\, {ans_y},\, {ans_z} \right]}}")
        
        # convert formula to numpy format for grid graphing
        u_calc = sp.lambdify((x, y, z), g_x, 'numpy')
        v_calc = sp.lambdify((x, y, z), g_y, 'numpy')
        w_calc = sp.lambdify((x, y, z), g_z, 'numpy')

    elif op == "Divergence":
        f_x = sp.sympify(eq_x)
        f_y = sp.sympify(eq_y)
        f_z = sp.sympify(eq_z)
        
        # apply divergence formula
        d_x = sp.diff(f_x, x)
        d_y = sp.diff(f_y, y)
        d_z = sp.diff(f_z, z)
        total_div = d_x + d_y + d_z
        
        st.latex(rf"\text{{Vector Field: }} \mathbf{{F}} = \left[ {sp.latex(f_x)},\, {sp.latex(f_y)},\, {sp.latex(f_z)} \right]")
        st.write("**Step 1: Differentiate each component by its axis**")
        st.latex(rf"\frac{{\partial F_x}}{{\partial x}} = {sp.latex(d_x)}")
        st.latex(rf"\frac{{\partial F_y}}{{\partial y}} = {sp.latex(d_y)}")
        st.latex(rf"\frac{{\partial F_z}}{{\partial z}} = {sp.latex(d_z)}")
        
        st.write("**Step 2: Sum them up for total Divergence**")
        st.latex(rf"\nabla \cdot \mathbf{{F}} = {sp.latex(total_div)}")
        
        div_ans = total_div.subs({x: val_x, y: val_y, z: val_z})
        st.write(f"**Step 3: Evaluate magnitude value at point ({val_x}, {val_y}, {val_z})**")
        st.latex(rf"\nabla \cdot \mathbf{{F}}|_{{point}} = \mathbf{{{div_ans}}}")
        
        u_calc = sp.lambdify((x, y, z), f_x, 'numpy')
        v_calc = sp.lambdify((x, y, z), f_y, 'numpy')
        w_calc = sp.lambdify((x, y, z), f_z, 'numpy')

    elif op == "Curl":
        f_x = sp.sympify(eq_x)
        f_y = sp.sympify(eq_y)
        f_z = sp.sympify(eq_z)
        
        # determinant matrix calculations
        c_x = sp.diff(f_z, y) - sp.diff(f_y, z)
        c_y = sp.diff(f_x, z) - sp.diff(f_z, x)
        c_z = sp.diff(f_y, x) - sp.diff(f_x, y)
        
        st.latex(rf"\text{{Vector Field: }} \mathbf{{F}} = \left[ {sp.latex(f_x)},\, {sp.latex(f_y)},\, {sp.latex(f_z)} \right]")
        st.write("**Step 1: Solve cross product matrix components**")
        st.latex(rf"i = \frac{{\partial F_z}}{{\partial y}} - \frac{{\partial F_y}}{{\partial z}} = {sp.latex(c_x)}")
        st.latex(rf"j = \frac{{\partial F_x}}{{\partial z}} - \frac{{\partial F_z}}{{\partial x}} = {sp.latex(c_y)}")
        st.latex(rf"k = \frac{{\partial F_y}}{{\partial x}} - \frac{{\partial F_x}}{{\partial y}} = {sp.latex(c_z)}")
        
        st.write("**Step 2: Collect everything into a Curl vector**")
        st.latex(rf"\nabla \times \mathbf{{F}} = \left[ {sp.latex(c_x)},\, {sp.latex(c_y)},\, {sp.latex(c_z)} \right]")
        
        ans_x = c_x.subs({x: val_x, y: val_y, z: val_z})
        ans_y = c_y.subs({x: val_x, y: val_y, z: val_z})
        ans_z = c_z.subs({x: val_x, y: val_y, z: val_z})
        
        st.write(f"**Step 3: Solve vector orientation at point ({val_x}, {val_y}, {val_z})**")
        st.latex(rf"\nabla \times \mathbf{{F}}|_{{point}} = \mathbf{{\left[ {ans_x},\, {ans_y},\, {ans_z} \right]}}")
        
        u_calc = sp.lambdify((x, y, z), c_x, 'numpy')
        v_calc = sp.lambdify((x, y, z), c_y, 'numpy')
        w_calc = sp.lambdify((x, y, z), c_z, 'numpy')

    # Graph plotting starts here
    st.header("3D Field Plot View")
    
    # create the space boundaries
    limit = 4
    x_space = np.linspace(-limit, limit, arrows)
    y_space = np.linspace(-limit, limit, arrows)
    z_space = np.linspace(-limit, limit, arrows)
    X, Y, Z = np.meshgrid(x_space, y_space, z_space)
    
    # fill up arrays with calculated numbers
    U = np.array(u_calc(X, Y, Z), dtype=float)
    V = np.array(v_calc(X, Y, Z), dtype=float)
    W = np.array(w_calc(X, Y, Z), dtype=float)
    
    # fix array formatting if values are static constants
    if U.ndim == 0: U = np.full_like(X, U)
    if V.ndim == 0: V = np.full_like(Y, V)
    if W.ndim == 0: W = np.full_like(Z, W)

    # flatten for plotting
    x_flat, y_flat, z_flat = X.flatten(), Y.flatten(), Z.flatten()
    u_flat, v_flat, w_flat = U.flatten(), V.flatten(), W.flatten()
    
    # build the 3D cone plot object
    my_plot = go.Figure(data=go.Cone(
        x=x_flat, y=y_flat, z=z_flat,
        u=u_flat, v=v_flat, w=w_flat,
        colorscale='Viridis',
        sizemode='scaled',
        sizeref=1.2,
        colorbar=dict(title="Intensity")
    ))
    
    # plot the target coordinate point marker
    my_plot.add_trace(go.Scatter3d(
        x=[val_x], y=[val_y], z=[val_z],
        mode='markers',
        marker=dict(size=7, color='red'),
        name="Target Point"
    ))
    
    my_plot.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[-limit, limit]),
            yaxis=dict(title='Y', range=[-limit, limit]),
            zaxis=dict(title='Z', range=[-limit, limit]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=600
    )
    
    st.plotly_chart(my_plot, use_container_width=True)

except Exception as math_error:
    st.error(f"Equation syntax error. Check your math symbols! Extra information: {math_error}")
