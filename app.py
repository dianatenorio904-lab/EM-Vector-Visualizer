import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp

st.set_page_config(page_title="EM Vector Tool", layout="wide")
st.title("Electromagnetics: Interactive Vector Calculus Visualizer")

# Sidebar setup for user input variables
st.sidebar.header("Control Console")
view_mode = "3D Space (X-Y-Z)" # Volumetric space optimized for mesh surface mapping

op = st.sidebar.selectbox(
    "Select Math Operator:",
    ["Gradient (∇f)", "Divergence (∇ · F)", "Curl (∇ × F)"]
)

st.sidebar.subheader("Equations (Use x, y, z)")
st.sidebar.caption("Tip: Write exponential functions simply as 'exp()', multiplication as '*', and powers as '**'.")

# Default equations setup
if op == "Gradient (∇f)":
    user_eq = st.sidebar.text_input("f(x, y) or f(x, y, z) =", "7*x*y / exp(x**2 + y**2)")
else:
    eq_x = st.sidebar.text_input("Fx =", "-y")
    eq_y = st.sidebar.text_input("Fy =", "x")
    eq_z = st.sidebar.text_input("Fz =", "0")

st.sidebar.subheader("Point Coordinates")
val_x = st.sidebar.number_input("Marker X position:", value=1.0)
val_y = st.sidebar.number_input("Marker Y position:", value=1.0)
val_z = st.sidebar.number_input("Marker Z position:", value=0.0)

gridlines = st.sidebar.slider("Mesh Resolution (Gridlines)", 10, 50, 31)

# Initialize variables for the symbolic calculus engine
x, y, z = sp.symbols('x y z')

safe_dict = {
    "x": x, "y": y, "z": z,
    "exp": sp.exp, "sin": sp.sin, "cos": sp.cos, "sqrt": sp.sqrt, "pi": sp.pi
}

try:
    # --- MATH COMPILATION PIPELINE ---
    if op == "Gradient (∇f)":
        my_function = sp.sympify(user_eq, locals=safe_dict)
        g_x = sp.diff(my_function, x)
        g_y = sp.diff(my_function, y)
        g_z = sp.diff(my_function, z)
        
        # Point values calculation
        ans_x = float(g_x.subs({x: val_x, y: val_y, z: val_z}))
        ans_y = float(g_y.subs({x: val_x, y: val_y, z: val_z}))
        ans_z = float(g_z.subs({x: val_x, y: val_y, z: val_z}))
        
        flattened_function = my_function.subs(z, val_z)
        z_mesh_func = sp.lambdify((x, y), flattened_function, 'numpy')

    elif op == "Divergence (∇ · F)":
        f_x = sp.sympify(eq_x, locals=safe_dict)
        f_y = sp.sympify(eq_y, locals=safe_dict)
        f_z = sp.sympify(eq_z, locals=safe_dict)
        
        d_x = sp.diff(f_x, x)
        d_y = sp.diff(f_y, y)
        d_z = sp.diff(f_z, z)
        total_div = d_x + d_y + d_z
        
        div_ans = float(total_div.subs({x: val_x, y: val_y, z: val_z}))
        
        flattened_function = total_div.subs(z, val_z)
        z_mesh_func = sp.lambdify((x, y), flattened_function, 'numpy')

    elif op == "Curl (∇ × F)":
        f_x = sp.sympify(eq_x, locals=safe_dict)
        f_y = sp.sympify(eq_y, locals=safe_dict)
        f_z = sp.sympify(eq_z, locals=safe_dict)
        
        c_x = sp.diff(f_z, y) - sp.diff(f_y, z)
        c_y = sp.diff(f_x, z) - sp.diff(f_z, x)
        c_z = sp.diff(f_y, x) - sp.diff(f_x, y)
        
        ans_x = float(c_x.subs({x: val_x, y: val_y, z: val_z}))
        ans_y = float(c_y.subs({x: val_x, y: val_y, z: val_z}))
        ans_z = float(c_z.subs({x: val_x, y: val_y, z: val_z}) )
        
        flattened_function = c_z.subs(z, val_z)
        z_mesh_func = sp.lambdify((x, y), flattened_function, 'numpy')

    # --- CREATIVITY REQUIREMENT: INTERACTIVE METRICS DASHBOARD ---
    st.markdown("### Real-Time Diagnostic Evaluation")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.metric(label="Target Coordinate Evaluation Point", value=f"({val_x}, {val_y}, {val_z})")
    with m_col2:
        st.metric(label="Active Mathematical Operator", value=f"{op}")
    with m_col3:
        # Calculate localized system output magnitude
        if op == "Divergence (∇ · F)":
            st.metric(label="Solved Divergence Scalar Value", value=f"{div_ans:.4f}")
        elif op == "Gradient (∇f)":
            vec_mag = np.sqrt(ans_x**2 + ans_y**2 + ans_z**2)
            st.metric(label="Gradient Vector Magnitude", value=f"{vec_mag:.4f}")
        else:
            vec_mag = np.sqrt(ans_x**2 + ans_y**2 + ans_z**2)
            st.metric(label="Curl Vector Magnitude", value=f"{vec_mag:.4f}")

    st.write("---")

    # --- ORIGINALITY REQUIREMENT: HIGH-LEVEL INTERACTIVE TABS CONFIGURATION ---
    tab1, tab2 = st.tabs(["3D Interactive Visualizer", "Analytical Calculus Solver Breakdown"])

    with tab1:
        # Build 3D mesh arrays
        limit = 3.0
        x_range = np.linspace(-limit, limit, gridlines)
        y_range = np.linspace(-limit, limit, gridlines)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array(z_mesh_func(X, Y), dtype=float)
        if Z.ndim == 0: Z = np.full_like(X, Z)

        my_plot = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Geyser', 
            contours=dict(
                x=dict(show=True, color="black", width=1),
                y=dict(show=True, color="black", width=1)
            ),
            lighting=dict(ambient=0.7, roughness=0.1) 
        )])
        
        eval_z_val = float(z_mesh_func(val_x, val_y))
        my_plot.add_trace(go.Scatter3d(
            x=[val_x], y=[val_y], z=[eval_z_val], mode='markers',
            marker=dict(size=9, color='red', symbol='circle'), name="Tracking Point"
        ))
        
        my_plot.update_layout(
            scene=dict(
                xaxis=dict(title='X Axis', backgroundcolor="white", gridcolor="lightgray"),
                yaxis=dict(title='Y Axis', backgroundcolor="white", gridcolor="lightgray"),
                zaxis=dict(title='Z Axis', backgroundcolor="white", gridcolor="lightgray"),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=10), height=600
        )
        st.plotly_chart(my_plot, use_container_width=True)

    with tab2:
        # Display the LaTeX breakdown steps here out of the way of the graph view canvas
        if op == "Gradient (∇f)":
            st.latex(rf"\text{{Given Scalar Field: }} f = {sp.latex(my_function)}")
            st.write("### **Step 1: Compute partial derivatives for each unit axis:**")
            st.latex(rf"\frac{{\partial f}}{{\partial x}} = {sp.latex(g_x)}")
            st.latex(rf"\frac{{\partial f}}{{\partial y}} = {sp.latex(g_y)}")
            st.latex(rf"\frac{{\partial f}}{{\partial z}} = {sp.latex(g_z)}")
            st.write("### **Step 2: Assemble the analytical Gradient expression:**")
            st.latex(rf"\nabla f = \left({sp.latex(g_x)}\right)\hat{{i}} + \left({sp.latex(g_y)}\right)\hat{{j}} + \left({sp.latex(g_z)}\right)\hat{{k}}")
            st.write(f"### **Step 3: Solved coordinate values evaluated at point ({val_x}, {val_y}, {val_z}):**")
            st.latex(rf"\nabla f|_{{point}} = \mathbf{{\left[ {ans_x},\, {ans_y},\, {ans_z} \right]}}")

        elif op == "Divergence (∇ · F)":
            st.latex(rf"\text{{Given Vector Field F: }} \mathbf{{F}} = \left[ {sp.latex(f_x)},\, {sp.latex(f_y)},\, {sp.latex(f_z)} \right]")
            st.write("### **Step 1: Compute corresponding directional partial derivatives:**")
            st.latex(rf"\frac{{\partial F_x}}{{\partial x}} = {sp.latex(d_x)}, \quad \frac{{\partial F_y}}{{\partial y}} = {sp.latex(d_y)}, \quad \frac{{\partial F_z}}{{\partial z}} = {sp.latex(d_z)}")
            st.write("### **Step 2: Sum the partial derivatives to determine net Divergence scale value:**")
            st.latex(rf"\nabla \cdot \mathbf{{F}} = {sp.latex(total_div)}")
            st.write(f"### **Step 3: Solved scalar system flow magnitude evaluated at point ({val_x}, {val_y}, {val_z}):**")
            st.latex(rf"\nabla \cdot \mathbf{{F}}|_{{point}} = \mathbf{{{div_ans}}}")

        elif op == "Curl (∇ × F)":
            st.latex(rf"\text{{Given Vector Field F: }} \mathbf{{F}} = \left[ {sp.latex(f_x)},\, {sp.latex(f_y)},\, {sp.latex(f_z)} \right]")
            st.write("### **Step 1: Expand the cross-product matrix determinant components:**")
            st.latex(rf"i = {sp.latex(c_x)}, \quad j = {sp.latex(c_y)}, \quad k = {sp.latex(c_z)}")
            st.write("### **Step 2: Assemble the analytical rotational Curl expression vector:**")
            st.latex(rf"\nabla \times \mathbf{{F}} = \left[ {sp.latex(c_x)},\, {sp.latex(c_y)},\, {sp.latex(c_z)} \right]")
            st.write(f"### **Step 3: Solved directional orientation vectors evaluated at point ({val_x}, {val_y}, {val_z}):**")
            st.latex(rf"\nabla \times \mathbf{{F}}|_{{point}} = \mathbf{{\left[ {ans_x},\, {ans_y},\, {ans_z} \right]}}")

except Exception as math_parse_error:
    st.error(f"❌ Input Equation Syntax Error. Remember to use explicit asterisks '*' for multiplication. Details: {math_parse_error}")
