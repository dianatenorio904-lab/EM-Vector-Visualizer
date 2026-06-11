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

st.sidebar.subheader("Equation Selection Method")
input_method = st.sidebar.radio("Choose Input Style:", ["Choose from Textbook Examples", "Type Custom Equation"])

# Smart math text formatter helper function
def format_user_string(input_str):
    """Automatically converts standard textbook math shorthand into clear explicit syntax."""
    # Handle implicit multiplication cases commonly typed by users
    input_str = input_str.replace("xy", "x*y").replace("xz", "x*z").replace("yz", "y*z")
    input_str = input_str.replace("2x", "2*x").replace("3x", "3*x").replace("4x", "4*x").replace("7x", "7*x").replace("8x", "8*x")
    input_str = input_str.replace("2y", "2*y").replace("3y", "3*y").replace("2z", "2*z")
    return input_str

# -------------------------------------------------------------
# DYNAMIC DROP-DOWN MENUS WITH PRESET PROBLEMS
# -------------------------------------------------------------
if op == "Gradient (∇f)":
    if input_method == "Choose from Textbook Examples":
        preset = st.sidebar.selectbox(
            "Select Example Problem:",
            [
                "Example 1: Wavy Waveform (7xy / exp(x^2 + y^2))",
                "Example 2: Paraboloid Basin (x^2 + y^2 - z)",
                "Example 3: Electric Field Potential (1 / sqrt(x^2 + y^2 + z^2 + 0.1))",
                "Example 4: Sinusoidal Grid Slices (sin(x) * cos(y))"
            ]
        )
        if "Example 1" in preset: user_eq = "7*x*y / exp(x^2 + y^2)"
        elif "Example 2" in preset: user_eq = "x^2 + y^2 - z"
        elif "Example 3" in preset: user_eq = "1 / sqrt(x^2 + y^2 + z^2 + 0.1)"
        else: user_eq = "sin(x) * cos(y)"
    else:
        st.sidebar.caption("💡 Guide: You can now use standard textbook notation like '^' for powers and ordinary terms like '8xy'.")
        raw_eq = st.sidebar.text_input("f(x, y, z) =", "7*x*y / exp(x^2 + y^2)")
        user_eq = format_user_string(raw_eq)

else: # Divergence and Curl Fields
    if input_method == "Choose from Textbook Examples":
        preset = st.sidebar.selectbox(
            "Select Example Problem Field:",
            [
                "Field A: Rotational Magnetic Vortex (Fx=-y, Fy=x, Fz=0)",
                "Field B: Outward Exploding Electric Field (Fx=x, Fy=y, Fz=z)",
                "Field C: Solenoidal Flow field (Fx=sin(y), Fy=cos(x), Fz=0)",
                "Field D: Polynomial Sample (Fx=8xy, Fy=2zx^2, Fz=-10x)"
            ]
        )
        if "Field A" in preset: eq_x, eq_y, eq_z = "-y", "x", "0"
        elif "Field B" in preset: eq_x, eq_y, eq_z = "x", "y", "z"
        elif "Field C" in preset: eq_x, eq_y, eq_z = "sin(y)", "cos(x)", "0"
        else: eq_x, eq_y, eq_z = "8*x*y", "2*z*x^2", "-10*x"
    else:
        st.sidebar.caption("💡 Guide: You can now use standard textbook notation like '^' for powers and ordinary terms like '8xy'.")
        raw_x = st.sidebar.text_input("Fx =", "-y")
        raw_y = st.sidebar.text_input("Fy =", "x")
        raw_z = st.sidebar.text_input("Fz =", "0")
        eq_x = format_user_string(raw_x)
        eq_y = format_user_string(raw_y)
        eq_z = format_user_string(raw_z)

st.sidebar.subheader("Point Evaluation Marker")
val_x = st.sidebar.number_input("Marker X position:", value=1.0)
val_y = st.sidebar.number_input("Marker Y position:", value=1.0)
val_z = st.sidebar.number_input("Marker Z position:", value=0.0)

gridlines = st.sidebar.slider("Number of Gridlines", 10, 50, 31)

# Initialize variables for the symbolic calculus engine
x, y, z = sp.symbols('x y z')

# Provide a dictionary mapping to safely translate raw strings into SymPy modules
safe_dict = {
    "x": x, "y": y, "z": z,
    "exp": sp.exp, "sin": sp.sin, "cos": sp.cos, "sqrt": sp.sqrt, "pi": sp.pi
}

st.header("📖 Analytical Step-by-Step Solution")

try:
    if op == "Gradient (∇f)":
        # FIXED: convert_xor=True turns standard ^ symbols into proper python exponents under the hood
        my_function = sp.sympify(user_eq, locals=safe_dict, convert_xor=True)
        
        # Calculate partial derivatives
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

        flattened_function = my_function.subs(z, val_z)
        z_mesh_func = sp.lambdify((x, y), flattened_function, 'numpy')

    elif op == "Divergence (∇ · F)":
        f_x = sp.sympify(eq_x, locals=safe_dict, convert_xor=True)
        f_y = sp.sympify(eq_y, locals=safe_dict, convert_xor=True)
        f_z = sp.sympify(eq_z, locals=safe_dict, convert_xor=True)
        
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
        
        flattened_function = total_div.subs(z, val_z)
        z_mesh_func = sp.lambdify((x, y), flattened_function, 'numpy')

    elif op == "Curl (∇ × F)":
        f_x = sp.sympify(eq_x, locals=safe_dict, convert_xor=True)
        f_y = sp.sympify(eq_y, locals=safe_dict, convert_xor=True)
        f_z = sp.sympify(eq_z, locals=safe_dict, convert_xor=True)
        
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
            xaxis=dict(title='X Axis', backgroundcolor="white", gridcolor="lightgray"))
