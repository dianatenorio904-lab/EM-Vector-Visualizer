import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp

# Page Layout configuration
st.set_page_config(page_title="EM Vector Calculus Tool", layout="wide")
st.title("Electromagnetics: Vector Calculus Visualizer & Solver")
st.write("Type in custom formulas to instantly compute step-by-step vector operations and see them in 3D.")

# Sidebar Controls & Mathematical Problem Inputs
st.sidebar.header("Problem Definition")

operator = st.sidebar.selectbox(
    "Select Math Operator:",
    ["Gradient (∇f)", "Divergence (∇ · F)", "Curl (∇ × F)"]
)

st.sidebar.subheader("Inputs (Use x, y, z)")
if operator == "Gradient (∇f)":
    st.sidebar.caption("Define a scalar field function f(x, y, z):")
    f_input = st.sidebar.text_input("f(x, y, z) =", "x**2 + y**2 - z")
else:
    st.sidebar.caption("Define vector components F = (Fx, Fy, Fz):")
    fx_input = st.sidebar.text_input("Fx =", "-y")
    fy_input = st.sidebar.text_input("Fy =", "x")
    fz_input = st.sidebar.text_input("Fz =", "0")

st.sidebar.subheader("Evaluation Point")
px = st.sidebar.number_input("Point X coordinate:", value=1.0)
py = st.sidebar.number_input("Point Y coordinate:", value=1.0)
pz = st.sidebar.number_input("Point Z coordinate:", value=1.0)

grid_density = st.sidebar.slider("3D Grid Arrow Density", 4, 10, 6)

# Symbolic Calculus Math Engine (SymPy Solvers)
x, y, z = sp.symbols('x y z')

st.header("Analytical Step-by-Step Solution")

try:
    if operator == "Gradient (∇f)":
        f_expr = sp.sympify(f_input)
        
        # Differentiation steps
        df_dx = sp.diff(f_expr, x)
        df_dy = sp.diff(f_expr, y)
        df_dz = sp.diff(f_expr, z)
        grad_vector = [df_dx, df_dy, df_dz]
        
        # Display symbolic breakdown
        st.latex(rf"\text{{Given Scalar Field: }} f(x,y,z) = {sp.latex(f_expr)}")
        st.markdown("**Step 1: Compute partial derivatives for each unit vector axis:**")
        st.latex(rf"\frac{{\partial f}}{{\partial x}} = {sp.latex(df_dx)}")
        st.latex(rf"\frac{{\partial f}}{{\partial y}} = {sp.latex(df_dy)}")
        st.latex(rf"\frac{{\partial f}}{{\partial z}} = {sp.latex(df_dz)}")
        
        st.markdown("**Step 2: Assemble the analytical Gradient expression:**")
        st.latex(rf"\nabla f = \left( {sp.latex(df_dx)} \right)\hat{{i}} + \left( {sp.latex(df_dy)} \right)\hat{{j}} + \left( {sp.latex(df_dz)} \right)\hat{{k}}")
        
        # Coordinate value substitution
        val_x = df_dx.subs({x: px, y: py, z: pz})
        val_y = df_dy.subs({x: px, y: py, z: pz})
        val_z = df_dz.subs({x: px, y: py, z: pz})
        
        st.markdown(f"**Step 3: Evaluate result coordinates at target point ({px}, {py}, {pz}):**")
        st.latex(rf"\nabla f|_{{({px},{py},{pz})}} = \mathbf{{\left[ {val_x},\, {val_y},\, {val_z} \right]}}")
        
        # Lambda conversion for numerical 3D graphing arrays
        U_func = sp.lambdify((x, y, z), df_dx, 'numpy')
        V_func = sp.lambdify((x, y, z), df_dy, 'numpy')
        W_func = sp.lambdify((x, y, z), df_dz, 'numpy')

    elif operator == "Divergence (∇ · F)":
        Fx = sp.sympify(fx_input)
        Fy = sp.sympify(fy_input)
        Fz = sp.sympify(fz_input)
        
        # Differentiation steps
        dFx_dx = sp.diff(Fx, x)
        dFy_dy = sp.diff(Fy, y)
        dFz_dz = sp.diff(Fz, z)
        div_expr = dFx_dx + dFy_dy + dFz_dz
        
        # Display symbolic breakdown
        st.latex(rf"\text{{Given Vector Field F: }} \mathbf{{F}} = \left[ {sp.latex(Fx)},\, {sp.latex(Fy)},\, {sp.latex(Fz)} \right]")
        st.markdown("**Step 1: Compute corresponding directional partial derivatives:**")
        st.latex(rf"\frac{{\partial F_x}}{{\partial x}} = {sp.latex(dFx_dx)}")
        st.latex(rf"\frac{{\partial F_y}}{{\partial y}} = {sp.latex(dFy_dy)}")
        st.latex(rf"\frac{{\partial F_z}}{{\partial z}} = {sp.latex(dFz_dz)}")
        
        st.markdown("**Step 2: Sum partial derivatives to form the Divergence expression:**")
        st.latex(rf"\nabla \cdot \mathbf{{F}} = \frac{{\partial F_x}}{{\partial x}} + \frac{{\partial F_y}}{{\partial y}} + \frac{{\partial F_z}}{{\partial z}} = {sp.latex(div_expr)}")
        
        # Coordinate value substitution
        div_val = div_expr.subs({x: px, y: py, z: pz})
        st.markdown(f"**Step 3: Evaluate scale magnitude at target point ({px}, {py}, {pz}):**")
        st.latex(rf"\nabla \cdot \mathbf{{F}}|_{{({px},{py},{pz})}} = \mathbf{{{div_val}}}")
        
        # Lambda conversion for vector baseline visualization
        U_func = sp.lambdify((x, y, z), Fx, 'numpy')
        V_func = sp.lambdify((x, y, z), Fy, 'numpy')
        W_func = sp.lambdify((x, y, z), Fz, 'numpy')

    elif operator == "Curl (∇ × F)":
        Fx = sp.sympify(fx_input)
        Fy = sp.sympify(fy_input)
        Fz = sp.sympify(fz_input)
        
        # Determinant cross-product expansions
        curl_x = sp.diff(Fz, y) - sp.diff(Fy, z)
        curl_y = sp.diff(Fx, z) - sp.diff(Fz, x)
        curl_z = sp.diff(Fy, x) - sp.diff(Fx, y)
        
        # Display symbolic breakdown
        st.latex(rf"\text{{Given Vector Field F: }} \mathbf{{F}} = \left[ {sp.latex(Fx)},\, {sp.latex(Fy)},\, {sp.latex(Fz)} \right]")
        st.markdown("**Step 1: Expand the cross-product matrix determinant components:**")
        st.latex(rf"(\nabla \times \mathbf{{F}})_x = \frac{{\partial F_z}}{{\partial y}} - \frac{{\partial F_y}}{{\partial z}} = {sp.latex(curl_x)}")
        st.latex(rf"(\nabla \times \mathbf{{F}})_y = \frac{{\partial F_x}}{{\partial z}} - \frac{{\partial F_z}}{{\partial x}} = {sp.latex(curl_y)}")
        st.latex(rf"(\nabla \times \mathbf{{F}})_z = \frac{{\partial F_y}}{{\partial x}} - \frac{{\partial F_x}}{{\partial y}} = {sp.latex(curl_z)}")
        
        st.markdown("**Step 2: Assemble the collective analytical Curl vector expression:**")
        st.latex(rf"\nabla \times \mathbf{{F}} = \left[ {sp.latex(curl_x)},\, {sp.latex(curl_y)},\, {sp.latex(curl_z)} \right]")
        
        # Coordinate value substitution
        val_x = curl_x.subs({x: px, y: py, z: pz})
        val_y = curl_y.subs({x: px, y: py, z: pz})
        val_z = curl_z.subs({x: px, y: py, z: pz})
        
        st.markdown(f"**Step 3: Evaluate directional vectors at target point ({px}, {py}, {pz}):**")
        st.latex(rf"\nabla \times \mathbf{{F}}|_{{({px},{py},{pz})}} = \mathbf{{\left[ {val_x},\, {val_y},\, {val_z} \right]}}")
        
        # Lambda conversion to map out rotational directions in the graph
        U_func = sp.lambdify((x, y, z), curl_x, 'numpy')
        V_func = sp.lambdify((x, y, z), curl_y, 'numpy')
        W_func = sp.lambdify((x, y, z), curl_z, 'numpy')

    # 3D Vector Graph Plotting Engine (Plotly Cones)
    st.header("3D Resultant Field Map Visualization")
    
    max_val = 4
    x_range = np.linspace(-max_val, max_val, grid_density)
    y_range = np.linspace(-max_val, max_val, grid_density)
    z_range = np.linspace(-max_val, max_val, grid_density)
    X, Y, Z = np.meshgrid(x_range, y_range, z_range)
    
    # Generate numerical vector grids safely
    U = np.array(U_func(X, Y, Z), dtype=float)
    V = np.array(V_func(X, Y, Z), dtype=float)
    W = np.array(W_func(X, Y, Z), dtype=float)
    
    # Broadcast scalar fields if inputs returned constant numerical values instead of arrays
    if U.ndim == 0: U = np.full_like(X, U)
    if V.ndim == 0: V = np.full_like(Y, V)
    if W.ndim == 0: W = np.full_like(Z, W)

    # Flatten coordinates for Plotly formatting requirements
    X_f, Y_f, Z_f = X.flatten(), Y.flatten(), Z.flatten()
    U_f, V_f, W_f = U.flatten(), V.flatten(), W.flatten()
    
    # Render the vector array
    fig = go.Figure(data=go.Cone(
        x=X_f, y=Y_f, z=Z_f,
        u=U_f, v=V_f, w=W_f,
        colorscale='Plasma',
        sizemode='scaled',
        sizeref=1.2,
        showscale=True,
        colorbar=dict(title="Field Strength")
    ))
    
    # Add a distinct marker dot exactly where the evaluation coordinates sit
    fig.add_trace(go.Scatter3d(
        x=[px], y=[py], z=[pz],
        mode='markers',
        marker=dict(size=8, color='cyan', symbol='diamond'),
        name="Target Eval Point"
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X Axis', range=[-max_val, max_val]),
            yaxis=dict(title='Y Axis', range=[-max_val, max_val]),
            zaxis=dict(title='Z Axis', range=[-max_val, max_val]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=20),
        height=650
    )
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Math Error: Make sure your formula syntax is correct. Use '*' for multiplication and '**' for powers. Details: {e}")

