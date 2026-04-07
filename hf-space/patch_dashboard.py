import re

file_path = r'c:\Users\Vincenzo\Desktop\METAHACKATHON\dashboard\app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Remove global `from evaluate import load_model`
target1 = r"""from evaluate import load_model"""
repl1 = r""
content = content.replace(target1, repl1)

# Fix 2: Add lazy import
target2 = r"""            if not model_file:
                st.error("Please enter a model path.")
                return
            agent = load_model(model_file)"""
repl2 = r"""            if not model_file:
                st.error("Please enter a model path.")
                return
            from evaluate import load_model
            agent = load_model(model_file)"""
content = content.replace(target2, repl2)

# Fix 3: auto_speed min_value=2
target3 = r"""auto_speed = st.sidebar.slider("Auto-play speed (steps/sec)", min_value=1, max_value=10, value=5)"""
repl3 = r"""auto_speed = st.sidebar.slider("Auto-play speed (steps/sec)", min_value=2, max_value=10, value=5)"""
content = content.replace(target3, repl3)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("dashboard app.py patched")
