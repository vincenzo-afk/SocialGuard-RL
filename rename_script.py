import os

def rename_in_file(filepath):
    # Added some more specific ones based on user's recent manual edits
    replacements = [
        ("SocialGuard-RL", "SocialGuard-RL"),
        ("SocialGuard_RL", "SocialGuard_RL"),
        ("socialguard-rl", "socialguard-rl"),
        ("socialguard_rl", "socialguard_rl"),
        ("SocialGuardMlpExtractor", "SocialGuardMlpExtractor"),
        ("SocialGuardNetBackbone", "SocialGuardNetBackbone"),
        ("SocialGuardPolicy", "SocialGuardPolicy"),
        ("SocialGuardNet", "SocialGuardNet"),
        ("SocialGuardEnv", "SocialGuardEnv"),
        ("SocialGuard", "SocialGuard"),
        ("SocialGuard", "SocialGuard"),
        ("socialguard", "socialguard"),
        ("SocialGuard_MODEL_PATH", "SocialGuard_MODEL_PATH"),
        ("X-SocialGuard-Version", "X-SocialGuard-Version"),
        ("socialguard_saved_runs", "socialguard_saved_runs"),
        ("socialguard_grade_history", "socialguard_grade_history"),
        ("socialguard_judge_notes", "socialguard_judge_notes"),
    ]
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        original_content = content
        for old, new in replacements:
            content = content.replace(old, new)
            
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Renamed in: {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', 'env'}
    extensions = {'.py', '.yaml', '.yml', '.md', '.html', '.txt', '.json', '.cfg', '.toml', '.ini', '.sh'}
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                rename_in_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
