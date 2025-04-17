import os

folders = [
    "data",
    "models",
    "nn",
    "options",
    "pipeline",
    "tests",
    "paper"
]

files = {
    "README.md": "# Cointegration Algorithm\n\nProject description here.",
    "requirements.txt": "",
    "LICENSE": "MIT License",
    ".gitignore": "__pycache__/\n*.pyc\n.env\n",
    "data/loader.py": "",
    "data/preprocessing.py": "",
    "models/cointegration.py": "",
    "models/ecm.py": "",
    "models/bayesian_filter.py": "",
    "models/monte_carlo.py": "",
    "nn/entry_model.py": "",
    "nn/risk_model.py": "",
    "nn/train.py": "",
    "options/filter.py": "",
    "options/mapper.py": "",
    "pipeline/main.py": "",
    "pipeline/cron_jobs.py": "",
    "pipeline/monitor.py": "",
    "tests/test_data.py": "",
    "tests/test_models.py": "",
    "paper/Cointegration_Pipeline.tex": "",
    "paper/README.md": "White paper LaTeX files go here.",
}

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)

print("✅ Project structure created.")
