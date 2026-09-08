# check_requirements.py

packages = [
    "langchain",
    "langchain_community",
    "langchain_openai",
    "streamlit",
    "pandas",
    "numpy",
    "sklearn",
    "torch",
    "sqlite3",
    "matplotlib",
    "seaborn",
    "meteostat",
    "joblib",
    "requests"
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except:
        print(f"❌ {pkg}")
