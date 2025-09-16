from setuptools import setup, find_packages

setup(
    name="P_KNN",
    version="1.0.0",
    description="P-KNN command line tool",
    author="Po-Yu Lin and Nadav Brandes",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "huggingface_hub"
    ],
    extras_require={
        "gpu": ["torch"],
        "cpu": ["joblib"],
        "all": ["torch", "joblib"]
    },
    entry_points={
        "console_scripts": [
            "P_KNN = P_KNN.P_KNN:main",
            "P_KNN_config = P_KNN.P_KNN_config:main"
            "P_KNN_memory_estimator = P_KNN.P_KNN_memory_estimator:main"
        ]
    },
    include_package_data=False
)