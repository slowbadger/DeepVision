# Minimal setup script for DeepVision forensic project
# Allows editable install with `pip install -e .`

from setuptools import setup, find_packages

setup(
    name="deepvision-forensics",
    version="0.1.0",
    description="DeepVision forensic model evaluation package",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=[
        # Core dependencies – ensure they match requirements.txt
        # These will be installed when you run `pip install -e .`
    ],
    # Optional: entry point for evaluation script
    entry_points={
        "console_scripts": [
            "deepvision-eval=src.evaluate:main",
        ]
    },
)
