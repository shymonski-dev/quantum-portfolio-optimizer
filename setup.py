from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent
README = (BASE_DIR / "README.md").read_text(encoding="utf-8")

setup(
    name="local-quantum-portfolio-optimizer",
    version="0.2.1",
    description="Institutional-grade quantum portfolio optimization toolkit",
    long_description=README,
    long_description_content_type="text/markdown",
    author="shymonski-dev",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "qiskit>=2.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "PyYAML>=6.0",
        "click>=8.0",
        "flask>=2.3.0",
        "qiskit-addon-cutting>=0.2.0",
        "networkx>=3.0",
        "openbb-core>=1.6.0",
        "openbb-equity>=1.6.0",
        "openbb-yfinance>=1.4.0",
        "openbb-tiingo>=1.4.0",
        "yfinance==0.2.55",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0", "ruff>=0.1.0"],
        "noise": ["qiskit-aer>=0.14.0"],
        "ibm": ["qiskit-ibm-runtime>=0.40.0"],
        "all": ["qiskit-aer>=0.14.0", "qiskit-ibm-runtime>=0.40.0"],
    },
    project_urls={
        "Homepage": "https://github.com/shymonski-dev/quantum-portfolio-optimizer",
        "Bug Tracker": "https://github.com/shymonski-dev/quantum-portfolio-optimizer/issues",
        "Source": "https://github.com/shymonski-dev/quantum-portfolio-optimizer",
        "Documentation": "https://github.com/shymonski-dev/quantum-portfolio-optimizer#readme",
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 (AGPLv3)",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    entry_points={
        "console_scripts": [
            "qpo=quantum_portfolio_optimizer.cli:cli",
        ],
    },
)
