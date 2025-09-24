#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocalLM 安裝配置文件
"""

from setuptools import setup, find_packages
import os

# 讀取 README 文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "LocalLM - 本地大語言模型智能助手"

# 讀取 requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="locallm",
    version="2.0.0",
    author="LocalLM Team",
    author_email="team@locallm.dev",
    description="本地大語言模型智能助手 - 提供強大的本地AI能力",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/locallm/locallm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "full": [
            "jupyter>=1.0",
            "matplotlib>=3.0",
            "seaborn>=0.11",
            "plotly>=5.0",
            "pandas>=1.3",
            "numpy>=1.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "locallm=locallm.cli.global_entry:main",
            "ai-assistant=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "locallm": [
            "config/*.yaml",
            "config/*.json",
            "data/**/*",
        ],
    },
    zip_safe=False,
    keywords="ai, llm, local, ollama, assistant, cli, automation",
    project_urls={
        "Bug Reports": "https://github.com/locallm/locallm/issues",
        "Source": "https://github.com/locallm/locallm",
        "Documentation": "https://docs.locallm.dev",
    },
)
