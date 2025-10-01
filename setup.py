from setuptools import setup, find_packages

setup(
    name='modellib',
    version='0.1.0',
    description='A custom Python module for streamlining plant disease detection model development pipeline.',
    author='vnltbb',  # 이 부분은 사용자님의 정보로 수정하세요.
    author_email='myhajung@gmail.com', # 이 부분은 사용자님의 정보로 수정하세요.
    url='https://github.com/vnltbb', # GitHub 리포지토리가 있다면 추가하세요.
    packages=["modellib"],
    install_requires=[
        'torch',
        'torchvision',
        'timm',
        'numpy',
        'optuna',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'torchinfo',
        'grad-cam',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)