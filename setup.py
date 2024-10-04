from setuptools import setup, find_packages

setup(
    name='video2panorama',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'video2panorama=video2panorama.main:main',  # Define the CLI command and link to your main function
        ],
    },
    python_requires='>=3.6',  # Adjust according to your Python version requirement
)