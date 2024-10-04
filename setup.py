from setuptools import setup, find_packages

setup(
    name='video2panorama',
    version='1.0.3',
    author='Bolborici Radu-George',
    author_email='radu.bolborici@gmail.com',
    description='A package to convert video sequences into panoramic images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RaduBolbo/video_to_panorama.git', 
    packages=find_packages(),
    install_requires=[
        'imageio>=2.35.1',
        'matplotlib>=3.9.2',
        'numpy>=2.1.1',
        'opencv-python>=4.10.0.84',
        'pillow>=10.4.0',
        'tqdm>=4.66.5'
    ],
    entry_points={
        'console_scripts': [
            'video2panorama=video2panorama.main:main',  # This connects the CLI command to your main function
        ],
    },
    package_data={
        'video2panorama': ['hyperparameters.json'],  # Specify the JSON file
    },
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
    ],
    python_requires='>=3.8',
)