from setuptools import setup, find_packages

setup(
    name='asl',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torch-geometric',
        # 'numpy',
        # 'matplotlib',
        # 'tqdm'
    ],
    # Metadata
    author='Moussa JAMOR',
    author_email='moussajamorsup@gmail.com',
    description='',
    url='https://github.com/JamorMoussa/Sign-Language-Translation-using-Deep-Learning',
)