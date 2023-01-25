from setuptools import setup, find_packages

setup(name='route-choice-env',
      version='0.0.1',
      description='Multi-Agent RL Route Choice Environment',
      url='https://github.com/ramos-ai/route-choice-gym',
      author='Luiz Alfredo Thomasini',
      author_email='luizalfredo@edu.unisinos.br',
      packages=find_packages(),
      install_requires=['gym', 'numpy']
      )
