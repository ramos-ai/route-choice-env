from setuptools import setup, find_packages

setup(
      name='route-choice-env',
      version='0.0.1',
      description='Multi-Agent RL Route Choice Environment',
      url='https://github.com/ramos-ai/route-choice-gym',
      author='LuizAlfredoThomasini',
      author_email='luizthomasini@gmail.com',
      packages=find_packages(),
      install_requires=[
            'gymnasium',
            'matplotlib',
            'numpy',
            'pandas',
            'pettingzoo',
            'py_expression_eval',
            'pygame',
            'scipy',
            'sympy',
      ]
)
