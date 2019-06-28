from setuptools import setup, find_packages

setup(name='gridworldsgym',
      version='0.1',
      description='A set of RL grid world environments.',
      author='Miguel Alonso Jr.',
      install_requires=[
          'numpy',
          'gym',
          'visdom'
      ],
      packages=find_packages(),
      python_requires='>=3.6',
      include_package_data=True,
      zip_safe=False)
