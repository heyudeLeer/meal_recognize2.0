from setuptools import setup
from setuptools import find_packages


setup(name='ImgSeg',
      version='1.0.0',
      description='image segment',
      author='he yude',
      author_email='heyude@meican.com',
      url='https://gitlab.planetmeican.com/nn/meal-recognize.git',
      license='MIT',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'six>=1.9.0',
                        'pyyaml',
			'keras'],
      extras_require={
          'h5py': ['h5py'],
      },
      packages=find_packages())
