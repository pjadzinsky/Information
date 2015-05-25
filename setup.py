try:
  from setuptools import setup
except:
  from distutils.core import setup

config = {
    'description': "Compute entropy, Shannon's information and several related quantities",
    'author': 'Pablo Jadzinsky',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it',
    'author_email': 'pjadzinsky@gmail.com',
    'version': '0.1',
    'install_requires': ['nose', 'numpy', 'scipy'],
    'packages': ['information', 'symbol_info'],
    'scripts': [],
    'name', 'Information'
    }

setup(**config)
