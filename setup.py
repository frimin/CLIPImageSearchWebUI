import setuptools
import os
import platform

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

requires = []
with open('requirements.txt', encoding='utf8') as f:
    for x in f.readlines():
        requires.append(f'{x.strip()}')

def get_data_files(data_dir, prefix=''):
    file_dict = {}
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if prefix+root not in file_dict:
                file_dict[prefix+root] = []
            file_dict[prefix+root].append(os.path.join(root, name))
    return [(k, v) for k, v in file_dict.items()]

setuptools.setup(
    name="module",
    py_modules=["module"],
    version="0.0.1",
    author="frimin",
    author_email="hi@frimin.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/frimin/CLIPImageSearchWebUI",
    packages=setuptools.find_packages(),
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.10',

    data_files=[
        *get_data_files('data', prefix='module/'),
    ],

    install_requires=requires
)
