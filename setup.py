from setuptools import setup, find_packages


with open("imagemap/__init__.py") as f:
    for line in f:
        if "__version__" in line:
            version = line.split("=")[1].strip().strip('"').strip("'")
            continue

setup(
    name='imagemap',
    version=version,
    url="git@gitlab.com/njanakiev/imagemap.git",
    author='Nikolai Janakiev',
    author_email='nikolai.janakiev@gmail.com',
    description='imagemap',
    platforms='any',
    packages=find_packages(),
    install_requires=[
        'mercantile',
        'numpy',
        'pandas',
        'Pillow'
    ],
    entry_points={
        "console_scripts": [
            "imagemap = imagemap.__main__:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)
