from setuptools import setup, find_packages

setup(
    name='rxnpredict',  
    version='0.1',  
    packages=find_packages("."),  
    description='Package for reaction prediction',
    author='Li-Cheng Xu', 
    author_email='licheng_xu@zju.edu.cn',  
    install_requires=[  
        
    ],
    extras_require={
        'conda': [
            
        ]
    },
    license="MIT",
    python_requires=">=3.10",
)