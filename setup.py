import setuptools #导入setuptools打包工具
 
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
 
setuptools.setup(
    name="dataset_convert_toolkit", # 用自己的名替换其中的YOUR_USERNAME_
    version="0.0.2",    #包版本号，便于维护版本
    author="porter pan",    #作者，可以写自己的姓名
    author_email="porter.pan@outlook.com",    #作者联系方式，可写自己的邮箱地址
    description="一个简单的深度学习数据集转换工具",#包的简述
    long_description=long_description,    #包的详细介绍，一般在README.md文件内
    long_description_content_type="text/markdown",
    url="https://github.com/porterpan/dataset_convert_toolkit",    #自己项目地址，比如github的项目地址
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"":"."},
    install_requires=[
        'argparse',
        'numpy<=1.24.2',
        'opencv_python<=4.9.0.80',
        'PyYAML<=6.0.1',
        'tqdm<=4.66.4',
        'jsons<=1.6.3',
    ],
    enter_points={
        'console_scripts':[
            'dataset_convert_tool = dataset_convert_toolkit.dataset_convert_tool:main',
            'ua_detrac_convert_tool = dataset_convert_toolkit.ua_detrac_convert_tool:main',
            ],
        },
    python_requires='>=3.6',    #对python的最低版本要求
)