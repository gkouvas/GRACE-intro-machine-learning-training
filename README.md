 

# Introduction to Machine Learning (with python)

This repository regroups works on the Introduction to Machine Learning (with Python) coure from the [GRACE training program](https://www.unibas.ch/en/University/Administration-Services/Vice-President-s-Office-for-Education/Academic-Programs/Graduate-Center/Transferable-Skills.html) of the university of Basel.



## pre-requisites

The course is targeted to PhD students and postdoctoral researchers who are already familiar with the Python programming language and who have basic knowledge on statistics, and would like learn the basics of Machine Learning and how it can help teir research.


In order to follow the course you need to have installed [python](https://www.python.org/) and [jupyter notebooks](https://www.jupyter.org/) (a web based notebook system for creating and sharing computational documents). 

If you need help with the installation, you can refer to these [tips and instructions](https://github.com/sib-swiss/first-steps-with-python-training/blob/master/setting_up_your_environment.md)(NB: this links to another github repo).


In addition, there are a number of libraries to install.
See the [intructions on installing prerequisite libraries](env_setup.md).


## course organization 

The course is organized in several, numbered, jupyter notebooks, each corresponding to a chapter which interleaves theory, code demo, and exercises.

The course does not require any particular expertise with jupyter notebooks to be followed, but if it is the first time you encounter them we recommend this [gentle introduction](https://realpython.com/jupyter-notebook-introduction/).

 * [Chapter0 : python warmup](python_notebooks/Chapter_0_python_warmup.ipynb): provides a gentle warm-up to the basic usage of the libraries which are a pre-requisite for this course. You can use this notebook to assess your level before the course, or just as a way to get (re-)acquainted with these libraries.
 * [Chapter1 : Exploratory analysis](python_notebooks/Chapter_1_Exploratory_analysis.ipynb)
 * [Chapter2 : Unsupervised learning](python_notebooks/Chapter_2_unsupervised_learning.ipynb)
 * [Chapter3 : Supervised learning and Machine Learning routine ](python_notebooks/Chapter_3_supervised_learning_and_ML_pipeline.ipynb)
 * [Chapter4 : Distance-based model for classification](python_notebooks/Chapter_4_distance_based_model_for_classification.ipynb)
 * [Chapter5 : Decision trees for classification](python_notebooks/Chapter_5_Machine_Learning_based_on_decision_trees_for_classification.ipynb)
 * [Chapter6 : Machine Learning for regression](python_notebooks/Chapter_6_Machine_Learning_for_regression.ipynb)
 * [Chapter Extra : Teasing Neural Network](python_notebooks/Chapter_Extra_Teasing_Neural_Network.ipynb)

Solutions to each practical can be found in the [`solutions/`](python_notebooks/solutions/) folder and should be loadable directly in the jupyter notebook themselves.

Note also the `utils.py` and `utils2.py` files which contain many utilitary function we use to visually showcase the effect of various ML algorithms' hyper-parameters.



## directory structure


* data : contains the datasets
* exam : contains the data and instruction of a facultative exam
* images : images generated or used in the notebooks
* python_notebooks
* references : reference to the article of some of the datasets
* slides : pptx / pdf of introductory slides 


## course material citation

> This course has been adapted from the ["introduction to machine learning" course of the SIB](https://github.com/sib-swiss/intro-machine-learning-training) : Mueller, M., & Duchemin, W. (2023, October 25). Introduction to Machine Learning - SIB training. Zenodo. https://doi.org/10.5281/zenodo.10039454

