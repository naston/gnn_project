# gnn_project
1. `brew install mariadb`
2. `poetry install`
3. `poetry shell`

# Navigating this project
All relevant code can be found under the `src` directory.  
`eval.py` - Small file containing functions for calculation of evaluation metrics.  
`models.py` - File defining the architecture of all models used in the project.  
`preprocess.py` - File defining functions related to data cleaning and formatting as well as data manipulation such as train/test splits.  
`results.py` - File containing definition of training procedures and pipelines.  

`train_model.py` - Deprecated utility function for training link prediction model.  
`plotting.py` - Deprecated utility file for visualization during initial EDA.  
`connect.py` - Deprecated method for connecting to MariaDB hosts to download datasets.  

# Running this project
Explore the `notebooks` directory for example usage of the framework in the project, or create a new one and use the functions defined in `src` as utility.