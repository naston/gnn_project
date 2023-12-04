# gnn_project

# Navigating this project
All relevant code can be found under the `src` directory.  
`eval.py` - Small file containing functions for calculation of evaluation metrics.  
`models.py` - File defining the architecture of all models used in the project.  
`preprocess.py` - File defining functions related to data cleaning and formatting as well as data manipulation such as train/test splits.  
`results.py` - File containing definition of training procedures and pipelines.  

`train_model.py` - Deprecated utility function for training link prediction model.  
`plotting.py` - Deprecated utility file for visualization during initial EDA.  
`connect.py` - Deprecated method for connecting to MariaDB hosts to download datasets.  

`lm_*.py` - files prepended with `lm_` were used to fine-tune an LM for creation of node embeddings.
`parse_*.py` - files prepedned with `parse_` were used to gather text data (abstracts) for use in creation of LM embeddings.

# Running this project
Explore the `notebooks` directory for example usage of the framework in the project, or create a new one and use the functions defined in `src` as utility.

If you experience any difficulty running this project, please feel free to reach out and we will help. 
The poetry file is currently out of date but should provide a good foundation for most necessary installations.