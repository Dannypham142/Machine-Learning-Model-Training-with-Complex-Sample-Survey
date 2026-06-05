# Machine Learning Model Training with Complex Sample Survey 
Exploration on survey weights in machine learning model training. `data_pipeline` synthesizes 250,000 samples based off of ACS PUMs data through the use of Synthetic Data Vault's GaussianCopulaSynthesizer. `ml_pipeline` trains various models to test performance between baseline and incorporation of survey weights as a factor of model optimization.

This repository contains two de-coupled pipelines: `data_pipeline` and `ml_pipeline` with their respecitive Docker containerization and `README.md`.