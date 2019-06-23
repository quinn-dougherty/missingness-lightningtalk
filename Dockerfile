FROM jupyter/scipy-notebook

# RUN pip install scikit-learn==0.21.2

RUN pip install fancyimpute category-encoders tqdm scikit-learn==0.21.2
