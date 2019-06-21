FROM jupyter/scipy-notebook

RUN pip install category-encoders fancyimpute tqdm
