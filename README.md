# Winograd-Schema-Challenge
CSE 842 Final Project

In order to run this project, you first must run the following command to download the appropriate spacy files.

```python -m spacy download en_core_web_sm```

After downloading, you can run the project using

```python3 winograd.py```

Note: For the version reported in the paper, the en_core_web_lg model was used instead of the en_core_web_sm model. For ease of running this code, I have modified the code to use the small file as the large model is about 5 gigabytes and takes a long while to download. The small model also works for predicting the schema, just at a 3% lower degree of accuracy. 
