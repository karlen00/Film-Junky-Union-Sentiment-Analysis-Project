### README for Film Junky Union Sentiment Analysis Project

## Project Description

Film Junky Union, a new community for classic movie enthusiasts, is developing a system to filter and categorize movie reviews. The primary mission is to train a model to automatically detect negative reviews. Using the IMDB movie review dataset with polarity labels, then create a model capable of classifying positive and negative reviews. The model should achieve an F1 score of at least 0.85.

## Project Instructions

### 1. Data Loading
- Load the dataset from the file `imdb_reviews.tsv`.

### 2. Data Preprocessing
- Perform data preprocessing as necessary to clean and prepare the data for analysis and modeling.

### 3. Exploratory Data Analysis (EDA)
- Conduct an EDA to understand the data distribution and identify any class imbalances.
- Create visualizations and summarize findings.

### 4. Data Preprocessing for Modeling
- Preprocess the data to prepare it for model training.
- This may include tokenization, removing stop words, and converting text to numerical features using techniques like TF-IDF or word embeddings.

### 5. Model Training
- Train at least three different models on the training dataset.
- Suggested models include logistic regression and gradient boosting, but you may also try other methods.

### 6. Model Evaluation
- Test the models on the test dataset.
- Use the provided `evaluate_model()` function to assess model performance.
- Ensure that the model achieves an F1 score of at least 0.85.

### 7. Additional Testing
- Write several movie reviews yourself and classify them using all trained models.
- Compare the model predictions on your reviews with the test dataset results and explain any differences.

### 8. Present Findings
- Document your findings and display the results.
- Include visualizations and summaries of the model performances.

## Data Description

The dataset is stored in the file `imdb_reviews.tsv`. This dataset was obtained from Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

### Dataset Columns:
- **review**: The text of the movie review.
- **pos**: The target label, '0' for negative and '1' for positive reviews.
- **ds_part**: Indicates whether the review is part of the 'train' or 'test' dataset.

## Key Steps and Code Snippets

1. **EDA with Plots**: Perform EDA using various plots to understand data distribution and identify class imbalances.
2. **evaluate_model()**: A function to evaluate classification models adhering to the scikit-learn prediction interface.
3. **BERT_text_to_embeddings()**: A routine to convert a list of texts into embeddings using BERT.

### Notes:
- You are encouraged to use and modify the provided code snippets as needed.
- While BERT is not required due to computational demands, you may include it for additional analysis if desired. BERT typically requires GPU for reasonable performance. If used, apply it to a smaller subset of the dataset and indicate its use in the project.

## Conclusion

This project aims to develop and evaluate models for classifying movie reviews as positive or negative based on text analysis. The goal is to create a reliable system for Film Junky Union that can automatically detect negative reviews, aiding in better content filtering and categorization. The results will include model evaluations, comparisons, and recommendations based on the performance metrics, specifically aiming for an F1 score of at least 0.85.
