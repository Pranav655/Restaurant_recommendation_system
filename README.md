# 🍴 Restaurant Recommendation & Review Prediction System

A machine learning and Natural Language Processing (NLP) based restaurant application that provides personalized restaurant recommendations and predicts the sentiment of customer reviews.

The project combines a **content-based recommendation system** with a **supervised machine learning based review prediction system**, and provides an interactive user interface using **Streamlit**.

---

# 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Project Objectives](#-project-objectives)
4. [Key Features](#-key-features)
5. [System Architecture](#-system-architecture)
6. [Dataset](#-dataset)
7. [Data Loading](#-data-loading)
8. [Data Cleaning](#-data-cleaning)
9. [Exploratory Data Analysis](#-exploratory-data-analysis)
10. [Feature Engineering](#-feature-engineering)
11. [Restaurant Recommendation System](#-restaurant-recommendation-system)
12. [TF-IDF Vectorization](#-tf-idf-vectorization)
13. [Cosine Similarity](#-cosine-similarity)
14. [User Preference Matching](#-user-preference-matching)
15. [Rating Score](#-rating-score)
16. [Popularity Score](#-popularity-score)
17. [Final Recommendation Score](#-final-recommendation-score)
18. [Review Prediction System](#-review-prediction-system)
19. [Review Preprocessing](#-review-preprocessing)
20. [Review Model Training](#-review-model-training)
21. [Train/Test Split](#-traintest-split)
22. [Model Evaluation](#-model-evaluation)
23. [Streamlit Application](#-streamlit-application)
24. [Project Structure](#-project-structure)
25. [Saved Model Files](#-saved-model-files)
26. [Technologies Used](#-technologies-used)
27. [Installation](#-installation)
28. [Running the Application](#-running-the-application)
29. [Complete Workflow](#-complete-workflow)
30. [Example](#-example)
31. [Advantages](#-advantages)
32. [Limitations](#-limitations)
33. [Future Improvements](#-future-improvements)
34. [Conclusion](#-conclusion)
35. [Author](#-author)

---

# 📌 Project Overview

The **Restaurant Recommendation & Review Prediction System** is designed to help users discover restaurants based on their preferences while also providing analysis of customer reviews.

The system consists of two major components:

### 1. Restaurant Recommendation

The recommendation component suggests restaurants based on:

- Location
- Cuisine
- Maximum budget
- Minimum rating
- Restaurant characteristics
- Content similarity
- Restaurant rating
- Restaurant popularity

The recommendation engine uses:

```text
Restaurant Information
        ↓
Feature Engineering
        ↓
Combined Text Features
        ↓
TF-IDF
        ↓
Numerical Restaurant Vectors
        ↓
Cosine Similarity
        ↓
Similarity Score
        ↓
Rating Score
        ↓
Popularity Score
        ↓
Final Recommendation Score
        ↓
Top N Restaurants
