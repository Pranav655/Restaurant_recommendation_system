# 🍴 Restaurant Recommendation & Review Prediction System

## 📌 Project Overview

This project is a **Restaurant Recommendation and Review Prediction System** designed to help users find suitable restaurants based on their preferences and predict restaurant review/rating outcomes.

The project has two main components:

1. **Restaurant Recommendation System**
2. **Restaurant Review Prediction System**

The system uses **Python, Pandas, Scikit-learn, NLP techniques, Machine Learning, and Streamlit**.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Restaurant Dataset │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Cleaning &     │
                    │ Preprocessing       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
      ┌─────────────────────┐     ┌─────────────────────┐
      │ Recommendation      │     │ Review Prediction   │
      │ Pipeline            │     │ Pipeline            │
      └──────────┬──────────┘     └──────────┬──────────┘
                 │                           │
                 ▼                           ▼
      ┌─────────────────────┐     ┌─────────────────────┐
      │ TF-IDF Vectorization │     │ Train/Test Split    │
      └──────────┬──────────┘     └──────────┬──────────┘
                 │                           │
                 ▼                           ▼
      ┌─────────────────────┐     ┌─────────────────────┐
      │ Cosine Similarity   │     │ ML Model Training   │
      └──────────┬──────────┘     └──────────┬──────────┘
                 │                           │
                 ▼                           ▼
      ┌─────────────────────┐     ┌─────────────────────┐
      │ Rating + Popularity │     │ Model Evaluation    │
      │ Score               │     └──────────┬──────────┘
      └──────────┬──────────┘                │
                 │                           ▼
                 ▼                 ┌─────────────────────┐
      ┌─────────────────────┐      │ Review Prediction  │
      │ Final Ranking       │      └─────────────────────┘
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │ Top N Restaurants   │
      └──────────┬──────────┘
                 │
                 └──────────────┐
                                ▼
                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    │                     │
                    │ User Preferences    │
                    │        ↓            │
                    │ Recommendations     │
                    │        ↓            │
                    │ Predictions         │
                    └─────────────────────┘
```

---

# 🎯 Project Scope

The main goal is to provide **relevant restaurant recommendations** instead of simply listing restaurants based only on their rating.

Users can specify:

* 📍 Location
* 🍛 Cuisine
* 💰 Maximum budget
* ⭐ Minimum rating
* 🔢 Number of recommendations

The system filters the available restaurants and ranks the suitable restaurants using multiple factors.

---

# 🔄 How the Recommendation System Works

**Restaurant Dataset → Data Cleaning → Feature Engineering → TF-IDF → Cosine Similarity → User Filtering → Ranking → Top Recommendations → Streamlit**

### 1. Data Collection

The dataset contains restaurant information such as:

* Restaurant name
* Location
* Cuisine
* Restaurant type
* Rating
* Votes/reviews
* Cost for two

### 2. Data Cleaning

Pandas is used to handle missing values, duplicates, incorrect data types, and inconsistent values.

### 3. Feature Engineering

Important textual attributes such as **location, cuisine, and restaurant type** are combined to represent the characteristics of each restaurant.

### 4. TF-IDF Vectorization

TF-IDF converts the textual restaurant information into **numerical vectors**, allowing restaurants to be mathematically compared.

### 5. Cosine Similarity

Cosine similarity measures the similarity between restaurant vectors.

A higher similarity score means the restaurants have more similar characteristics.

### 6. User Preference Filtering

The user's requirements are applied to filter restaurants based on:

**Location → Cuisine → Budget → Minimum Rating**

### 7. Final Ranking

Suitable restaurants are ranked using:

* **Cosine similarity** — similarity between restaurant characteristics
* **Rating score** — restaurant quality
* **Popularity score** — number of votes/reviews

These factors are combined into a **final recommendation score**.

### 8. Top Recommendations

The restaurants are sorted according to the final score and the requested number of restaurants is displayed.

---

# 🤖 Review Prediction

The review prediction component uses a traditional **supervised machine-learning workflow**:

**Review Data → Cleaning → Preprocessing → Feature Engineering → Train/Test Split → Model Training → Evaluation → Prediction**

Historical review/rating data is used to train a machine-learning model.

The dataset is divided into training and testing data. The model learns patterns from the training data and is then evaluated using unseen testing data.

---

# 🖥️ Streamlit Application

Streamlit provides the user interface.

The application follows:

**User Input → Restaurant Filtering → Recommendation Scoring → Ranking → Top Results → UI Display**

The UI displays:

* Restaurant name
* Rating
* Location
* Cuisine
* Cost for two
* Recommendation score

---

# 🧠 Techniques Used

### Recommendation

* Pandas
* Feature Engineering
* TF-IDF Vectorization
* Cosine Similarity
* Weighted Ranking
* Content-Based Recommendation

### Review Prediction

* Data preprocessing
* Feature engineering
* Train/Test Split
* Supervised Machine Learning
* Model Evaluation

---

# 🚀 Future Scope

The project can be extended with:

* Personalized recommendations based on user history
* Collaborative filtering
* Hybrid recommendation algorithms
* Review sentiment analysis
* Restaurant distance calculation
* Maps and restaurant locations
* Real-time restaurant information
* User ratings and feedback
* Personalized user profiles

---

# 📌 Project Summary

The project combines **data preprocessing, NLP, content-based recommendation, machine learning, and web deployment** to create a practical restaurant recommendation system.

The recommendation component identifies restaurants matching the user's requirements and ranks them using **similarity, rating, and popularity**.

The review prediction component uses historical data and **supervised machine learning** to make predictions.

Finally, **Streamlit provides a user-friendly interface** through which users can interact with the recommendation and prediction system.

