import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TasteMatch | Smart Restaurant Recommendations",
    page_icon="🍽️",
    layout="wide"
)

# ============================================================
# LOAD RESTAURANT DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_pickle(
        "restaurant_data.pkl"
    )


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

@st.cache_resource
def load_tfidf_vectorizer():

    return joblib.load(
        "tfidf_vectorizer.pkl"
    )


# ============================================================
# LOAD TF-IDF MATRIX
# ============================================================

@st.cache_resource
def load_tfidf_matrix():

    return joblib.load(
        "tfidf_matrix.pkl"
    )


# ============================================================
# LOAD FILES
# ============================================================

try:

    df = load_data()

    tfidf_vectorizer = load_tfidf_vectorizer()

    tfidf_matrix = load_tfidf_matrix()

except Exception as e:

    st.error(
        "Could not load recommendation files."
    )

    st.exception(e)

    st.stop()


# ============================================================
# PREPARE DATA
# ============================================================

df = df.reset_index(
    drop=True
)


# Convert numeric columns

df["rate"] = pd.to_numeric(
    df["rate"],
    errors="coerce"
)

df["votes"] = pd.to_numeric(
    df["votes"],
    errors="coerce"
).fillna(0)

df["cost_for_two"] = pd.to_numeric(
    df["cost_for_two"],
    errors="coerce"
)


# ============================================================
# CHECK TF-IDF MATRIX SIZE
# ============================================================

if len(df) != tfidf_matrix.shape[0]:

    st.error(
        "Restaurant data and TF-IDF matrix "
        "have different numbers of restaurants."
    )

    st.write(
        "Restaurants:",
        len(df)
    )

    st.write(
        "TF-IDF rows:",
        tfidf_matrix.shape[0]
    )

    st.stop()


# ============================================================
# CREATE COMBINED TEXT FOR USER VECTOR
# ============================================================

def create_user_text(
    location="",
    cuisine=""
):

    parts = []

    if location:

        parts.append(
            str(location)
        )

    if cuisine:

        parts.append(
            str(cuisine)
        )

    return " ".join(parts)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_restaurants(
    location="",
    cuisine="",
    max_budget=None,
    min_rating=0,
    top_n=10
):

    # --------------------------------------------------------
    # Start with complete dataset
    # --------------------------------------------------------

    result = df.copy()


    # ========================================================
    # 1. LOCATION FILTER
    # ========================================================

    if location:

        result = result[
            result["location"]
            .astype(str)
            .str.contains(
                location,
                case=False,
                na=False
            )
        ]


    # ========================================================
    # 2. CUISINE FILTER
    # ========================================================

    if cuisine:

        result = result[
            result["cuisines"]
            .astype(str)
            .str.contains(
                cuisine,
                case=False,
                na=False
            )
        ]


    # ========================================================
    # 3. BUDGET FILTER
    # ========================================================

    if max_budget is not None:

        result = result[
            result["cost_for_two"]
            <= max_budget
        ]


    # ========================================================
    # 4. MINIMUM RATING FILTER
    # ========================================================

    result = result[
        result["rate"]
        >= min_rating
    ]


    # ========================================================
    # CHECK FILTERED RESULTS
    # ========================================================

    if result.empty:

        return pd.DataFrame()


    # ========================================================
    # 5. CREATE USER PREFERENCE VECTOR
    # ========================================================

    user_text = create_user_text(
        location=location,
        cuisine=cuisine
    )


    # --------------------------------------------------------
    # If user entered location/cuisine
    # --------------------------------------------------------

    if user_text.strip():

        user_vector = tfidf_vectorizer.transform(
            [user_text]
        )


        # ====================================================
        # 6. COSINE SIMILARITY
        # ====================================================

        all_similarity = cosine_similarity(
            user_vector,
            tfidf_matrix
        ).flatten()


        # Assign similarity score to restaurants

        result["similarity_score"] = [
            all_similarity[i]
            for i in result.index
        ]


    else:

        # If no text preference was entered,
        # similarity cannot be calculated.

        result["similarity_score"] = 0


    # ========================================================
    # 7. RATING SCORE
    # ========================================================

    result["rating_score"] = (
        result["rate"] / 5
    )

    result["rating_score"] = (
        result["rating_score"]
        .clip(0, 1)
        .fillna(0)
    )


    # ========================================================
    # 8. POPULARITY SCORE
    # ========================================================

    result["popularity_raw"] = np.log1p(
        result["votes"]
    )


    popularity_min = (
        result["popularity_raw"].min()
    )

    popularity_max = (
        result["popularity_raw"].max()
    )


    if popularity_max > popularity_min:

        result["popularity_score"] = (

            result["popularity_raw"]
            - popularity_min

        ) / (

            popularity_max
            - popularity_min
        )

    else:

        result["popularity_score"] = 0


    # ========================================================
    # 9. FINAL HYBRID SCORE
    # ========================================================

    result["final_score"] = (

        0.50
        * result["similarity_score"]

        +

        0.30
        * result["rating_score"]

        +

        0.20
        * result["popularity_score"]
    )


    # ========================================================
    # 10. SORT
    # ========================================================

    result = result.sort_values(
        by="final_score",
        ascending=False
    )


    # ========================================================
    # 11. TOP N
    # ========================================================

    return result.head(
        top_n
    ).reset_index(
        drop=True
    )


# ============================================================
# TITLE
# ============================================================

st.title("🍽️ TasteMatch")

st.markdown(
    "### Find Your Next Favorite Restaurant"
)

st.write(
    "Personalized restaurant recommendations based on your "
    "location, cuisine, budget, and rating preferences."
)
st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🔍 Your Preferences"
)


# ============================================================
# LOCATION
# ============================================================

locations = sorted(
    df["location"]
    .dropna()
    .astype(str)
    .unique()
)

location = st.sidebar.selectbox(
    "📍 Location",
    ["Any Location"] + locations
)


# ============================================================
# CUISINE
# ============================================================

cuisine = st.sidebar.text_input(
    "🍛 Cuisine",
    placeholder="Example: North Indian"
)


# ============================================================
# BUDGET
# ============================================================

budget = st.sidebar.number_input(
    "💰 Maximum Cost for Two",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100
)


# ============================================================
# MINIMUM RATING
# ============================================================

rating = st.sidebar.slider(
    "⭐ Minimum Rating",
    min_value=0.0,
    max_value=5.0,
    value=4.0,
    step=0.1
)


# ============================================================
# NUMBER OF RECOMMENDATIONS
# ============================================================

top_n = st.sidebar.slider(
    "Number of Restaurants",
    min_value=5,
    max_value=20,
    value=10
)


# ============================================================
# BUTTON
# ============================================================

recommend = st.sidebar.button(
    "🔍 Find Restaurants",
    type="primary"
)


# ============================================================
# RUN RECOMMENDATION
# ============================================================

if recommend:

    selected_location = (

        ""

        if location == "Any Location"

        else location
    )


    result = recommend_restaurants(

        location=selected_location,

        cuisine=cuisine.strip(),

        max_budget=budget,

        min_rating=rating,

        top_n=top_n
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    if result.empty:

        st.warning(
            "😕 No restaurants found for your preferences."
        )

        st.info(
            "Try increasing the budget or "
            "reducing the minimum rating."
        )


    else:

        st.success(
            f"🎉 Found {len(result)} recommended restaurants!"
        )


        st.subheader(
            "🍽 Recommended Restaurants"
        )


        for rank, (_, restaurant) in enumerate(
            result.iterrows(),
            start=1
        ):

            with st.container():

                # ------------------------------------------------
                # Restaurant name
                # ------------------------------------------------

                st.markdown(
                    f"## #{rank} 🍴 {restaurant['name']}"
                )


                # ------------------------------------------------
                # Basic information
                # ------------------------------------------------

                col1, col2, col3 = st.columns(3)


                with col1:

                    st.write(
                        f"⭐ **Rating:** "
                        f"{restaurant['rate']:.1f}"
                    )


                with col2:

                    st.write(
                        f"📍 **Location:** "
                        f"{restaurant['location']}"
                    )


                with col3:

                    st.write(
                        f"💰 **Cost for Two:** "
                        f"₹{restaurant['cost_for_two']}"
                    )


                # ------------------------------------------------
                # Cuisine
                # ------------------------------------------------

                st.write(
                    f"🍛 **Cuisines:** "
                    f"{restaurant['cuisines']}"
                )


                # =================================================
                # SCORE INFORMATION
                # =================================================

                st.markdown(
                    "### 📊 Recommendation Scores"
                )


                score_col1, score_col2, score_col3, score_col4 = (
                    st.columns(4)
                )


                with score_col1:

                    st.metric(
                        "Similarity",
                        f"{restaurant['similarity_score']:.3f}"
                    )


                with score_col2:

                    st.metric(
                        "Rating Score",
                        f"{restaurant['rating_score']:.3f}"
                    )


                with score_col3:

                    st.metric(
                        "Popularity Score",
                        f"{restaurant['popularity_score']:.3f}"
                    )


                with score_col4:

                    st.metric(
                        "Final Score",
                        f"{restaurant['final_score']:.3f}"
                    )


                st.caption(
                    "Final Score = "
                    "80% Similarity + "
                    "10% Rating + "
                    "10% Popularity"
                )


                st.divider()