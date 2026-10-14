import pandas as pd
import ast
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

DATASET_PATH = "datasets/clean_movies.csv"

print("Loading dataset...")

movies = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Total movies:", len(movies))
print("Columns:", list(movies.columns))


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "movie_id",
    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "crew"
]

missing_columns = [
    column for column in required_columns
    if column not in movies.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    print(missing_columns)
    exit()

print("\nAll required columns are available.")


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

text_columns = [
    "overview",
    "genres",
    "keywords",
    "cast",
    "crew"
]

for column in text_columns:
    movies[column] = movies[column].fillna("")


movies["title"] = movies["title"].fillna("Unknown")


# ============================================================
# 4. CONVERT TMDB LIST DATA INTO TEXT
# ============================================================

def convert_to_text(value):
    """
    Converts TMDB JSON/list formatted data into
    simple text suitable for recommendation.
    """

    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    try:
        data = ast.literal_eval(str(value))

        if isinstance(data, list):

            names = []

            for item in data:

                if isinstance(item, dict):

                    # Get name from TMDB dictionary
                    if "name" in item:
                        names.append(str(item["name"]))

                else:
                    names.append(str(item))

            return " ".join(names)

        return str(data)

    except:
        return str(value)


# ============================================================
# 5. PROCESS GENRES
# ============================================================

print("\nProcessing genres...")

movies["genres"] = movies["genres"].apply(convert_to_text)


# ============================================================
# 6. PROCESS KEYWORDS
# ============================================================

print("Processing keywords...")

movies["keywords"] = movies["keywords"].apply(convert_to_text)


# ============================================================
# 7. PROCESS CAST
# ============================================================

print("Processing cast...")

movies["cast"] = movies["cast"].apply(convert_to_text)


# ============================================================
# 8. PROCESS CREW
# ============================================================

print("Processing crew...")

movies["crew"] = movies["crew"].apply(convert_to_text)


# ============================================================
# 9. CLEAN OVERVIEW
# ============================================================

movies["overview"] = (
    movies["overview"]
    .astype(str)
    .str.lower()
)


# ============================================================
# 10. CREATE COMBINED FEATURES / TAGS
# ============================================================

print("\nCreating combined movie features...")

movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["cast"] + " " +
    movies["crew"]
)


# ============================================================
# 11. CLEAN TAGS
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


movies["tags"] = movies["tags"].apply(clean_text)


# ============================================================
# 12. REMOVE DUPLICATE MOVIES
# ============================================================

before_duplicates = len(movies)

movies.drop_duplicates(
    subset="movie_id",
    inplace=True
)

after_duplicates = len(movies)

print("\nDuplicate movies removed:",
      before_duplicates - after_duplicates)


# ============================================================
# 13. RESET INDEX
# ============================================================

movies.reset_index(drop=True, inplace=True)


# ============================================================
# 14. TF-IDF VECTORIZATION
# ============================================================

print("\nCreating TF-IDF vectors...")

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(movies["tags"])

print("TF-IDF matrix created successfully!")
print("Matrix shape:", tfidf_matrix.shape)


# ============================================================
# 15. COSINE SIMILARITY
# ============================================================

print("\nCalculating cosine similarity...")

cosine_sim = cosine_similarity(tfidf_matrix)

print("Cosine similarity calculated successfully!")


# ============================================================
# 16. CREATE MOVIE INDEX
# ============================================================

movie_indices = pd.Series(
    movies.index,
    index=movies["title"].str.lower()
).drop_duplicates()


# ============================================================
# 17. RECOMMENDATION FUNCTION
# ============================================================

def recommend_movies(movie_title, number_of_recommendations=10):

    movie_title = movie_title.strip().lower()

    # Check whether movie exists
    if movie_title not in movie_indices:
        return None

    movie_index = movie_indices[movie_title]

    # Get similarity scores
    similarity_scores = list(
        enumerate(cosine_sim[movie_index])
    )

    # Sort by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the input movie itself
    similarity_scores = similarity_scores[1:]

    # Select top recommendations
    top_movies = similarity_scores[
        :number_of_recommendations
    ]

    recommendations = []

    for index, score in top_movies:

        recommendations.append({
            "title": movies.iloc[index]["title"],
            "similarity": round(float(score) * 100, 2)
        })

    return recommendations


# ============================================================
# 18. DISPLAY RECOMMENDATIONS
# ============================================================

def display_recommendations(movie_title):

    print("\n" + "=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)

    print("\nInput Movie:", movie_title)

    recommendations = recommend_movies(movie_title, 10)

    if recommendations is None:

        print("\nMovie not found in the dataset.")
        print("Please enter another movie name.")

        return

    print("\nRecommended Movies:")
    print("-" * 60)

    for number, movie in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{number}. {movie['title']} "
            f"({movie['similarity']}% similarity)"
        )


# ============================================================
# 19. DATASET TESTING
# ============================================================

print("\n")
print("=" * 60)
print("DATASET TESTING")
print("=" * 60)


# Test 1: Dataset loading
print("\nTEST CASE 1 - Dataset Loading")

if len(movies) > 0:
    print("PASS - Dataset loaded successfully.")
else:
    print("FAIL - Dataset is empty.")


# Test 2: Required columns
print("\nTEST CASE 2 - Required Columns")

if not missing_columns:
    print("PASS - All required columns are available.")
else:
    print("FAIL - Required columns are missing.")


# Test 3: Missing movie IDs
print("\nTEST CASE 3 - Missing Movie IDs")

missing_movie_ids = movies["movie_id"].isnull().sum()

if missing_movie_ids == 0:
    print("PASS - No missing movie IDs.")
else:
    print(
        "FAIL - Missing movie IDs:",
        missing_movie_ids
    )


# Test 4: Missing titles
print("\nTEST CASE 4 - Missing Movie Titles")

missing_titles = (
    movies["title"]
    .isnull()
    .sum()
)

if missing_titles == 0:
    print("PASS - No missing movie titles.")
else:
    print(
        "FAIL - Missing movie titles:",
        missing_titles
    )


# Test 5: Duplicate movie IDs
print("\nTEST CASE 5 - Duplicate Movie IDs")

duplicate_ids = movies["movie_id"].duplicated().sum()

if duplicate_ids == 0:
    print("PASS - No duplicate movie IDs.")
else:
    print(
        "FAIL - Duplicate movie IDs:",
        duplicate_ids
    )


# Test 6: Movie search
print("\nTEST CASE 6 - Movie Search")

test_movie = "Avatar"

if test_movie.lower() in movie_indices:
    print(
        f"PASS - '{test_movie}' found in dataset."
    )
else:
    print(
        f"FAIL - '{test_movie}' not found."
    )


# ============================================================
# 20. INTERACTIVE TESTING
# ============================================================

print("\n")
print("=" * 60)
print("RECOMMENDATION TESTING")
print("=" * 60)

print("\nEnter a movie name to get recommendations.")
print("Type 'exit' to stop.")


while True:

    user_input = input("\nEnter movie name: ").strip()

    if user_input.lower() == "exit":
        print("\nProgram ended.")
        break

    if user_input == "":
        print("Please enter a movie name.")
        continue

    display_recommendations(user_input)