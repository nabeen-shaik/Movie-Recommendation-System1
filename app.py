from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import ast
import json
import re
import sqlite3

import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "movie-recommender-secret-key-2026"
)

ADMIN_USERNAME = os.environ.get(
    "ADMIN_USERNAME",
    "admin"
)

ADMIN_PASSWORD = os.environ.get(
    "ADMIN_PASSWORD",
    "admin123"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "movie_recommender.db"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "clean_movies_with_posters.csv"
)


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():

    conn = get_db_connection()

    # USERS
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # DISCUSSIONS
    conn.execute("""
        CREATE TABLE IF NOT EXISTS discussions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_name TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # DISCUSSION REPLIES
    conn.execute("""
        CREATE TABLE IF NOT EXISTS discussion_replies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            discussion_id INTEGER NOT NULL,
            user_id INTEGER,
            user_name TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # STUDENT MODE TABLES
    conn.execute("""
        CREATE TABLE IF NOT EXISTS student_bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, movie_title)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS student_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_title TEXT NOT NULL,
            note TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS student_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            completed INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, topic)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_title TEXT,
            score INTEGER NOT NULL,
            total INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

    print("SQLite database ready!")


init_database()


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"""
Dataset not found:

{DATASET_PATH}

Please place your CSV here:

datasets/clean_movies_with_posters.csv
"""
    )


movies = pd.read_csv(DATASET_PATH)

print("==========================================")
print("Dataset loaded successfully!")
print("Total movies:", len(movies))
print("Columns:", movies.columns.tolist())
print("==========================================")


# ============================================================
# COLUMN DETECTION
# ============================================================

possible_poster_columns = [
    "poster_url",
    "poster_path",
    "poster",
    "Poster",
    "Poster_Path",
    "Poster_URL",
    "posterUrl",
    "posterPath"
]

poster_column = next(
    (
        column
        for column in possible_poster_columns
        if column in movies.columns
    ),
    None
)


possible_rating_columns = [
    "vote_average",
    "rating",
    "Rating",
    "ratings",
    "score",
    "Score"
]

rating_column = next(
    (
        column
        for column in possible_rating_columns
        if column in movies.columns
    ),
    None
)

if rating_column is None:

    rating_column = "vote_average"

    movies[rating_column] = ""


possible_year_columns = [
    "release_date",
    "release_year",
    "year",
    "Year",
    "releaseYear",
    "release_years"
]

year_column = next(
    (
        column
        for column in possible_year_columns
        if column in movies.columns
    ),
    None
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "overview",
    "genres",
    "keywords",
    "cast",
    "crew"
]

for column in required_columns:

    if column not in movies.columns:
        movies[column] = ""

    movies[column] = (
        movies[column]
        .fillna("")
        .astype(str)
    )


if "title" not in movies.columns:

    raise ValueError(
        "ERROR: 'title' column not found in dataset."
    )


movies["title"] = (
    movies["title"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)


# ============================================================
# SAVE ORIGINAL CAST / CREW
# ============================================================

movies["_original_cast"] = movies["cast"].copy()
movies["_original_crew"] = movies["crew"].copy()


# ============================================================
# POSTER URL
# ============================================================

def make_poster_url(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    value = str(value).strip()

    if not value:
        return ""

    if value.lower() == "nan":
        return ""

    if value.startswith(("http://", "https://")):
        return value

    if value.startswith("/"):
        return (
            "https://image.tmdb.org/t/p/w500"
            + value
        )

    if value.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    ):
        return (
            "https://image.tmdb.org/t/p/w500/"
            + value
        )

    return ""


def get_poster_from_row(row):

    if poster_column is None:
        return ""

    try:

        value = row.get(
            poster_column,
            ""
        )

        return make_poster_url(value)

    except Exception:

        return ""


# ============================================================
# RATING
# ============================================================

def get_rating_from_row(row):

    try:

        value = row.get(
            rating_column,
            ""
        )

    except Exception:

        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:

        pass

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return ""

    try:

        number = float(value)

        if number.is_integer():
            return str(int(number))

        return str(round(number, 1))

    except Exception:

        return value


# ============================================================
# PARSE LIST
# ============================================================

def parse_list(value):

    if value is None:
        return []

    try:

        if pd.isna(value):
            return []

    except Exception:

        pass

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return []

    try:

        data = ast.literal_eval(value)

        if isinstance(data, list):
            return data

    except Exception:

        pass

    try:

        data = json.loads(value)

        if isinstance(data, list):
            return data

    except Exception:

        pass

    return []


# ============================================================
# EXTRACT NAMES
# ============================================================

def extract_names(value, limit=5):

    data = parse_list(value)

    names = []

    if data:

        for item in data:

            if isinstance(item, dict):

                name = item.get(
                    "name",
                    ""
                )

            else:

                name = str(item)

            name = str(name).strip()

            if name and name not in names:

                names.append(name)

            if len(names) >= limit:
                break

        return names

    text = str(value).strip()

    if not text:
        return []

    return [
        item.strip()
        for item in text.split(",")[:limit]
        if item.strip()
    ]


# ============================================================
# CAST NAME EXTRACTION
# ============================================================

def get_cast_names(row, limit=10):

    value = row.get(
        "_original_cast",
        ""
    )

    text = str(value).strip()

    if not text:
        return []

    data = parse_list(text)

    if data:

        return extract_names(
            text,
            limit
        )

    known_names = [

        "Sam Worthington",
        "Zoe Saldana",
        "Sigourney Weaver",
        "Stephen Lang",
        "Michelle Rodriguez",
        "Giovanni Ribisi",
        "Joel David Moore",
        "CCH Pounder",
        "Wes Studi",
        "Laz Alonso",
        "Dileep Rao",
        "Matt Gerald",

        "Anne Hathaway",
        "Christian Bale",
        "Leonardo DiCaprio",
        "Joseph Gordon-Levitt",
        "Ellen Page",
        "Tom Hardy",
        "Morgan Freeman",
        "Michael Caine",

        "Scarlett Johansson",
        "Robert Downey Jr.",
        "Chris Evans",
        "Mark Ruffalo",
        "Chris Hemsworth",
        "Jeremy Renner",
        "Tom Hiddleston",
        "Natalie Portman",

        "Gal Gadot",
        "Henry Cavill",
        "Ben Affleck",
        "Amy Adams",
        "Jesse Eisenberg",

        "Margot Robbie",
        "Ryan Gosling",
        "Emma Stone",
        "Jennifer Lawrence",

        "Brad Pitt",
        "Angelina Jolie",
        "Johnny Depp",
        "Keira Knightley",
        "Orlando Bloom",

        "Daniel Radcliffe",
        "Emma Watson",
        "Rupert Grint",
        "Robert Pattinson",
        "Kristen Stewart",

        "Dwayne Johnson",
        "Jason Statham",
        "Vin Diesel",
        "Paul Walker",

        "Will Smith",
        "Jaden Smith",

        "Jackie Chan",
        "Jet Li",
        "Bruce Willis",
        "Sylvester Stallone",
        "Arnold Schwarzenegger",

        "Jim Carrey",
        "Adam Sandler",
        "Robin Williams",

        "Meryl Streep",
        "Julia Roberts",
        "Sandra Bullock",
        "Charlize Theron",
        "Cate Blanchett",

        "Saoirse Ronan",
        "Florence Pugh",
        "Zendaya",
        "Elizabeth Olsen",
        "Brie Larson",
        "Karen Gillan",

        "Rachel McAdams",
        "Jessica Chastain",
        "Mila Kunis",
        "Kristen Wiig",

        "Emily Blunt",
        "Emily Mortimer",
        "Naomie Harris",
        "Eva Green",
        "Lea Seydoux",
        "Monica Bellucci"
    ]

    found = []

    lower_text = text.lower()

    for name in known_names:

        if name.lower() in lower_text:

            if name not in found:
                found.append(name)

        if len(found) >= limit:
            break

    return found


# ============================================================
# ACTRESS
# ============================================================

def get_actresses_from_row(row):

    cast_names = get_cast_names(
        row,
        limit=15
    )

    female_names = {

        "Zoe Saldana",
        "Sigourney Weaver",
        "Michelle Rodriguez",
        "Anne Hathaway",
        "Ellen Page",
        "Scarlett Johansson",
        "Natalie Portman",
        "Gal Gadot",
        "Amy Adams",
        "Margot Robbie",
        "Emma Stone",
        "Jennifer Lawrence",
        "Angelina Jolie",
        "Keira Knightley",
        "Emma Watson",
        "Kristen Stewart",
        "Meryl Streep",
        "Julia Roberts",
        "Sandra Bullock",
        "Charlize Theron",
        "Cate Blanchett",
        "Saoirse Ronan",
        "Florence Pugh",
        "Zendaya",
        "Elizabeth Olsen",
        "Brie Larson",
        "Karen Gillan",
        "Rachel McAdams",
        "Jessica Chastain",
        "Mila Kunis",
        "Emily Blunt",
        "Naomie Harris",
        "Eva Green",
        "Lea Seydoux",
        "Monica Bellucci"
    }

    actresses = []

    for name in cast_names:

        if name in female_names:
            actresses.append(name)

        if len(actresses) >= 2:
            break

    return actresses


# ============================================================
# ACTORS
# ============================================================

def get_actors_from_row(row):

    cast_names = get_cast_names(
        row,
        limit=10
    )

    actresses = set(
        get_actresses_from_row(row)
    )

    actors = []

    for name in cast_names:

        if name not in actresses:
            actors.append(name)

        if len(actors) >= 3:
            break

    return actors


# ============================================================
# DIRECTOR
# ============================================================
# Director is intentionally NOT displayed.
# We keep this function only for compatibility.
# ============================================================

def get_director_from_row(row):

    return ""


# ============================================================
# YEAR
# ============================================================

def get_year_from_row(row):

    columns = []

    if year_column:
        columns.append(year_column)

    columns.extend([
        "release_date",
        "release_year",
        "year",
        "Year"
    ])

    for column in columns:

        if column not in movies.columns:
            continue

        try:

            value = row.get(
                column,
                ""
            )

        except Exception:

            continue

        try:

            if pd.isna(value):
                continue

        except Exception:

            pass

        match = re.search(
            r"(19|20)\d{2}",
            str(value)
        )

        if match:
            return match.group(0)

    return ""


# ============================================================
# GENRES
# ============================================================

def format_genres(value):

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:

        pass

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return ""

    data = parse_list(text)

    if data:

        names = []

        for item in data:

            if isinstance(item, dict):

                name = item.get(
                    "name",
                    ""
                )

            else:

                name = str(item)

            name = str(name).strip()

            if name and name not in names:
                names.append(name)

        if names:
            return ", ".join(names)

    return text


# ============================================================
# CONVERT DATA TO TEXT
# ============================================================

def convert_to_text(value):

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:

        pass

    text = str(value).strip()

    if not text:
        return ""

    try:

        data = ast.literal_eval(text)

        if isinstance(data, list):

            names = []

            for item in data:

                if isinstance(item, dict):

                    if item.get("name"):

                        names.append(
                            str(item["name"])
                        )

                else:

                    names.append(
                        str(item)
                    )

            return " ".join(names)

        return str(data)

    except Exception:

        return text


# ============================================================
# PREPARE TEXT COLUMNS
# ============================================================

for column in [
    "genres",
    "keywords",
    "cast",
    "crew"
]:

    movies[column] = (
        movies[column]
        .apply(convert_to_text)
    )


# ============================================================
# RECOMMENDATION MODEL
# ============================================================

tfidf = None
tfidf_matrix = None
movie_indices = None


def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def rebuild_recommendation_model():

    global tfidf
    global tfidf_matrix
    global movie_indices

    for column in [
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew"
    ]:

        if column not in movies.columns:
            movies[column] = ""

        movies[column] = (
            movies[column]
            .fillna("")
            .astype(str)
        )

    movies["tags"] = (
        movies["overview"]
        + " "
        + movies["genres"]
        + " "
        + movies["keywords"]
        + " "
        + movies["cast"]
        + " "
        + movies["crew"]
    )

    movies["tags"] = (
        movies["tags"]
        .apply(clean_text)
    )

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    tfidf_matrix = tfidf.fit_transform(
        movies["tags"]
    )

    movie_indices = pd.Series(
        movies.index,
        index=(
            movies["title"]
            .astype(str)
            .str.lower()
            .str.strip()
        )
    ).drop_duplicates()

    print("==========================================")
    print("TF-IDF recommendation model ready!")
    print("Movies:", len(movies))
    print("==========================================")


rebuild_recommendation_model()


# ============================================================
# RECOMMEND MOVIES
# ============================================================

def recommend_movies(
    movie_title,
    number_of_recommendations=10
):

    movie_title = (
        movie_title
        .strip()
        .lower()
    )

    if movie_title not in movie_indices:
        return None

    movie_index = movie_indices[
        movie_title
    ]

    similarity_scores = (
        cosine_similarity(
            tfidf_matrix[movie_index],
            tfidf_matrix
        ).flatten()
    )

    ranked_movies = sorted(
        enumerate(similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in ranked_movies:

        if index == movie_index:
            continue

        row = movies.iloc[index]

        recommendations.append({

            "title":
                str(row["title"]),

            "similarity":
                round(
                    float(score) * 100,
                    2
                ),

            "poster_url":
                get_poster_from_row(row),

            "rating":
                get_rating_from_row(row),

            "year":
                get_year_from_row(row),

            "genres":
                format_genres(
                    row.get(
                        "genres",
                        ""
                    )
                )
        })

        if len(recommendations) >= number_of_recommendations:
            break

    return recommendations


# ============================================================
# SELECTED MOVIE INFORMATION
# ============================================================

def get_movie_info(movie_title):

    movie_title = (
        movie_title
        .strip()
        .lower()
    )

    if movie_title not in movie_indices:
        return None

    index = movie_indices[
        movie_title
    ]

    row = movies.iloc[index]

    overview = row.get(
        "overview",
        ""
    )

    try:

        if pd.isna(overview):
            overview = ""

    except Exception:

        overview = ""

    actresses = get_actresses_from_row(row)
    actors = get_actors_from_row(row)

    return {

        "title":
            str(
                row.get(
                    "title",
                    "Unknown"
                )
            ),

        "poster_url":
            get_poster_from_row(row),

        "rating":
            get_rating_from_row(row),

        "year":
            get_year_from_row(row),

        "genres":
            format_genres(
                row.get(
                    "genres",
                    ""
                )
            ),

        "actors":
            actors,

        "actress":
            actresses,

        "cast":
            actors,

        "overview":
            str(overview).strip()
    }


# ============================================================
# MOVIE SUGGESTIONS
# ============================================================

@app.route("/movie-suggestions")
def movie_suggestions():

    query = request.args.get(
        "q",
        ""
    ).strip().lower()

    if not query:

        return jsonify({
            "suggestions": []
        })

    matches = movies[
        movies["title"]
        .astype(str)
        .str.lower()
        .str.contains(
            query,
            regex=False,
            na=False
        )
    ]

    suggestions = []

    for _, row in matches.head(8).iterrows():

        suggestions.append({

            "title":
                str(row["title"]),

            "poster_url":
                get_poster_from_row(row),

            "rating":
                get_rating_from_row(row),

            "year":
                get_year_from_row(row),

            "genres":
                format_genres(
                    row.get(
                        "genres",
                        ""
                    )
                )
        })

    return jsonify({
        "suggestions": suggestions
    })


# ============================================================
# TRENDING MOVIES
# ============================================================

TRENDING_TITLES = [
    "Interstellar",
    "Inception",
    "The Dark Knight",
    "The Matrix",
    "Avengers: Endgame",
    "The Lord of the Rings"
]


def get_trending_movies():

    trending = []

    for title in TRENDING_TITLES:

        matches = movies[
            movies["title"]
            .astype(str)
            .str.lower()
            .str.strip()
            == title.lower()
        ]

        if len(matches) == 0:
            continue

        row = matches.iloc[0]

        trending.append({

            "title":
                str(row["title"]),

            "poster_url":
                get_poster_from_row(row),

            "rating":
                get_rating_from_row(row),

            "year":
                get_year_from_row(row)
        })

    return trending


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    recommendations = []
    selected_movie = None
    movie_name = ""
    error = ""

    trending_movies = get_trending_movies()

    if request.method == "POST":

        movie_name = request.form.get(
            "movie_name",
            ""
        ).strip()

        if not movie_name:

            error = (
                "Please enter a movie name."
            )

        else:

            selected_movie = get_movie_info(
                movie_name
            )

            if selected_movie is None:

                error = (
                    "Movie not found in the "
                    "dataset. Please enter another "
                    "movie name."
                )

            else:

                recommendations = recommend_movies(
                    movie_name,
                    10
                )

    return render_template(
        "index.html",
        recommendations=recommendations,
        selected_movie=selected_movie,
        movie_name=movie_name,
        error=error,
        trending_movies=trending_movies,
        user_logged_in=session.get(
            "user_logged_in",
            False
        ),
        user_name=session.get(
            "user_name",
            ""
        )
    )


# ============================================================
# MOVIES PAGE
# ============================================================

@app.route("/movies")
def movies_page():

    search = request.args.get(
        "search",
        ""
    ).strip()

    genre = request.args.get(
        "genre",
        ""
    ).strip()

    page = request.args.get(
        "page",
        1,
        type=int
    )

    page = max(page, 1)

    per_page = 24

    filtered = movies.copy()

    if search:

        filtered = filtered[
            filtered["title"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
        ]

    if genre:

        filtered = filtered[
            filtered["genres"]
            .astype(str)
            .str.contains(
                genre,
                case=False,
                na=False,
                regex=False
            )
        ]

    total_movies = len(filtered)

    total_pages = max(
        1,
        (total_movies + per_page - 1)
        // per_page
    )

    page = min(
        page,
        total_pages
    )

    start = (
        page - 1
    ) * per_page

    end = start + per_page

    page_movies = filtered.iloc[
        start:end
    ]

    movie_list = []

    for _, row in page_movies.iterrows():

        movie_list.append({

            "title":
                str(row["title"]),

            "poster_url":
                get_poster_from_row(row),

            "rating":
                get_rating_from_row(row),

            "year":
                get_year_from_row(row),

            "genres":
                format_genres(
                    row.get(
                        "genres",
                        ""
                    )
                )
        })

    return render_template(
        "movies.html",
        movies=movie_list,
        search=search,
        genre=genre,
        page=page,
        total_pages=total_pages,
        total_movies=total_movies,
        user_logged_in=session.get(
            "user_logged_in",
            False
        ),
        user_name=session.get(
            "user_name",
            ""
        )
    )


# # ============================================================
# AI ASSISTANT
# ============================================================

@app.route(
    "/ai-assistant",
    methods=["GET", "POST"]
)
def ai_assistant():

    answer = ""
    recommendations = []
    movie_name = session.get("ai_movie_name", "")
    question = session.get("ai_question", "")
    error = ""
    movie = None

    if request.method == "POST":

        movie_name = request.form.get("movie_name", "").strip()
        question = request.form.get("question", "").strip()

        session["ai_movie_name"] = movie_name
        session["ai_question"] = question

        if not movie_name:
            error = "Please enter a movie name."

        elif not question:
            error = "Please select a learning topic."

        else:
            movie_info = get_movie_info(movie_name)

            if movie_info is None:
                matches = movies[
                    movies["title"].astype(str).str.lower().str.contains(
                        movie_name.lower(),
                        regex=False,
                        na=False
                    )
                ]

                if len(matches) > 0:
                    found_title = str(matches.iloc[0]["title"])
                    movie_info = get_movie_info(found_title)
                    movie_name = found_title
                    session["ai_movie_name"] = movie_name

            if movie_info is None:
                error = (
                    "Movie not found in the dataset. "
                    "Please enter a valid movie name."
                )

            else:
                movie = dict(movie_info)

                movie["title"] = str(
                    movie_info.get("title", movie_name)
                )

                movie["poster_url"] = (
                    movie_info.get("poster_url")
                    or get_poster_from_row(movie_info)
                )

                movie["rating"] = (
                    movie_info.get("rating")
                    or get_rating_from_row(movie_info)
                )

                movie["year"] = (
                    movie_info.get("year")
                    or get_year_from_row(movie_info)
                )

                movie["genres"] = format_genres(
                    movie_info.get("genres", "")
                )

                movie["overview"] = str(
                    movie_info.get("overview", "") or ""
                )

                title = movie["title"]
                title_lower = title.lower()
                genres = movie["genres"]
                overview = movie["overview"]
                lower_question = question.lower().strip()

                # SCIENCE
                if (
                    "scientific concepts" in lower_question
                    or "science concepts" in lower_question
                    or "scientific concept" in lower_question
                    or lower_question == "science"
                    or lower_question.startswith("science ")
                ):
                    if "interstellar" in title_lower:
                        answer = (
                            f"🔬 Science Concepts from {title}\n\n"
                            "• Black holes and extreme gravity.\n"
                            "• Time dilation and relativity.\n"
                            "• Wormholes as a theoretical concept.\n"
                            "• Space travel and planetary environments.\n"
                            "• The relationship between space and time.\n\n"
                            "📚 Student takeaway:\n"
                            "The movie can be connected with physics, "
                            "astronomy and space exploration."
                        )
                    elif "martian" in title_lower:
                        answer = (
                            f"🔬 Science Concepts from {title}\n\n"
                            "• Planetary science and the Martian environment.\n"
                            "• Botany and growing food in difficult conditions.\n"
                            "• Chemistry and resource management.\n"
                            "• Engineering and scientific experimentation.\n\n"
                            "📚 Student takeaway:\n"
                            "The movie demonstrates how science and "
                            "engineering can solve practical problems."
                        )
                    elif "oppenheimer" in title_lower:
                        answer = (
                            f"🔬 Science Concepts from {title}\n\n"
                            "• Nuclear physics and atomic energy.\n"
                            "• Scientific research and experimentation.\n"
                            "• Scientific responsibility and ethics.\n"
                            "• The impact of discoveries on society.\n\n"
                            "📚 Student takeaway:\n"
                            "Science should be studied together with "
                            "ethics and social responsibility."
                        )
                    else:
                        answer = (
                            f"🔬 Science Concepts from {title}\n\n"
                            "Students can identify the scientific or "
                            "technical ideas connected with this movie.\n\n"
                            f"🎬 Genre: {genres or 'Not available'}\n\n"
                            f"📖 Movie context:\n"
                            f"{overview or 'Overview not available in the dataset.'}\n\n"
                            "📚 Student activity:\n"
                            "Identify the scientific or technical ideas "
                            "shown in the movie and compare them with "
                            "what you learned in class."
                        )

                # LIFE LESSONS
                elif (
                    "life lesson" in lower_question
                    or "life lessons" in lower_question
                    or "lesson" in lower_question
                ):
                    answer = (
                        f"👤 Life Lessons from {title}\n\n"
                        "• Learn from challenges and mistakes.\n"
                        "• Think carefully before important decisions.\n"
                        "• Teamwork and communication can solve problems.\n"
                        "• Actions have consequences.\n"
                        "• Persistence is important during difficulties.\n\n"
                        "📚 Student activity:\n"
                        f"Choose one character from {title}. "
                        "Write one difficult decision they made, "
                        "what happened because of it, and what lesson "
                        "you can apply to your own life."
                    )

                # CHARACTER ANALYSIS
                elif (
                    "character analysis" in lower_question
                    or "what can students learn from the characters" in lower_question
                    or "characters" in lower_question
                    or "character" in lower_question
                    or "actors" in lower_question
                    or "cast" in lower_question
                ):
                    actors = movie_info.get("actors", [])

                    if isinstance(actors, str):
                        actor_text = actors
                    elif actors:
                        actor_text = ", ".join(
                            str(actor) for actor in actors
                        )
                    else:
                        actor_text = (
                            "Cast information is not available "
                            "in the dataset."
                        )

                    answer = (
                        f"👥 Character Analysis — {title}\n\n"
                        f"Main cast: {actor_text}\n\n"
                        "Students can analyse:\n"
                        "• Character goals and motivation.\n"
                        "• Decisions and consequences.\n"
                        "• Leadership and teamwork.\n"
                        "• Communication and relationships.\n"
                        "• How characters respond to problems.\n\n"
                        "📚 Student activity:\n"
                        "Choose one important character and write "
                        "their goal, main problem, decision and the "
                        "lesson you learned from that character."
                    )

                # REAL WORLD
                elif (
                    "real world" in lower_question
                    or "real-world" in lower_question
                    or "real life" in lower_question
                    or "real-life" in lower_question
                    or "connection" in lower_question
                ):
                    answer = (
                        f"🌐 Real-World Connection — {title}\n\n"
                        f"🎬 Genre: {genres or 'Not available'}\n\n"
                        "Students can connect the movie with real life by "
                        "studying:\n\n"
                        "• Social and human issues.\n"
                        "• Technology or science shown in the story.\n"
                        "• Decisions and their consequences.\n"
                        "• Communication, teamwork and leadership.\n"
                        "• Ethical questions raised by the movie.\n\n"
                        "📚 Student activity:\n"
                        "Pick one situation from the movie and explain "
                        "how a similar situation could happen in real life."
                    )

                # PROBLEM SOLVING
                elif (
                    "problem solving" in lower_question
                    or "problem-solving" in lower_question
                    or "solve problems" in lower_question
                    or "solving problems" in lower_question
                ):
                    answer = (
                        f"💡 Problem-Solving Skills from {title}\n\n"
                        "• Identify the main problem.\n"
                        "• Break a large problem into smaller parts.\n"
                        "• Collect information before deciding.\n"
                        "• Compare possible solutions.\n"
                        "• Work with others when necessary.\n"
                        "• Learn from unsuccessful attempts.\n\n"
                        "📚 Student activity:\n"
                        "Choose one major problem from the movie and "
                        "write the problem, possible solutions, chosen "
                        "solution and result."
                    )

                # GENERAL STUDENT LEARNING
                elif any(
                    keyword in lower_question
                    for keyword in [
                        "student",
                        "students",
                        "learn",
                        "learning",
                        "educational",
                        "education",
                        "knowledge",
                        "what can we learn"
                    ]
                ):
                    answer = (
                        f"🎓 What Students Can Learn from {title}\n\n"
                        f"Students can study {title} through its story, "
                        "characters, themes and real-world connections.\n\n"
                        f"🎬 Genre: {genres or 'Not available'}\n\n"
                        "Useful areas include:\n"
                        "• Critical thinking\n"
                        "• Problem solving\n"
                        "• Character decisions\n"
                        "• Social and real-world connections\n"
                        "• Science or technology, where relevant\n\n"
                        "📚 Student activity:\n"
                        "Write three important ideas you learned from "
                        "the movie and explain how they connect to "
                        "your studies or real life."
                    )

                # NORMAL MOVIE QUESTIONS
                elif "genre" in lower_question:
                    answer = (
                        f"🎬 Genre of {title}: "
                        f"{genres or 'Not available'}."
                    )

                elif (
                    "rating" in lower_question
                    or "score" in lower_question
                    or "imdb" in lower_question
                ):
                    answer = (
                        f"⭐ Rating of {title}: "
                        f"{movie.get('rating') or 'Not available'}."
                    )

                elif (
                    "year" in lower_question
                    or "released" in lower_question
                    or "release date" in lower_question
                ):
                    answer = (
                        f"📅 Release year of {title}: "
                        f"{movie.get('year') or 'Not available'}."
                    )

                elif (
                    "about" in lower_question
                    or "story" in lower_question
                    or "plot" in lower_question
                    or "summary" in lower_question
                    or "overview" in lower_question
                ):
                    answer = (
                        f"📖 About {title}\n\n"
                        f"{overview or 'Overview is not available in the dataset.'}"
                    )

                elif (
                    "recommend" in lower_question
                    or "similar" in lower_question
                    or "suggest" in lower_question
                ):
                    recommendations = (
                        recommend_movies(title, 5)
                        or []
                    )

                    if recommendations:
                        answer = (
                            f"🎬 Movies similar to {title} "
                            "are shown below."
                        )
                    else:
                        answer = (
                            "I couldn't find similar movies "
                            "for this title."
                        )

                else:
                    answer = (
                        f"🎬 {title}\n\n"
                        f"Genre: {genres or 'Not available'}\n"
                        f"Year: {movie.get('year') or 'Not available'}\n"
                        f"Rating: {movie.get('rating') or 'Not available'}\n\n"
                        f"Overview:\n"
                        f"{overview or 'Not available'}"
                    )

    return render_template(
        "ai_assistant.html",
        answer=answer,
        recommendations=recommendations,
        movie=movie,
        movie_name=movie_name,
        question=question,
        error=error,
        user_logged_in=session.get(
            "user_logged_in",
            False
        ),
        user_name=session.get(
            "user_name",
            ""
        )
    )


@app.route(
    "/api/ai-assistant",
    methods=["POST"]
)
def ai_assistant_api():

    data = request.get_json(
        silent=True
    ) or {}

    movie_name = str(
        data.get(
            "movie_name",
            ""
        )
    ).strip()

    question = str(
        data.get(
            "question",
            ""
        )
    ).strip()

    if not movie_name:

        return jsonify({

            "success": False,

            "answer":
                "Please select a movie."

        })

    if not question:

        return jsonify({

            "success": False,

            "answer":
                "Please enter a question."

        })

    exact_movie = get_movie_info(
        movie_name
    )

    if exact_movie is None:

        return jsonify({

            "success": False,

            "answer":
                "Movie not found in the dataset."

        })

    lower_question = question.lower()

    # --------------------------------------------------------
    # STUDENT QUESTION
    # --------------------------------------------------------

    if any(
        keyword in lower_question
        for keyword in [
            "student",
            "students",
            "learn",
            "learning",
            "lesson",
            "lessons",
            "education",
            "educational",
            "science",
            "scientific"
        ]
    ):

        answer = (
            f"Students can learn from "
            f"{exact_movie['title']} by studying its "
            "scientific concepts, themes, characters, "
            "problem-solving methods and real-world connections."
        )

    # --------------------------------------------------------
    # GENRE
    # --------------------------------------------------------

    elif (
        "genre" in lower_question
        or "genres" in lower_question
    ):

        answer = (
            f"{exact_movie['title']} belongs to "
            f"{exact_movie.get('genres') or 'Not available'}."
        )

    # --------------------------------------------------------
    # STORY
    # --------------------------------------------------------

    elif (
        "story" in lower_question
        or "plot" in lower_question
        or "overview" in lower_question
    ):

        answer = (
            exact_movie.get(
                "overview",
                "Overview not available."
            )
        )

    else:

        answer = (
            f"Here is information about "
            f"{exact_movie['title']}."
        )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = (
        recommend_movies(
            exact_movie["title"],
            5
        )
        or []
    )

    return jsonify({

        "success":
            True,

        "answer":
            answer,

        "recommendations":
            recommendations

    })

# ============================================================
# STUDENT MODE - COMPLETE ROUTING
# ============================================================

STUDENT_TOPIC_ALIASES = {
    "concepts": "data-ml-concepts",
    "scientific-concepts": "data-ml-concepts",
    "science": "data-ml-concepts",

    "lessons": "project-learning",
    "what-can-we-learn": "project-learning",

    "characters": "movie-analysis",
    "character-analysis": "movie-analysis",

    "real-world": "real-world-connection",
    "realworld": "real-world-connection",
    "real-world-connection": "real-world-connection",

    "ai": "ai-concepts",

    "recommendations": "recommendation-system",

    "data-analysis": "data-ml-concepts",
    "machine-learning": "data-ml-concepts",
    "data-ml": "data-ml-concepts",

    "project": "project-learning",
    "workflow": "project-learning",
}

STUDENT_TOPICS = {
    "movie-analysis": {
        "title": "Characters & Themes",
        "icon": "🎬",
        "description": "Learn how movie title, genre, rating, year, cast and overview are represented in the recommendation system.",
        "content": "Movie analysis means studying important information about a movie. Students can examine title, genre, rating, release year, cast, overview and other movie features."
    },
    "recommendation-system": {
        "title": "Recommendation System",
        "icon": "🤖",
        "description": "Understand how the system finds movies similar to a selected movie.",
        "content": "The recommendation engine compares movie features. The project uses TF-IDF to represent movie text and Cosine Similarity to compare movie feature vectors."
    },
    "data-ml-concepts": {
        "title": "Scientific Concepts",
        "icon": "📊",
        "description": "Learn dataset, preprocessing, feature extraction, TF-IDF and Cosine Similarity.",
        "content": "Important concepts include the movie dataset, data preprocessing, feature extraction, TF-IDF, Cosine Similarity and recommendation generation."
    },
    "ai-concepts": {
        "title": "AI Concepts",
        "icon": "🧠",
        "description": "Learn the AI concepts used for movie recommendation and movie-related assistance.",
        "content": "Artificial Intelligence enables computer systems to perform tasks that normally require human intelligence. In this project, AI concepts support movie recommendation and movie-related learning assistance."
    },
    "search-discovery": {
        "title": "Search & Discovery",
        "icon": "🔍",
        "description": "Learn how movie search finds matching titles from the dataset.",
        "content": "The search feature accepts a movie name and compares it with titles in the dataset. Matching movies are displayed with useful information such as poster, rating, year and genre."
    },
    "project-learning": {
        "title": "What Can We Learn?",
        "icon": "📝",
        "description": "Study the complete Movie Recommendation System workflow and modules.",
        "content": "Project workflow: User Input → Movie Search → Data Preprocessing → Feature Engineering → TF-IDF → Cosine Similarity → Recommendation Engine → Recommended Movies."
    },
        "real-world-connection": {
        "title": "Real World Connection",
        "icon": "🌍",
        "description": "Connect movie ideas with real-world situations, education, technology and careers.",
        "content": "Movies can help students understand how concepts relate to real life. Students can connect movie themes with science, technology, education, teamwork, problem solving, communication and career-related situations."
    }
}


def normalize_student_topic(topic):
    topic = str(topic or "movie-analysis").strip().lower()
    return STUDENT_TOPIC_ALIASES.get(topic, topic)


def get_student_quiz_questions(movie_title=""):

    """
    Generate a movie-specific quiz based on the movie
    selected/search by the student.
    """

    movie_title = str(movie_title or "").strip()

    # ---------------------------------------------------------
    # GET SELECTED MOVIE
    # ---------------------------------------------------------

    movie = get_movie_info(movie_title)

    # If no movie is selected, use featured movie
    if not movie:
        movie = get_featured_movie()

    if not movie:
        return []


    title = str(movie.get("title") or "this movie").strip()

    overview = str(
        movie.get("overview") or ""
    ).strip()

    genres = str(
        movie.get("genres") or ""
    ).strip()

    year = str(
        movie.get("year") or ""
    ).strip()

    rating = str(
        movie.get("rating") or ""
    ).strip()


    # ---------------------------------------------------------
    # CAST
    # ---------------------------------------------------------

    cast_list = []

    actors = movie.get("actors") or []

    if isinstance(actors, list):

        for actor in actors:

            actor = str(actor).strip()

            if actor and actor not in cast_list:
                cast_list.append(actor)

    else:

        actor_text = str(actors).strip()

        if actor_text:

            cast_list = [
                item.strip()
                for item in actor_text.split(",")
                if item.strip()
            ]


    # ---------------------------------------------------------
    # SAFE FALLBACK VALUES
    # ---------------------------------------------------------

    if not genres:
        genres = "Movie"

    if not year:
        year = "the available release year"

    if not rating:
        rating = "Not available"


    # ---------------------------------------------------------
    # FIND OTHER MOVIES FOR DISTRACTORS
    # ---------------------------------------------------------

    other_movies = []

    try:

        for _, row in movies.iterrows():

            other_title = str(
                row.get("title", "")
            ).strip()

            if not other_title:
                continue

            if other_title.lower() == title.lower():
                continue

            other_movies.append(row)

            if len(other_movies) >= 20:
                break

    except Exception:

        other_movies = []


    # ---------------------------------------------------------
    # HELPER: GET OTHER MOVIE OVERVIEWS
    # ---------------------------------------------------------

    other_overviews = []

    for row in other_movies:

        text = str(
            row.get("overview", "")
        ).strip()

        if text and text.lower() != "nan":

            other_overviews.append(text)


    # ---------------------------------------------------------
    # HELPER: GET OTHER GENRES
    # ---------------------------------------------------------

    other_genres = []

    for row in other_movies:

        value = str(
            row.get("genres", "")
        ).strip()

        if value and value.lower() != "nan":

            formatted = format_genres(value)

            if formatted and formatted not in other_genres:

                other_genres.append(formatted)


    # ---------------------------------------------------------
    # HELPER: GET OTHER YEARS
    # ---------------------------------------------------------

    other_years = []

    for row in other_movies:

        try:

            other_year = str(
                get_year_from_row(row)
            ).strip()

            if (
                other_year
                and other_year.lower() != "nan"
                and other_year != year
            ):

                if other_year not in other_years:

                    other_years.append(other_year)

        except Exception:

            pass


    # =========================================================
    # QUESTION 1 - STORY / OVERVIEW
    # =========================================================

    questions = []


    if overview:

        story_options = [
            overview
        ]

        for text in other_overviews:

            if text not in story_options:

                story_options.append(text)

            if len(story_options) == 4:
                break


        # If dataset doesn't provide enough other overviews,
        # use safe generic alternatives.

        fallback_story_options = [

            "A story unrelated to the movie's main events.",

            "A completely different story with unrelated characters.",

            "A story focused on events that are not part of this movie."

        ]


        for text in fallback_story_options:

            if len(story_options) >= 4:
                break

            if text not in story_options:

                story_options.append(text)


        questions.append({

            "question":
                f"What is the main story of {title}?",

            "options":
                story_options[:4],

            "answer":
                overview

        })


    # =========================================================
    # QUESTION 2 - GENRE
    # =========================================================

    genre_options = [
        genres
    ]


    for value in other_genres:

        if value not in genre_options:

            genre_options.append(value)

        if len(genre_options) == 4:

            break


    genre_fallbacks = [

        "Drama",

        "Comedy",

        "Documentary"

    ]


    for value in genre_fallbacks:

        if len(genre_options) >= 4:

            break

        if value not in genre_options:

            genre_options.append(value)


    questions.append({

        "question":
            f"Which genre is associated with {title}?",

        "options":
            genre_options[:4],

        "answer":
            genres

    })


    # =========================================================
    # QUESTION 3 - RELEASE YEAR
    # =========================================================

    year_options = [
        year
    ]


    for value in other_years:

        if value not in year_options:

            year_options.append(value)

        if len(year_options) == 4:

            break


    # Final fallback years

    fallback_years = [

        "2010",

        "2012",

        "2015",

        "2018"

    ]


    for value in fallback_years:

        if len(year_options) >= 4:

            break

        if value not in year_options:

            year_options.append(value)


    questions.append({

        "question":
            f"In which year was {title} released?",

        "options":
            year_options[:4],

        "answer":
            year

    })


    # =========================================================
    # QUESTION 4 - CAST
    # =========================================================

    if cast_list:

        main_actor = cast_list[0]

        cast_options = [
            main_actor
        ]


        # Get actors from other movies

        for row in other_movies:

            try:

                other_info = get_movie_info(
                    str(row.get("title", ""))
                )

                if not other_info:
                    continue

                other_actors = (
                    other_info.get("actors") or []
                )


                if isinstance(other_actors, list):

                    for actor in other_actors:

                        actor = str(actor).strip()

                        if (
                            actor
                            and actor not in cast_options
                        ):

                            cast_options.append(actor)

                        if len(cast_options) == 4:

                            break


                if len(cast_options) == 4:

                    break

            except Exception:

                continue


        cast_fallbacks = [

            "Leonardo DiCaprio",

            "Tom Hanks",

            "Robert Downey Jr."

        ]


        for actor in cast_fallbacks:

            if len(cast_options) >= 4:

                break

            if actor not in cast_options:

                cast_options.append(actor)


        questions.append({

            "question":
                f"Which actor is listed in the cast of {title}?",

            "options":
                cast_options[:4],

            "answer":
                main_actor

        })


    # =========================================================
    # QUESTION 5 - RATING
    # =========================================================

    rating_options = [
        rating
    ]


    rating_fallbacks = [

        "7.5",

        "8.0",

        "8.5",

        "9.0"

    ]


    for value in rating_fallbacks:

        if len(rating_options) >= 4:

            break

        if value not in rating_options:

            rating_options.append(value)


    questions.append({

        "question":
            f"What rating is recorded for {title} in the movie dataset?",

        "options":
            rating_options[:4],

        "answer":
            rating

    })


    # =========================================================
    # ENSURE EXACTLY 5 QUESTIONS
    # =========================================================

    return questions[:5]


def current_student_id():
    return session.get("user_id")


def get_featured_movie():
    """Return a valid featured movie from the loaded dataset."""
    preferred = ["Interstellar", "Inception", "The Dark Knight", "The Matrix"]
    for title in preferred:
        info = get_movie_info(title)
        if info:
            return info
    if len(movies) > 0:
        return get_movie_info(str(movies.iloc[0]["title"]))
    return None


def student_context(**kwargs):
    """Common context for every Student Mode page."""
    defaults = {
        "query": "",
        "results": [],
        "topic": "",
        "result": "",
        "error": "",
        "featured_movie": get_featured_movie(),
        "user_logged_in": session.get("user_logged_in", False),
        "user_name": session.get("user_name", ""),
        "student_section": "dashboard",
        "student_progress": 0,
        "student_bookmarks": [],
        "student_notes": [],
        "quiz_questions": [],
        "quiz_result": None,
        "movie": None,
        "movie_title": "",
        "learning_topics": list(STUDENT_TOPICS.items()),
        "achievements": [],
    }
    defaults.update(kwargs)
    return defaults


def render_student_dashboard(**kwargs):
    return render_template("student_mode.html", **student_context(**kwargs))


# ------------------------------------------------------------
# MAIN STUDENT DASHBOARD
# ------------------------------------------------------------
@app.route("/student-mode", methods=["GET", "POST"])
def student_mode():

    query = (
        request.args.get("query")
        or request.form.get("query")
        or ""
    ).strip()

    results = []
    error = ""

    # ================= SEARCH =================

    if query:

        matches = movies[
            movies["title"]
            .astype(str)
            .str.contains(
                query,
                case=False,
                na=False,
                regex=False
            )
        ]

        for _, row in matches.head(10).iterrows():

            results.append({
                "title": str(row["title"]),
                "poster_url": get_poster_from_row(row),
                "rating": get_rating_from_row(row),
                "year": get_year_from_row(row),
                "genres": format_genres(
                    row.get("genres", "")
                )
            })

        if not results:
            error = "No movies found for your search."

    # ================= FEATURED MOVIE =================

    featured = get_featured_movie()

    if results:
        featured = (
            get_movie_info(results[0]["title"])
            or featured
        )

    # ================= STUDENT PROGRESS =================

    user_id = current_student_id()

    completed_modules = 0
    total_modules = len(STUDENT_TOPICS)
    quiz_count = 0
    saved_count = 0
    progress_percent = 0

    if user_id:

        conn = get_db_connection()

        completed_modules = conn.execute(
            """
            SELECT COUNT(*)
            FROM student_progress
            WHERE user_id=?
            AND completed=1
            """,
            (user_id,)
        ).fetchone()[0]

        quiz_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM quiz_results
            WHERE user_id=?
            """,
            (user_id,)
        ).fetchone()[0]

        saved_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM student_bookmarks
            WHERE user_id=?
            """,
            (user_id,)
        ).fetchone()[0]

        conn.close()

        progress_percent = round(
            (completed_modules / total_modules) * 100
        ) if total_modules else 0

    # ================= DASHBOARD =================

    return render_student_dashboard(

        query=query,

        results=results,

        featured_movie=featured,

        error=error,

        student_section="dashboard",

        learning_topics=list(
            STUDENT_TOPICS.items()
        ),

        student_progress=progress_percent,

        completed_modules=completed_modules,

        total_modules=total_modules,

        quiz_count=quiz_count,

        saved_count=saved_count,

        progress_percent=progress_percent
    )

# ------------------------------------------------------------
# STUDENT MOVIE SEARCH
# ------------------------------------------------------------
@app.route("/student-mode/search", methods=["GET", "POST"])
def student_mode_search():
    query = (request.args.get("query") or request.form.get("query") or
             request.args.get("movie") or request.form.get("movie") or "").strip()

    results = []
    if query:
        matches = movies[movies["title"].astype(str).str.contains(
            query, case=False, na=False, regex=False
        )]
        for _, row in matches.head(10).iterrows():
            results.append({
                "title": str(row["title"]),
                "poster_url": get_poster_from_row(row),
                "rating": get_rating_from_row(row),
                "year": get_year_from_row(row),
                "genres": format_genres(row.get("genres", ""))
            })

    return render_template(
        "student_search.html" if os.path.exists(os.path.join(BASE_DIR, "templates", "student_search.html")) else "student_mode.html",
        **student_context(
            query=query,
            results=results,
            featured_movie=(get_movie_info(results[0]["title"]) if results else get_featured_movie()),
            error="No movies found for your search." if query and not results else "",
            student_section="search"
        )
    )


# Old URL used by the existing student_mode.html.
@app.route("/student-search", methods=["GET", "POST"])
def student_search_compat():
    query = (request.args.get("movie") or request.args.get("query") or
             request.form.get("movie") or request.form.get("query") or "").strip()
    return redirect(url_for("student_mode_search", query=query))


# ------------------------------------------------------------
# STUDENT TOPIC / LEARNING MODULE
# ------------------------------------------------------------
def _render_student_topic(topic=None):

    # ========================================================
    # NORMALIZE TOPIC
    # ========================================================

    topic = normalize_student_topic(
        topic
        or request.args.get("topic")
        or "movie-analysis"
    )

    if topic not in STUDENT_TOPICS:
        topic = "movie-analysis"


    # ========================================================
    # GET SELECTED MOVIE
    # ========================================================

    movie_title = (
        request.args.get("movie")
        or request.args.get("query")
        or request.form.get("movie")
        or request.form.get("query")
        or ""
    ).strip()

    movie = None


    # ========================================================
    # FIND MOVIE FROM DATASET
    # ========================================================

    if movie_title:

        movie = get_movie_info(movie_title)

        # Exact dataset-title fallback
        if movie is None:

            matches = movies[
                movies["title"]
                .astype(str)
                .str.strip()
                .str.lower()
                == movie_title.lower()
            ]

            if not matches.empty:

                real_title = str(
                    matches.iloc[0]["title"]
                )

                movie = get_movie_info(
                    real_title
                )


    # ========================================================
    # FEATURED MOVIE FALLBACK
    # ========================================================

    if movie is None and not movie_title:

        movie = get_featured_movie()


    # ========================================================
    # BASE TOPIC DATA
    # ========================================================

    topic_data = dict(
        STUDENT_TOPICS[topic]
    )


    # ========================================================
    # MOVIE-SPECIFIC LEARNING CONTENT
    # ========================================================

    if movie:

        title = str(
            movie.get(
                "title",
                "Selected Movie"
            )
        ).strip()


        genres = str(
            movie.get(
                "genres",
                ""
            )
            or "Not available"
        ).strip()


        overview = str(
            movie.get(
                "overview",
                ""
            )
            or "Overview not available."
        ).strip()


        rating = str(
            movie.get(
                "rating",
                "N/A"
            )
            or "N/A"
        ).strip()


        year = str(
            movie.get(
                "year",
                "N/A"
            )
            or "N/A"
        ).strip()


        # ====================================================
        # ACTORS
        # ====================================================

        actors = movie.get(
            "actors",
            []
        )

        if isinstance(actors, list):

            actor_names = [
                str(actor).strip()
                for actor in actors
                if str(actor).strip()
            ]

        else:

            actor_names = [
                item.strip()
                for item in str(actors).split(",")
                if item.strip()
            ]


        actor_text = (
            ", ".join(actor_names[:5])
            if actor_names
            else "Actor information not available."
        )


        # ====================================================
        # MOVIE ANALYSIS
        # ====================================================

        if topic == "movie-analysis":

            topic_data["content"] = f"""
            <p>
                <strong>{title}</strong> is the selected movie for
                this learning module.
            </p>

            <p>
                Students can analyse this movie using its
                <strong>genre, release year, rating and story overview</strong>.
            </p>

            <p>
                <strong>Movie Title:</strong>
                {title}
            </p>

            <p>
                <strong>Genre:</strong>
                {genres}
            </p>

            <p>
                <strong>Release Year:</strong>
                {year}
            </p>

            <p>
                <strong>Rating:</strong>
                {rating}
            </p>

            <p>
                <strong>Cast Information:</strong>
                {actor_text}
            </p>

            <p>
                <strong>Story / Overview:</strong>
                {overview}
            </p>

            <p>
                <strong>Learning Point:</strong>
                Students can use the story and movie attributes
                to understand how real-world information can be
                represented as structured data in a movie dataset.
            </p>
            """


        # ====================================================
        # RECOMMENDATION SYSTEM
        # ====================================================

        elif topic == "recommendation-system":

            topic_data["content"] = f"""
            <p>
                The selected movie is
                <strong>{title}</strong>.
            </p>

            <p>
                A movie recommendation system represents each
                movie using useful information such as
                <strong>genre, story description and other
                textual features</strong>.
            </p>

            <p>
                For <strong>{title}</strong>, the important movie
                information includes:
            </p>

            <ul>
                <li>
                    <strong>Genres:</strong> {genres}
                </li>

                <li>
                    <strong>Overview:</strong> {overview}
                </li>

                <li>
                    <strong>Release Year:</strong> {year}
                </li>

                <li>
                    <strong>Rating:</strong> {rating}
                </li>
            </ul>

            <p>
                The project uses
                <strong>TF-IDF</strong> to convert movie text
                into numerical feature vectors.
            </p>

            <p>
                <strong>Cosine Similarity</strong> compares the
                selected movie with other movie vectors.
            </p>

            <p>
                Therefore, movies having similar feature
                representations can be discovered as
                recommendations for <strong>{title}</strong>.
            </p>
            """


        # ====================================================
        # SCIENTIFIC / DATA & ML CONCEPTS
        # ====================================================

        elif topic == "data-ml-concepts":

            topic_data["content"] = f"""
            <p>
                <strong>{title}</strong> is one record in the
                Movie Recommendation System dataset.
            </p>

            <p>
                This movie provides real data that can be used
                to understand important data-processing concepts.
            </p>

            <p>
                <strong>Movie Data:</strong>
            </p>

            <ul>
                <li>
                    <strong>Title:</strong> {title}
                </li>

                <li>
                    <strong>Genres:</strong> {genres}
                </li>

                <li>
                    <strong>Release Year:</strong> {year}
                </li>

                <li>
                    <strong>Rating:</strong> {rating}
                </li>

                <li>
                    <strong>Overview:</strong> {overview}
                </li>
            </ul>

            <p>
                <strong>Data Preprocessing:</strong>
                Movie information is cleaned and prepared before
                it is used by the recommendation system.
            </p>

            <p>
                <strong>Feature Extraction:</strong>
                Important textual information from movies can be
                converted into useful numerical representations.
            </p>

            <p>
                <strong>TF-IDF:</strong>
                Text information is represented numerically so
                that movies can be compared computationally.
            </p>

            <p>
                <strong>Cosine Similarity:</strong>
                The system compares feature vectors to identify
                movies with similar characteristics.
            </p>

            <p>
                Therefore, <strong>{title}</strong> can be used
                as a real dataset example for understanding
                data preprocessing, feature extraction and
                similarity-based recommendation.
            </p>
            """


        # ====================================================
        # AI CONCEPTS
        # ====================================================

        elif topic == "ai-concepts":

            topic_data["content"] = f"""
            <p>
                <strong>{title}</strong> can be used as an example
                to understand how Artificial Intelligence and
                data-processing techniques can support
                movie discovery.
            </p>

            <p>
                The system first represents the movie using
                information such as:
            </p>

            <ul>
                <li>
                    <strong>Genre:</strong> {genres}
                </li>

                <li>
                    <strong>Story:</strong> {overview}
                </li>

                <li>
                    <strong>Release Year:</strong> {year}
                </li>

                <li>
                    <strong>Rating:</strong> {rating}
                </li>
            </ul>

            <p>
                These features help the recommendation system
                understand the characteristics of
                <strong>{title}</strong>.
            </p>

            <p>
                The project then uses
                <strong>TF-IDF</strong> and
                <strong>Cosine Similarity</strong>
                to compare movie information.
            </p>

            <p>
                This demonstrates how AI-related techniques and
                data processing can be combined to build an
                interactive movie recommendation application.
            </p>
            """


        # ====================================================
        # SEARCH & DISCOVERY
        # ====================================================

        elif topic == "search-discovery":

            topic_data["content"] = f"""
            <p>
                The selected movie is
                <strong>{title}</strong>.
            </p>

            <p>
                The Student Mode search feature allows a student
                to enter a movie name and find the corresponding
                record in the movie dataset.
            </p>

            <p>
                Information available for
                <strong>{title}</strong> includes:
            </p>

            <ul>
                <li>
                    <strong>Genre:</strong> {genres}
                </li>

                <li>
                    <strong>Release Year:</strong> {year}
                </li>

                <li>
                    <strong>Rating:</strong> {rating}
                </li>

                <li>
                    <strong>Overview:</strong> {overview}
                </li>
            </ul>

            <p>
                After the movie is found, its information can be
                used by the recommendation engine to discover
                other movies with similar characteristics.
            </p>

            <p>
                This demonstrates the connection between
                <strong>search, dataset retrieval and
                recommendation</strong>.
            </p>
            """


        # ====================================================
        # WHAT CAN WE LEARN?
        # ====================================================

        elif topic == "project-learning":

            topic_data["content"] = f"""
            <p>
                <strong>{title}</strong> can be studied as a
                movie-specific learning example.
            </p>

            <p>
                <strong>What can students learn from this movie?</strong>
            </p>

            <p>
                First, students can understand the story described
                in the movie overview:
            </p>

            <p>
                <strong>{overview}</strong>
            </p>

            <p>
                The movie belongs to the following genre category:
                <strong>{genres}</strong>.
            </p>

            <p>
                Students can observe the story, situations,
                challenges and decisions shown in
                <strong>{title}</strong> and connect them with
                concepts such as:
            </p>

            <ul>
                <li>Problem solving</li>
                <li>Critical thinking</li>
                <li>Decision making</li>
                <li>Communication</li>
                <li>Teamwork</li>
            </ul>

            <p>
                The exact lessons depend on the story and themes
                represented in <strong>{title}</strong>.
            </p>

            <p>
                <strong>Student Activity:</strong>
                Read the movie overview, identify the main problem
                or challenge, and think about how the situations
                shown in the movie can be connected with real-life
                learning.
            </p>
            """


        # ====================================================
        # REAL WORLD CONNECTION
        # ====================================================

        elif topic == "real-world-connection":

            topic_data["content"] = f"""
            <p>
                <strong>{title}</strong> can be connected with
                real-world learning by studying the information
                represented in its dataset record and its story.
            </p>

            <p>
                <strong>Movie Context:</strong>
                {overview}
            </p>

            <p>
                <strong>Genre:</strong>
                {genres}
            </p>

            <p>
                The movie can be used as an example to understand
                how entertainment data is stored, searched,
                analysed and used for personalized content
                discovery.
            </p>

            <p>
                In the project, information about
                <strong>{title}</strong> becomes part of the
                recommendation workflow.
            </p>

            <p>
                The system can process movie information and
                compare it with other movies to help users discover
                related content.
            </p>

            <p>
                <strong>Real-World Learning:</strong>
                Students can understand how data-driven systems
                are used in digital entertainment applications
                for search, discovery and personalization.
            </p>
            """


    # ========================================================
    # NO MOVIE SELECTED
    # ========================================================

    else:

        topic_data["content"] = """
        <p>
            Please select or search for a movie to explore
            this learning module.
        </p>

        <p>
            Once a movie is selected, the learning content will
            be generated using that movie's information from
            the dataset.
        </p>
        """


    # ========================================================
    # STORE TOPIC INFORMATION
    # ========================================================

    topic_data["slug"] = topic
    topic_data["movie"] = movie


    # ========================================================
    # STUDENT PROGRESS
    # ========================================================

    user_id = current_student_id()

    completed = False

    if user_id:

        conn = get_db_connection()

        row = conn.execute(
            """
            SELECT completed
            FROM student_progress
            WHERE user_id=? AND topic=?
            """,
            (
                user_id,
                topic
            )
        ).fetchone()

        conn.close()

        completed = bool(
            row and row["completed"]
        )


    # ========================================================
    # RENDER LEARNING PAGE
    # ========================================================

    return render_template(
        "student_topic.html",
        **student_context(
            topic=topic_data,
            movie=movie,
            movie_title=(
                movie["title"]
                if movie
                else movie_title
            ),
            topic_name=topic,
            completed=completed,
            student_section="learning"
        )
    )


@app.route("/student-topic")
def student_topic():

    topic = request.args.get(
        "topic",
        "movie-analysis"
    )

    return _render_student_topic(topic)


@app.route("/student-mode/topic/<topic_name>")
def student_mode_topic(topic_name):
    return _render_student_topic(topic_name)


@app.route("/student-topic/<topic_name>")
def student_topic_path(topic_name):
    return _render_student_topic(topic_name)




# ------------------------------------------------------------
# MY LEARNING
# ------------------------------------------------------------
@app.route("/student-learning")
def student_learning():

    featured = get_featured_movie()

    return render_template(
        "student_learning.html",
        **student_context(
            learning_topics=list(STUDENT_TOPICS.items()),
            featured_movie=featured,
            movie=featured,
            movie_title=featured["title"] if featured else "",
            student_section="learning"
        )
    )


@app.route("/student-mode/learning")
def student_learning_compat():
    return redirect(url_for("student_learning"))


# ------------------------------------------------------------
# QUIZ
# ------------------------------------------------------------
@app.route("/student-quizzes", methods=["GET", "POST"])
def student_quizzes():
    movie_title = (request.args.get("movie") or request.form.get("movie") or "").strip()
    quiz_questions = get_student_quiz_questions(movie_title)
    quiz_result = None

    if request.method == "POST":
        score = 0
        for i, q in enumerate(quiz_questions):
            if request.form.get(f"q{i}", "") == q["answer"]:
                score += 1

        quiz_result = {
            "score": score,
            "total": len(quiz_questions),
            "percentage": round(score * 100 / len(quiz_questions)),
            "message": "Excellent! 🎉" if score == len(quiz_questions) else "Good attempt! Keep learning and try again. 📚"
        }

        user_id = current_student_id()
        if user_id:
            conn = get_db_connection()
            conn.execute(
                "INSERT INTO quiz_results (user_id, movie_title, score, total) VALUES (?, ?, ?, ?)",
                (user_id, movie_title or None, score, len(quiz_questions))
            )
            conn.commit()
            conn.close()

    movie = get_movie_info(movie_title) if movie_title else get_featured_movie()
    template = "student_quiz.html" if os.path.exists(os.path.join(BASE_DIR, "templates", "student_quiz.html")) else "student_mode.html"
    return render_template(
        template,
        **student_context(
            quiz_questions=quiz_questions,
            quiz_result=quiz_result,
            movie=movie,
            featured_movie=movie,
            movie_title=movie["title"] if movie else movie_title,
            student_section="quizzes"
        )
    )


@app.route("/student-quiz", methods=["GET", "POST"])
def student_quiz():
    return student_quizzes()


@app.route("/student-mode/quiz", methods=["GET", "POST"])
def student_quiz_compat():
    return student_quizzes()


# ------------------------------------------------------------
# ACHIEVEMENTS
# ------------------------------------------------------------
@app.route("/student-achievements")
def student_achievements():
    user_id = current_student_id()
    completed = 0
    quiz_count = 0
    best_score = 0

    if user_id:
        conn = get_db_connection()
        completed = conn.execute(
            "SELECT COUNT(*) AS c FROM student_progress WHERE user_id=? AND completed=1",
            (user_id,)
        ).fetchone()["c"]
        quiz_count = conn.execute(
            "SELECT COUNT(*) AS c FROM quiz_results WHERE user_id=?",
            (user_id,)
        ).fetchone()["c"]
        row = conn.execute(
            "SELECT MAX(CASE WHEN total>0 THEN ROUND(score*100.0/total) ELSE 0 END) AS best FROM quiz_results WHERE user_id=?",
            (user_id,)
        ).fetchone()
        best_score = int(row["best"] or 0)
        conn.close()

    achievements = [
        {"icon": "🔬", "title": "Science Explorer", "text": "Complete learning topics", "value": completed},
        {"icon": "🎬", "title": "Movie Scholar", "text": "Explore movie learning content", "value": completed},
        {"icon": "🏆", "title": "Quiz Master", "text": "Best quiz score", "value": f"{best_score}%"},
    ]

    return render_student_dashboard(
        achievements=achievements,
        student_section="achievements"
    )


@app.route("/student-mode/achievements")
def student_achievements_compat():
    return redirect(url_for("student_achievements"))


# ------------------------------------------------------------
# BOOKMARKS
# ------------------------------------------------------------
@app.route("/student-bookmarks", methods=["GET", "POST"])
def student_bookmarks():

    user_id = current_student_id()

    # ========================================================
    # GET MOVIE
    # ========================================================

    movie_title = (
        request.args.get("movie")
        or request.form.get("movie_title")
        or request.form.get("movie")
        or ""
    ).strip()


    # ========================================================
    # SAVE MOVIE
    # ========================================================

    if user_id and movie_title:

        movie_info = get_movie_info(movie_title)

        if movie_info:

            conn = get_db_connection()

            conn.execute(
                """
                INSERT OR IGNORE INTO student_bookmarks
                (user_id, movie_title)
                VALUES (?, ?)
                """,
                (
                    user_id,
                    movie_info["title"]
                )
            )

            conn.commit()
            conn.close()


    # ========================================================
    # LOAD SAVED MOVIES
    # ========================================================

    bookmarks = []

    if user_id:

        conn = get_db_connection()

        rows = conn.execute(
            """
            SELECT movie_title
            FROM student_bookmarks
            WHERE user_id=?
            ORDER BY created_at DESC
            """,
            (user_id,)
        ).fetchall()

        conn.close()


        for row in rows:

            info = get_movie_info(
                row["movie_title"]
            )

            if info:
                bookmarks.append(info)


    # ========================================================
    # SHOW SAVED MOVIES PAGE
    # ========================================================

    return render_template(
        "student_bookmarks.html",
        **student_context(
            student_bookmarks=bookmarks,
            student_section="bookmarks",
            movie_title=movie_title
        )
    )


@app.route("/student-bookmark", methods=["GET", "POST"])
def student_bookmark():
    return student_bookmarks()


@app.route("/student-mode/bookmarks", methods=["GET", "POST"])
def student_bookmarks_compat():
    return student_bookmarks()


# ------------------------------------------------------------
# NOTES
# ------------------------------------------------------------
@app.route("/student-notes", methods=["GET", "POST"])
def student_notes():

    user_id = current_student_id()

    # ========================================================
    # GET MOVIE
    # ========================================================

    movie_title = (
        request.args.get("movie")
        or request.form.get("movie_title")
        or request.form.get("movie")
        or ""
    ).strip()


    # ========================================================
    # SAVE NOTE
    # ========================================================

    if request.method == "POST" and user_id:

        movie_title = (
            request.form.get("movie_title")
            or request.form.get("movie")
            or ""
        ).strip()

        note = (
            request.form.get("note")
            or ""
        ).strip()


        if movie_title and note:

            conn = get_db_connection()

            conn.execute(
                """
                INSERT INTO student_notes
                (user_id, movie_title, note)
                VALUES (?, ?, ?)
                """,
                (
                    user_id,
                    movie_title,
                    note
                )
            )

            conn.commit()
            conn.close()


    # ========================================================
    # LOAD NOTES
    # ========================================================

    notes = []

    if user_id:

        conn = get_db_connection()

        notes = conn.execute(
    """
    SELECT
        id,
        movie_title,
        note,
        created_at
    FROM student_notes
    WHERE user_id=?
    ORDER BY created_at DESC
    """,
    (user_id,)
).fetchall()

        conn.close()


    # ========================================================
    # SHOW NOTES PAGE
    # ========================================================

    return render_template(
        "student_notes.html",
        **student_context(
            student_notes=notes,
            student_section="notes",
            movie_title=movie_title
        )
    )


# PROGRESS
# ------------------------------------------------------------
@app.route("/student-progress")
def student_progress():

    user_id = current_student_id()

    progress_rows = []
    completed = 0
    percent = 0

    if user_id:

        conn = get_db_connection()

        progress_rows = conn.execute(
            """
            SELECT topic, completed, updated_at
            FROM student_progress
            WHERE user_id=?
            ORDER BY updated_at DESC
            """,
            (user_id,)
        ).fetchall()

        completed = sum(
            int(row["completed"] or 0)
            for row in progress_rows
        )

        total = len(STUDENT_TOPICS)

        percent = round(
            (completed / total) * 100
        ) if total else 0

        conn.close()

    return render_template(
        "student_progress.html",
        progress_rows=progress_rows,
        completed=completed,
        total=len(STUDENT_TOPICS),
        percent=percent,
        featured_movie=get_featured_movie(),
        user_logged_in=session.get(
            "user_logged_in",
            False
        ),
        user_name=session.get(
            "user_name",
            ""
        ),
        student_section="progress"
    )

@app.route("/student-mode/progress")
def student_progress_compat():
    return redirect(url_for("student_progress"))


@app.route("/student-progress/complete", methods=["POST"])
def student_progress_complete():
    user_id = current_student_id()
    topic = normalize_student_topic(request.form.get("topic", ""))
    if user_id and topic in STUDENT_TOPICS:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO student_progress (user_id, topic, completed) VALUES (?, ?, 1) "
            "ON CONFLICT(user_id, topic) DO UPDATE SET completed=1, updated_at=CURRENT_TIMESTAMP",
            (user_id, topic)
        )
        conn.commit()
        conn.close()

    return redirect(url_for("student_topic", topic=topic))


# Compatibility endpoint for templates that post a learning topic directly.
@app.route("/student-topic/<topic>/complete", methods=["POST"])
def student_topic_complete(topic):
    user_id = current_student_id()
    topic = normalize_student_topic(topic)
    if user_id and topic in STUDENT_TOPICS:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO student_progress (user_id, topic, completed) VALUES (?, ?, 1) "
            "ON CONFLICT(user_id, topic) DO UPDATE SET completed=1, updated_at=CURRENT_TIMESTAMP",
            (user_id, topic)
        )
        conn.commit()
        conn.close()
    return redirect(url_for("student_topic", topic=topic))

# ============================================================
# DISCUSSION PAGE
# ============================================================

@app.route(
    "/discussion",
    methods=["GET", "POST"]
)
def discussion():

    error = ""
    success = ""

    if request.method == "POST":

        if not session.get(
            "user_logged_in"
        ):

            return redirect(
                url_for("user_login")
            )

        title = request.form.get(
            "title",
            ""
        ).strip()

        content = request.form.get(
            "content",
            ""
        ).strip()

        if not title:

            error = (
                "Please enter a discussion title."
            )

        elif not content:

            error = (
                "Please enter discussion content."
            )

        else:

            conn = get_db_connection()

            conn.execute(
                """
                INSERT INTO discussions
                (user_id, user_name, title, content)
                VALUES (?, ?, ?, ?)
                """,
                (
                    session.get("user_id"),
                    session.get("user_name", "User"),
                    title,
                    content
                )
            )

            conn.commit()
            conn.close()

            return redirect(
                url_for("discussion")
            )

    conn = get_db_connection()

    discussion_rows = conn.execute(
        """
        SELECT *
        FROM discussions
        ORDER BY created_at DESC
        """
    ).fetchall()

    discussions_list = []

    for item in discussion_rows:

        replies = conn.execute(
            """
            SELECT *
            FROM discussion_replies
            WHERE discussion_id = ?
            ORDER BY created_at ASC
            """,
            (item["id"],)
        ).fetchall()

        discussions_list.append({

            "id":
                item["id"],

            "user_name":
                item["user_name"],

            "title":
                item["title"],

            "content":
                item["content"],

            "created_at":
                item["created_at"],

            "replies":
                replies
        })

    conn.close()

    return render_template(
        "discussion.html",
        discussions=discussions_list,
        error=error,
        success=success,
        user_logged_in=session.get(
            "user_logged_in",
            False
        ),
        user_name=session.get(
            "user_name",
            ""
        )
    )


# ============================================================
# ADD DISCUSSION REPLY
# ============================================================

@app.route(
    "/discussion/<int:discussion_id>/reply",
    methods=["POST"]
)
def discussion_reply(discussion_id):

    if not session.get(
        "user_logged_in"
    ):

        return redirect(
            url_for("user_login")
        )

    content = request.form.get(
        "content",
        ""
    ).strip()

    if content:

        conn = get_db_connection()

        discussion_exists = conn.execute(
            """
            SELECT id
            FROM discussions
            WHERE id = ?
            """,
            (discussion_id,)
        ).fetchone()

        if discussion_exists:

            conn.execute(
                """
                INSERT INTO discussion_replies
                (discussion_id, user_id, user_name, content)
                VALUES (?, ?, ?, ?)
                """,
                (
                    discussion_id,
                    session.get("user_id"),
                    session.get("user_name", "User"),
                    content
                )
            )

            conn.commit()

        conn.close()

    return redirect(
        url_for("discussion")
    )


# ============================================================
# DELETE OWN DISCUSSION
# ============================================================

@app.route(
    "/discussion/<int:discussion_id>/delete",
    methods=["POST"]
)
def delete_discussion(discussion_id):

    if not session.get(
        "user_logged_in"
    ):

        return redirect(
            url_for("user_login")
        )

    conn = get_db_connection()

    discussion_item = conn.execute(
        """
        SELECT *
        FROM discussions
        WHERE id = ?
        """,
        (discussion_id,)
    ).fetchone()

    if discussion_item:

        if (
            discussion_item["user_id"]
            == session.get("user_id")
        ):

            conn.execute(
                """
                DELETE FROM discussion_replies
                WHERE discussion_id = ?
                """,
                (discussion_id,)
            )

            conn.execute(
                """
                DELETE FROM discussions
                WHERE id = ?
                """,
                (discussion_id,)
            )

            conn.commit()

    conn.close()

    return redirect(
        url_for("discussion")
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    error = ""
    success = ""

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        ).strip()

        confirm_password = request.form.get(
            "confirm_password",
            ""
        ).strip()

        if not name:

            error = "Please enter your name."

        elif not email:

            error = "Please enter your email."

        elif not password:

            error = "Please enter a password."

        elif len(password) < 6:

            error = (
                "Password must contain at least "
                "6 characters."
            )

        elif password != confirm_password:

            error = "Passwords do not match."

        if not error:

            conn = get_db_connection()

            existing_user = conn.execute(
                """
                SELECT id
                FROM users
                WHERE email = ?
                """,
                (email,)
            ).fetchone()

            if existing_user:

                error = (
                    "An account with this "
                    "email already exists."
                )

            else:

                hashed_password = (
                    generate_password_hash(
                        password
                    )
                )

                conn.execute(
                    """
                    INSERT INTO users
                    (name, email, password)
                    VALUES (?, ?, ?)
                    """,
                    (
                        name,
                        email,
                        hashed_password
                    )
                )

                conn.commit()

                success = (
                    "Account created successfully! "
                    "You can now login."
                )

            conn.close()

    return render_template(
        "register.html",
        error=error,
        success=success
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def user_login():

    error = ""

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        ).strip()

        if not email or not password:

            error = (
                "Please enter email and password."
            )

        else:

            conn = get_db_connection()

            user = conn.execute(
                """
                SELECT *
                FROM users
                WHERE email = ?
                """,
                (email,)
            ).fetchone()

            conn.close()

            if user and check_password_hash(
                user["password"],
                password
            ):

                session["user_logged_in"] = True
                session["user_id"] = user["id"]
                session["user_name"] = user["name"]
                session["user_email"] = user["email"]

                return redirect(
                    url_for("home")
                )

            error = (
                "Invalid email or password."
            )

    return render_template(
        "user_login.html",
        error=error
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def user_logout():

    session.pop(
        "user_logged_in",
        None
    )

    session.pop(
        "user_id",
        None
    )

    session.pop(
        "user_name",
        None
    )

    session.pop(
        "user_email",
        None
    )

    return redirect(
        url_for("home")
    )


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile")
def user_profile():

    if not session.get(
        "user_logged_in"
    ):

        return redirect(
            url_for("user_login")
        )

    conn = get_db_connection()

    user = conn.execute(
        """
        SELECT
            id,
            name,
            email,
            created_at
        FROM users
        WHERE id = ?
        """,
        (
            session.get("user_id"),
        )
    ).fetchone()

    conn.close()

    if user is None:

        session.clear()

        return redirect(
            url_for("user_login")
        )

    return render_template(
        "profile.html",
        user=user
    )


# ============================================================
# ADMIN LOGIN
# ============================================================

@app.route(
    "/admin",
    methods=["GET", "POST"]
)
def admin_login():

    error = ""

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        if (
            username == ADMIN_USERNAME
            and password == ADMIN_PASSWORD
        ):

            session["admin_logged_in"] = True

            return redirect(
                url_for("admin_dashboard")
            )

        error = (
            "Invalid username or password."
        )

    return render_template(
        "admin_login.html",
        error=error
    )


@app.route("/admin/login")
def admin_login_page():

    return redirect(
        url_for("admin_login")
    )


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@app.route("/admin/dashboard")
def admin_dashboard():

    if not session.get(
        "admin_logged_in"
    ):

        return redirect(
            url_for("admin_login")
        )

    conn = get_db_connection()

    total_users = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    total_discussions = conn.execute(
        "SELECT COUNT(*) FROM discussions"
    ).fetchone()[0]

    conn.close()

    recent_movies = []

    for index, row in movies.head(5).iterrows():

        recent_movies.append({

            "index":
                int(index),

            "title":
                str(row["title"]),

            "genres":
                format_genres(
                    row.get(
                        "genres",
                        "Movie"
                    )
                ),

            "rating":
                get_rating_from_row(row),

            "poster_url":
                get_poster_from_row(row)
        })

    return render_template(
        "admin_dashboard.html",
        total_movies=len(movies),
        total_users=total_users,
        total_ratings=0,
        total_discussions=total_discussions,
        recent_movies=recent_movies
    )


# ============================================================
# ADMIN MOVIES
# ============================================================

@app.route("/admin/movies")
def admin_movies():

    if not session.get(
        "admin_logged_in"
    ):

        return redirect(
            url_for("admin_login")
        )

    search = request.args.get(
        "search",
        ""
    ).strip()

    try:

        page = int(
            request.args.get(
                "page",
                1
            )
        )

    except ValueError:

        page = 1

    page = max(page, 1)

    per_page = 50

    if search:

        filtered_movies = movies[
            movies["title"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
        ].copy()

    else:

        filtered_movies = movies.copy()

    total_filtered = len(
        filtered_movies
    )

    total_pages = max(
        1,
        (
            total_filtered
            + per_page
            - 1
        ) // per_page
    )

    page = min(
        page,
        total_pages
    )

    start = (
        page - 1
    ) * per_page

    end = start + per_page

    page_movies = filtered_movies.iloc[
        start:end
    ]

    movie_list = []

    for index, row in page_movies.iterrows():

        movie_list.append({

            "index":
                int(index),

            "title":
                str(row["title"]),

            "genres":
                format_genres(
                    row.get(
                        "genres",
                        "Movie"
                    )
                ),

            "rating":
                get_rating_from_row(row)
                or "N/A",

            "poster_url":
                get_poster_from_row(row)
        })

    return render_template(
        "admin_movies.html",
        movies=movie_list,
        total_movies=len(movies),
        active_movies=len(movies),
        dataset_status="Ready",
        search=search,
        page=page,
        per_page=per_page,
        total_filtered=total_filtered,
        total_pages=total_pages
    )


# ============================================================
# ADMIN ADD MOVIE
# ============================================================

@app.route(
    "/admin/movies/add",
    methods=["GET", "POST"]
)
def admin_add_movie():

    if not session.get(
        "admin_logged_in"
    ):

        return redirect(
            url_for("admin_login")
        )

    error = ""

    if request.method == "POST":

        title = request.form.get(
            "title",
            ""
        ).strip()

        genres = request.form.get(
            "genres",
            ""
        ).strip()

        overview = request.form.get(
            "overview",
            ""
        ).strip()

        poster_url = request.form.get(
            "poster_url",
            ""
        ).strip()

        rating = request.form.get(
            "rating",
            ""
        ).strip()

        if not title:

            error = (
                "Movie title is required."
            )

        if not error:

            existing = movies[
                movies["title"]
                .astype(str)
                .str.lower()
                .str.strip()
                == title.lower()
            ]

            if len(existing) > 0:

                error = (
                    "Movie already exists."
                )

        if not error:

            new_movie = {
                column: ""
                for column in movies.columns
            }

            new_movie["title"] = title
            new_movie["genres"] = genres
            new_movie["overview"] = overview

            if poster_column is not None:

                new_movie[
                    poster_column
                ] = poster_url

            if rating:

                try:

                    new_movie[
                        rating_column
                    ] = float(rating)

                except ValueError:

                    new_movie[
                        rating_column
                    ] = ""

            movies.loc[
                len(movies)
            ] = new_movie

            movies["_original_cast"] = (
                movies["cast"].copy()
            )

            movies["_original_crew"] = (
                movies["crew"].copy()
            )

            movies.to_csv(
                DATASET_PATH,
                index=False
            )

            rebuild_recommendation_model()

            return redirect(
                url_for("admin_movies")
            )

    return render_template(
        "admin_add_movie.html",
        error=error
    )


# ============================================================
# ADMIN EDIT MOVIE
# ============================================================

@app.route(
    "/admin/movies/edit/<int:movie_index>",
    methods=["GET", "POST"]
)
def admin_edit_movie(movie_index):

    if not session.get(
        "admin_logged_in"
    ):

        return redirect(
            url_for("admin_login")
        )

    if (
        movie_index < 0
        or movie_index >= len(movies)
    ):

        return redirect(
            url_for("admin_movies")
        )

    actual_index = movies.index[
        movie_index
    ]

    row = movies.loc[
        actual_index
    ]

    if request.method == "POST":

        title = request.form.get(
            "title",
            ""
        ).strip()

        genres = request.form.get(
            "genres",
            ""
        ).strip()

        overview = request.form.get(
            "overview",
            ""
        ).strip()

        poster_url = request.form.get(
            "poster_url",
            ""
        ).strip()

        rating = request.form.get(
            "rating",
            ""
        ).strip()

        error = ""

        if not title:

            error = (
                "Movie title is required."
            )

        else:

            duplicate = movies[
                (
                    movies["title"]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    == title.lower()
                )
                &
                (
                    movies.index
                    != actual_index
                )
            ]

            if len(duplicate) > 0:

                error = (
                    "Another movie with this "
                    "title already exists."
                )

        if error:

            movie_data = {

                "title":
                    title,

                "genres":
                    genres,

                "overview":
                    overview,

                "poster_url":
                    poster_url,

                "rating":
                    rating
            }

            return render_template(
                "admin_edit_movie.html",
                movie=movie_data,
                movie_index=movie_index,
                error=error
            )

        old_title = str(
            row.get(
                "title",
                ""
            )
        ).strip()

        old_genres = str(
            row.get(
                "genres",
                ""
            )
        ).strip()

        old_overview = str(
            row.get(
                "overview",
                ""
            )
        ).strip()

        model_needs_rebuild = (
            title != old_title
            or genres != old_genres
            or overview != old_overview
        )

        movies.at[
            actual_index,
            "title"
        ] = title

        movies.at[
            actual_index,
            "genres"
        ] = genres

        movies.at[
            actual_index,
            "overview"
        ] = overview

        if poster_column is not None:

            movies.at[
                actual_index,
                poster_column
            ] = poster_url

        if rating:

            try:

                movies.at[
                    actual_index,
                    rating_column
                ] = float(rating)

            except ValueError:

                movies.at[
                    actual_index,
                    rating_column
                ] = ""

        else:

            movies.at[
                actual_index,
                rating_column
            ] = ""

        movies.to_csv(
            DATASET_PATH,
            index=False
        )

        if model_needs_rebuild:

            movies["_original_cast"] = (
                movies["cast"].copy()
            )

            movies["_original_crew"] = (
                movies["crew"].copy()
            )

            rebuild_recommendation_model()

        return redirect(
            url_for("admin_movies")
        )

    movie_data = {

        "title":
            row.get(
                "title",
                ""
            ),

        "genres":
            format_genres(
                row.get(
                    "genres",
                    ""
                )
            ),

        "overview":
            row.get(
                "overview",
                ""
            ),

        "poster_url":
            get_poster_from_row(row),

        "rating":
            get_rating_from_row(row)
    }

    return render_template(
        "admin_edit_movie.html",
        movie=movie_data,
        movie_index=movie_index,
        error=""
    )


# ============================================================
# ADMIN DELETE MOVIE
# ============================================================

@app.route(
    "/admin/movies/delete/<int:movie_index>",
    methods=["POST"]
)
def admin_delete_movie(movie_index):

    if not session.get(
        "admin_logged_in"
    ):

        return redirect(
            url_for("admin_login")
        )

    if (
        movie_index < 0
        or movie_index >= len(movies)
    ):

        return redirect(
            url_for("admin_movies")
        )

    movies.drop(
        movies.index[movie_index],
        inplace=True
    )

    movies.reset_index(
        drop=True,
        inplace=True
    )

    movies["_original_cast"] = (
        movies["cast"].copy()
    )

    movies["_original_crew"] = (
        movies["crew"].copy()
    )

    movies.to_csv(
        DATASET_PATH,
        index=False
    )

    rebuild_recommendation_model()

    return redirect(
        url_for("admin_movies")
    )

# ============================================================
# ADMIN USERS
# ============================================================

@app.route("/admin/users")
def admin_users():

    if not session.get("admin_logged_in"):
        return redirect(
            url_for("admin_login")
        )

    search = request.args.get(
        "search",
        ""
    ).strip()

    try:
        page = int(
            request.args.get(
                "page",
                1
            )
        )
    except ValueError:
        page = 1

    page = max(page, 1)

    per_page = 20

    conn = get_db_connection()

    if search:
        users = conn.execute(
            """
            SELECT
                id,
                name,
                email,
                created_at
            FROM users
            WHERE
                name LIKE ?
                OR email LIKE ?
            ORDER BY id DESC
            """,
            (
                f"%{search}%",
                f"%{search}%"
            )
        ).fetchall()

    else:
        users = conn.execute(
            """
            SELECT
                id,
                name,
                email,
                created_at
            FROM users
            ORDER BY id DESC
            """
        ).fetchall()

    total_users = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    conn.close()

    total_filtered = len(users)

    total_pages = max(
        1,
        (
            total_filtered
            + per_page
            - 1
        ) // per_page
    )

    page = min(
        page,
        total_pages
    )

    start = (
        page - 1
    ) * per_page

    end = start + per_page

    page_users = users[start:end]

    user_list = []

    for user in page_users:
        user_list.append({
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "created_at": user["created_at"],
            "status": "Active"
        })

    return render_template(
        "admin_users.html",
        users=user_list,
        total_users=total_users,
        active_users=total_users,
        search=search,
        page=page,
        per_page=per_page,
        total_filtered=total_filtered,
        total_pages=total_pages
    )


# ============================================================
# ADMIN DELETE USER
# ============================================================

@app.route(
    "/admin/users/delete/<int:user_id>",
    methods=["POST"]
)
def admin_delete_user(user_id):

    if not session.get("admin_logged_in"):
        return redirect(
            url_for("admin_login")
        )

    conn = get_db_connection()

    conn.execute(
        "DELETE FROM users WHERE id = ?",
        (user_id,)
    )

    conn.commit()
    conn.close()

    return redirect(
        url_for("admin_users")
    )

# ============================================================
# ADMIN RATINGS
# ============================================================

@app.route("/admin/ratings")
def admin_ratings():

    if not session.get("admin_logged_in"):
        return redirect(
            url_for("admin_login")
        )

    search = request.args.get(
        "search",
        ""
    ).strip()

    rating_movies = []

    if search:
        filtered_movies = movies[
            movies["title"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
        ].copy()

    else:
        filtered_movies = movies.copy()

    for index, row in filtered_movies.iterrows():

        rating = get_rating_from_row(row)

        if rating is None or str(rating).strip() == "":
            rating = "N/A"

        rating_movies.append({

            "title":
                str(row["title"]),

            "rating":
                rating,

            "poster_url":
                get_poster_from_row(row),

            "genres":
                format_genres(
                    row.get(
                        "genres",
                        "Movie"
                    )
                )
        })


    # --------------------------------------------------------
    # RATING STATISTICS
    # --------------------------------------------------------

    numeric_ratings = []

    for item in rating_movies:

        try:

            value = float(
                item["rating"]
            )

            numeric_ratings.append(
                value
            )

        except (
            ValueError,
            TypeError
        ):

            continue


    total_ratings = len(
        numeric_ratings
    )


    average_rating = 0

    if numeric_ratings:

        average_rating = round(
            sum(numeric_ratings)
            / len(numeric_ratings),
            1
        )


    return render_template(
        "admin_ratings.html",

        ratings=rating_movies,

        total_ratings=
            total_ratings,

        average_rating=
            average_rating,

        total_movies=
            len(rating_movies),

        search=search
    )

# ============================================================
# ADMIN DATASETS
# ============================================================

@app.route("/admin/datasets")
def admin_datasets():

    if not session.get("admin_logged_in"):
        return redirect(
            url_for("admin_login")
        )

    dataset_exists = os.path.exists(
        DATASET_PATH
    )

    total_rows = 0
    total_columns = 0
    file_size = 0
    columns = []
    preview = []

    if dataset_exists:

        try:

            dataset_df = pd.read_csv(
                DATASET_PATH
            )

            total_rows = len(
                dataset_df
            )

            total_columns = len(
                dataset_df.columns
            )

            columns = (
                dataset_df.columns.tolist()
            )

            file_size = round(
                os.path.getsize(
                    DATASET_PATH
                ) / (1024 * 1024),
                2
            )

            preview_df = (
                dataset_df
                .head(10)
                .fillna("")
                .astype(str)
            )

            preview = (
                preview_df.to_dict(
                    orient="records"
                )
            )

        except Exception as e:

            print(
                "Dataset error:",
                e
            )

    return render_template(
        "admin_datasets.html",
        dataset_exists=dataset_exists,
        total_rows=total_rows,
        total_columns=total_columns,
        file_size=file_size,
        columns=columns,
        preview=preview,
        dataset_path=
            "datasets/clean_movies_with_posters.csv"
    )
# ============================================================
# ADMIN AI FEATURES
# ============================================================

@app.route("/admin/ai-features")
def admin_ai_features():

    if not session.get("admin_logged_in"):
        return redirect(
            url_for("admin_login")
        )

    recommendation_engine = "Active"

    recommendation_method = (
        "Content-Based"
    )

    similarity_algorithm = (
        "Cosine Similarity"
    )

    model_status = "Ready"

    total_movies = len(movies)

    total_features = 5

    ai_features = [

        {
            "name":
                "Movie Recommendation",

            "description":
                "Recommends movies based on movie features and similarity.",

            "status":
                "Active"
        },

        {
            "name":
                "Content-Based Filtering",

            "description":
                "Uses genres, keywords, overview, cast and crew information.",

            "status":
                "Active"
        },

        {
            "name":
                "TF-IDF Feature Extraction",

            "description":
                "Converts movie text information into numerical feature vectors.",

            "status":
                "Active"
        },

        {
            "name":
                "Cosine Similarity",

            "description":
                "Measures similarity between movie feature vectors.",

            "status":
                "Active"
        },

        {
            "name":
                "Student Learning Assistant",

            "description":
                "Provides movie-related educational information for students.",

            "status":
                "Active"
        }

    ]

    return render_template(
        "admin_ai_features.html",

        recommendation_engine=
            recommendation_engine,

        recommendation_method=
            recommendation_method,

        similarity_algorithm=
            similarity_algorithm,

        model_status=
            model_status,

        total_movies=
            total_movies,

        total_features=
            total_features,

        ai_features=
            ai_features
    )
# ============================================================
# ADMIN DISCUSSIONS
# ============================================================

@app.route("/admin/discussions")
def admin_discussions():

    if not session.get(
        "admin_logged_in"
    ):

        return redirect(
            url_for("admin_login")
        )

    conn = get_db_connection()

    discussions = conn.execute(
        """
        SELECT *
        FROM discussions
        ORDER BY created_at DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
        "admin_discussions.html",
        discussions=discussions
    )


# ============================================================
# ADMIN DELETE DISCUSSION
# ============================================================

@app.route(
    "/admin/discussions/delete/<int:discussion_id>",
    methods=["POST"]
)
def admin_delete_discussion(discussion_id):

    if not session.get(
        "admin_logged_in"
    ):

        return redirect(
            url_for("admin_login")
        )

    conn = get_db_connection()

    conn.execute(
        """
        DELETE FROM discussion_replies
        WHERE discussion_id = ?
        """,
        (discussion_id,)
    )

    conn.execute(
        """
        DELETE FROM discussions
        WHERE id = ?
        """,
        (discussion_id,)
    )

    conn.commit()
    conn.close()

    return redirect(
        url_for("admin_discussions")
    )


# ============================================================
# ADMIN LOGOUT
# ============================================================

@app.route("/admin/logout")
def admin_logout():

    session.pop(
        "admin_logged_in",
        None
    )

    return redirect(
        url_for("admin_login")
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    try:
        return render_template("404.html"), 404
    except Exception:
        return "Page not found", 404


@app.errorhandler(500)
def internal_server_error(error):

    return (
        "Internal server error. "
        "Please check the Flask terminal."
    ), 500


# ============================================================
# RUN FLASK
# ============================================================
print("\n========== REGISTERED ROUTES ==========")

for rule in app.url_map.iter_rules():
    print(rule, "->", rule.endpoint)

print("=======================================\n")


if __name__ == "__main__":
    app.run(
        debug=True
    )