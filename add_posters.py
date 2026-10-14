import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv

# Load API key
load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    print("ERROR: TMDB_API_KEY not found in .env")
    exit()

# Files
input_file = "datasets/clean_movies.csv"
output_file = "datasets/clean_movies_with_posters.csv"

# Load dataset
df = pd.read_csv(input_file)

print("======================================")
print("TMDB POSTER FETCHING")
print("======================================")
print("Total movies:", len(df))
print()

# Create poster_path column
poster_paths = []

session = requests.Session()

for index, movie_id in enumerate(df["movie_id"], start=1):

    print(
        f"[{index}/{len(df)}] "
        f"Fetching Movie ID: {movie_id}"
    )

    url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}"

    params = {
        "api_key": API_KEY,
        "language": "en-US"
    }

    poster_path = None

    try:

        response = session.get(
            url,
            params=params,
            timeout=10
        )

        if response.status_code == 200:

            data = response.json()

            poster_path = data.get("poster_path")

            if poster_path:
                print(
                    "   Poster found:",
                    poster_path
                )
            else:
                print(
                    "   No poster available"
                )

        elif response.status_code == 429:

            print(
                "   Rate limit reached. "
                "Waiting 10 seconds..."
            )

            time.sleep(10)

            response = session.get(
                url,
                params=params,
                timeout=10
            )

            if response.status_code == 200:

                data = response.json()
                poster_path = data.get(
                    "poster_path"
                )

                if poster_path:
                    print(
                        "   Poster found:",
                        poster_path
                    )
                else:
                    print(
                        "   No poster available"
                    )

            else:
                print(
                    "   Failed after retry:",
                    response.status_code
                )

        else:

            print(
                "   API Error:",
                response.status_code
            )

    except Exception as e:

        print(
            "   Request error:",
            e
        )

    poster_paths.append(poster_path)

    # Small delay to avoid excessive requests
    time.sleep(0.25)


# Add poster paths to dataset
df["poster_path"] = poster_paths


# Save as a NEW CSV
df.to_csv(
    output_file,
    index=False
)


# Final statistics
total_movies = len(df)
posters_found = df["poster_path"].notna().sum()
posters_missing = total_movies - posters_found

print()
print("======================================")
print("FULL POSTER FETCH COMPLETED")
print("======================================")
print("Total movies:", total_movies)
print("Posters found:", posters_found)
print("Posters missing:", posters_missing)
print()
print("New dataset:")
print(output_file)
print("======================================")