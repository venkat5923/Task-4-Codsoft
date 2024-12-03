# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Create a sample movie dataset with new movies and additional details
data = {
    "movie_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "title": [
        "The Matrix", "Jurassic Park", "The Godfather", "Gladiator", "The Shawshank Redemption", 
        "Pulp Fiction", "Fight Club", "The Lion King", "Forrest Gump", "The Dark Knight Rises", 
        "Spider-Man: No Way Home", "The Lord of the Rings: The Fellowship of the Ring", "Inception", 
        "Interstellar", "The Prestige"
    ],
    "genre": [
        "Sci-Fi", "Adventure", "Crime", "Action", "Drama", 
        "Crime", "Drama", "Animation", "Drama", "Action", 
        "Action", "Adventure", "Sci-Fi", "Sci-Fi", "Drama"
    ],
    "description": [
        "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
        "A group of scientists at a remote island research facility must survive after dinosaurs are resurrected through genetic engineering.",
        "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
        "A betrayed Roman general seeks revenge against the corrupt emperor who murdered his family and sent him into slavery.",
        "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
        "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
        "An insomniac office worker and a soap salesman form an underground fight club.",
        "Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself.",
        "A man with a low IQ has accomplished great things in his life and been present during significant historical events, but his true love eludes him.",
        "Eight years after the Joker's reign of anarchy, the Dark Knight resurfaces to protect Gotham City from the new terrorist threat.",
        "Spider-Man faces a new challenge as he works with other superheroes to battle a multiverse-level threat.",
        "A young hobbit embarks on a dangerous quest to destroy a powerful ring that threatens to rule the world.",
        "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea.",
        "A team of explorers travel through a wormhole in space to ensure humanity's survival.",
        "A magician searches for the secret of a perfect illusion that has a deadly consequence."
    ],
    "rating": [8.7, 8.1, 9.2, 8.5, 9.3, 8.9, 8.8, 8.5, 8.8, 8.4, 8.3, 8.8, 8.8, 8.6, 8.5],
    "release_year": [1999, 1993, 1972, 2000, 1994, 1994, 1999, 1994, 1994, 2012, 2021, 2001, 2010, 2014, 2006]
}

# Step 2: Convert the dataset to a DataFrame
df = pd.DataFrame(data)

# Step 3: Combine important features into a single string
df["combined_features"] = df["genre"] + " " + df["description"]

# Step 4: Transform text into numeric features using TF-IDF
tfidf = TfidfVectorizer(stop_words="english")  # Remove common words (stop words)
tfidf_matrix = tfidf.fit_transform(df["combined_features"])  # Generate TF-IDF matrix

# Step 5: Compute pairwise cosine similarity between movies
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 6: Define a recommendation function
def recommend_content_based(movie_title, df, similarity_matrix, num_recommendations=3):
    """
    Recommend movies based on content similarity and display movie details.

    Args:
    - movie_title (str): Title of the movie for recommendations.
    - df (DataFrame): DataFrame containing movie details.
    - similarity_matrix (array): Precomputed cosine similarity matrix.
    - num_recommendations (int): Number of recommendations to return.

    Returns:
    - Details of the input movie and a list of recommended movie titles with details.
    """
    try:
        # Find the index of the movie in the DataFrame
        movie_index = df[df["title"].str.lower() == movie_title.lower()].index[0]

        # Extract details of the selected movie
        selected_movie_details = df.iloc[movie_index]

        # Get similarity scores for the target movie
        similarity_scores = list(enumerate(similarity_matrix[movie_index]))

        # Sort movies by similarity score (excluding the input movie itself)
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]

        # Extract details of recommended movies
        recommended_movies = [df.iloc[i[0]][["title", "rating", "release_year"]] for i in sorted_scores]

        return selected_movie_details, recommended_movies
    except IndexError:
        return None, ["Movie not found! Please check the title."]

# Step 7: Interactive user input
print("\n🎥 Welcome to the Movie Recommendation System!")
movie_to_search = input("Enter the name of a movie you like: ")

# Fetch recommendations
selected_movie, recommendations = recommend_content_based(movie_to_search, df, similarity_matrix)

# Step 8: Display the results
if selected_movie is not None:
    print(f"\n🎬 **Selected Movie**: {selected_movie['title']}")
    print(f"   - Genre: {selected_movie['genre']}")
    print(f"   - Rating: {selected_movie['rating']}")
    print(f"   - Release Year: {selected_movie['release_year']}")
    print(f"   - Description: {selected_movie['description']}\n")

    print(f"✨ **Top {len(recommendations)} Recommendations**:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['title']} (Rating: {rec['rating']}, Year: {rec['release_year']})")
else:
    print("❌ Movie not found! Please check your spelling or try another movie.")
