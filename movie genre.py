# Install required packages
!pip install pandas scikit-learn

# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 3: Read TXT file from Drive
file_path = '/content/drive/MyDrive/Codsoft_Movie Recommendation/test_data.txt'

numbers = []  # store movie numbers
titles = []
plots = []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(' ::: ')
        if len(parts) == 3:
            number, title, plot = parts
            numbers.append(int(number.strip()))
            titles.append(title.strip())
            plots.append(plot.strip())

print(f"✅ Total samples read: {len(plots)}")
print("\n📌 First title:", titles[0])
print("📌 First plot:", plots[0])

# Step 4: Create DataFrame & add dummy genres
dummy_genres = ['Drama', 'Comedy', 'Documentary', 'Action', 'Thriller']
# Repeat to match data length
dummy_genres = (dummy_genres * ((len(plots) // len(dummy_genres)) + 1))[:len(plots)]

df = pd.DataFrame({
    'number': numbers,
    'title': titles,
    'plot': plots,
    'genre': dummy_genres
})
print("\n📊 DataFrame sample:")
print(df.head())

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['plot'], df['genre'], test_size=0.3, random_state=42
)

# Step 6: Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 7: Train classifier
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test_tfidf)
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

# ✅ Step 9: Ask user for movie number and predict genre
while True:
    try:
        movie_number = int(input("\nEnter movie number to predict its genre (or -1 to exit): "))
        if movie_number == -1:
            print("Goodbye!")
            break

        # Find the plot by number
        row = df[df['number'] == movie_number]
        if row.empty:
            print("❌ Movie number not found.")
        else:
            plot_text = row.iloc[0]['plot']
            title_text = row.iloc[0]['title']
            plot_tfidf = vectorizer.transform([plot_text])
            predicted_genre = model.predict(plot_tfidf)[0]
            print(f"\n🎬 Title: {title_text}")
            print(f"📝 Plot: {plot_text[:100]}...")  # show first 100 chars
            print(f"✅ Predicted Genre: {predicted_genre}")
    except ValueError:
        print("⚠️ Please enter a valid number.")
