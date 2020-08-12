import time

start = time.time()

# Import modules
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import re
import pickle

end = time.time()
print(end - start)

########################################################################################################################
start = time.time()

# Read in the data
df = pd.read_csv('data/reviewers_data.csv')
df2 = pd.read_csv('data/hotel_reviews.csv')

# Important columns. City, Country, and Province columns are excluded because they are not reliable
imp_col_list = ['address', 'name', 'reviews.date', 'reviews.text', 'reviews.title']
df = df.loc[:, imp_col_list]
df2 = df2.loc[:, imp_col_list]
df = df.append(df2)

end = time.time()
print(end - start)

########################################################################################################################
start = time.time()

# Prep text, add some columns and fillna, and rename columns
df['reviews.text'] = df['reviews.text'].str.lower()
df['reviews.text'] = df['reviews.text'].replace(to_replace='[^A-Za-z0-9]+', regex=True, value=' ')
df['reviews.text'] = df['reviews.text'].fillna('')
df['review_date'] = pd.to_datetime(df['reviews.date']).dt.date
df['review_month'] = pd.to_datetime(df['reviews.date']).dt.month
df['words_in_review'] = [len(i.split()) for i in df['reviews.text']]
season_dict = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer',
               7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
df['review_season'] = df['review_month'].map(season_dict).fillna('Summer')
df.rename(columns={'address': 'hotel_address', 'city': 'hotel_city', 'country': 'hotel_country',
                   'name': 'hotel_name'}, inplace=True)

end = time.time()
print(end - start)
########################################################################################################################
print(df.head())
print(df.describe())
print(df.memory_usage().sum()/1024/1024)
print(df.shape, df.isnull().sum())
########################################################################################################################

vectorizer = CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 1))
X = vectorizer.fit_transform(df['reviews.text'])
bag_of_words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
full_df = df.join(bag_of_words)
print(full_df.head())
#full_df.to_csv(r'vectors.csv')
print(full_df.memory_usage().sum()/1024/1024)
########################################################################################################################

X = bag_of_words
y = df['hotel_name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)

# Import the random forest model classifier.
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(min_samples_leaf=10, random_state=8675309)
import time
start = time.time()
# Fit the model to the data.
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy score: ",round((accuracy_score(y_test, y_pred)*100),2), "%")

end=time.time()
print(end-start)
########################################################################################################################

test_review = 'I loved the beach, the nearby bars, the live music, and the walkable neighborhood. The weather was great and it was sunny.'

est_review = test_review.lower()
test_review = re.sub('[^A-Za-z0-9]+', ' ', test_review)
test_review = [test_review]

X_test = vectorizer.transform(test_review).toarray()

prediction = clf.predict(X_test)[0]
print(df[df['hotel_name'] == prediction][['hotel_name', 'hotel_address', 'review_date','review_month','review_season']].head(15))
