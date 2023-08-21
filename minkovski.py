from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, ConceptsOptions, EmotionOptions, EntitiesOptions, KeywordsOptions, SemanticRolesOptions, SentimentOptions, CategoriesOptions, SyntaxOptions, SyntaxOptionsTokens,ClassificationsOptions
import numpy as np
from scipy.spatial.distance import minkowski

# Initialize IBM NLU
apikey = ''
url = ''

authenticator = IAMAuthenticator( apikey )
nlu = NaturalLanguageUnderstandingV1( version='2022-04-07', authenticator=authenticator )
nlu.set_service_url( url )

# Example email texts
email_texts = [
    "Hello, this is a positive email.",
    "This is a neutral email.",
    "I'm unhappy with your service."
]

# Analyze email texts with NLU and extract sentiment scores
sentiments = []
for email_text in email_texts:
    response = nlu.analyze(
        text=email_text,
        features=Features(sentiment=SentimentOptions())
    )
    sentiment_score = response.result['sentiment']['document']['score']
    sentiments.append(sentiment_score)

# Convert sentiment scores to numerical vectors
vectorized_features = np.array(sentiments).reshape(-1, 1)

# Calculate Minkowski distance between vectors
minkowski_distances = []
for i in range(len(vectorized_features)):
    for j in range(i + 1, len(vectorized_features)):
        distance = minkowski(vectorized_features[i], vectorized_features[j], p=2)  # Use p=2 for Euclidean distance
        minkowski_distances.append(distance)

print("Minkowski Distances:", minkowski_distances)