import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
# ── Config (must match scripte.py) ──────────────────────────
VOCAB_SIZE = 3000
MAX_LEN    = 200
 
# ── Load word index ─────────────────────────────────────────
print("Loading word index...")
word_index = imdb.get_word_index()
 
# ── Load your trained model ─────────────────────────────────
# Option A: if you just ran scripte.py, the model is still in memory
# Option B: load from saved file (uncomment the line below)
# model = load_model('gru_sentiment_model.h5')
 
# ── Prediction function ─────────────────────────────────────
def predict_sentiment(model, review_text: str) -> dict:
    """
    Takes a raw text review and returns the sentiment + confidence.
    """
    # 1. Lowercase and split into words
    words = review_text.lower().split()
 
    # 2. Encode each word to its integer ID (unknown words → 2 = <UNK>)
    encoded = [word_index.get(w, 2) + 3 for w in words]
 
    # 3. Pad / truncate to MAX_LEN
    padded = pad_sequences([encoded], maxlen=MAX_LEN,
                            padding='post', truncating='post')
 
    # 4. Model prediction → probability between 0 and 1
    prob  = model.predict(padded, verbose=0)[0][0]
    label = "Positive ✅" if prob >= 0.5 else "Negative ❌"
 
    return {"label": label, "confidence": prob, "text": review_text}
 
 
def run_tests(model, reviews: list) -> None:
    """
    Runs prediction on a list of reviews and prints a formatted report.
    """
    print("\n" + "="*65)
    print("       SENTIMENT ANALYSIS — CUSTOM REVIEW TESTER")
    print("="*65)
 
    for i, review in enumerate(reviews, 1):
        result = predict_sentiment(model, review)
        conf   = result["confidence"]
        bar    = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
 
        print(f"\n[Review {i}]")
        print(f"  Text       : {result['text'][:70]}{'...' if len(result['text']) > 70 else ''}")
        print(f"  Sentiment  : {result['label']}")
        print(f"  Confidence : {bar}  {conf:.1%}")
        print(f"  Raw score  : {conf:.4f}  (>0.5 = Positive, <0.5 = Negative)")
 
    print("\n" + "="*65)
    print(f"  Total reviews tested: {len(reviews)}")
    print("="*65 + "\n")
 
 
# ── Your custom reviews — ADD / EDIT AS MANY AS YOU WANT ────
custom_reviews = [
 
    # --- Clearly Positive ---
    "This movie was absolutely fantastic! I loved every minute of it.",
    "One of the best films I have ever seen. Outstanding performances.",
    "A masterpiece of cinema. The story was moving and beautifully told.",
 
    # --- Clearly Negative ---
    "Terrible film. Boring plot and awful acting. Complete waste of time.",
    "I walked out after 30 minutes. Painfully bad script and direction.",
    "The worst movie of the year. No story, no emotion, nothing.",
 
    # --- Neutral / Mixed ---
    "It was okay, nothing special but not bad either.",
    "Some good moments but overall a bit disappointing.",
    "Decent enough for a lazy afternoon but I would not watch it again.",
 
    # --- Edge cases ---
    "Not bad at all actually.",                  # double negation
    "I can not say this was good.",              # negation
    "The film was not what I expected.",         # ambiguous
 
    # ────────────────────────────────────────────────────────
    # ADD YOUR OWN REVIEWS BELOW THIS LINE
    # Just copy the format:  "Your review text here.",
    # ────────────────────────────────────────────────────────
 
]

if __name__ == "__main__":
   model = load_model('gru_sentiment_model.h5')
   run_tests(model, custom_reviews)
