import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, GRU, Dense,
                                     Dropout, Bidirectional)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import (classification_report,
                              confusion_matrix, roc_curve, auc)

print("TensorFlow version:", tf.__version__)

# ── 2. Hyperparameters ───────────────────────────────────────
VOCAB_SIZE   = 3000   # top N words to keep
MAX_LEN      = 200      # max review length (pad/truncate)
EMBED_DIM    = 64       # embedding vector size
GRU_UNITS    = 64       # neurons in GRU layer
DROPOUT_RATE = 0.3
BATCH_SIZE   = 128
EPOCHS       = 15

# ── 3. Load & Preprocess Data ────────────────────────────────
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)
print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Sample review    : {X_train[0][:10]}...")

# Pad sequences to fixed length
X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding='post',
                         truncating='post')
X_test  = pad_sequences(X_test,  maxlen=MAX_LEN, padding='post',
                         truncating='post')

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")


# ── 4. Decode a sample review (optional, for understanding) ──
word_index = imdb.get_word_index()
reverse_word_index = {v+3: k for k, v in word_index.items()}
reverse_word_index.update({0:'<PAD>', 1:'<START>',
                            2:'<UNK>', 3:'<UNUSED>'})

def decode_review(sequence):
    return ' '.join(reverse_word_index.get(i, '?')
                    for i in sequence if i != 0)

print("\nSample review decoded:")
print(decode_review(X_train[0]))
print(f"Label: {'Positive ✅' if y_train[0] == 1 else 'Negative ❌'}")

# ── 5. Build the GRU Model ───────────────────────────────────
model = Sequential([
    # Word embedding layer
    Embedding(input_dim=VOCAB_SIZE,
              output_dim=EMBED_DIM,
              input_length=MAX_LEN),

    # Bidirectional GRU (reads sequence forward AND backward)
    Bidirectional(GRU(GRU_UNITS,
                      return_sequences=True,  # stack another GRU
                      dropout=DROPOUT_RATE,
                      recurrent_dropout=0.1)),

    # Second GRU layer
    GRU(GRU_UNITS // 2,
        dropout=DROPOUT_RATE,
        recurrent_dropout=0.1),

    # Regularization
    Dropout(DROPOUT_RATE),

    # Output layer (binary classification)
    Dense(1, activation='sigmoid')
])

model.summary()

# ── 6. Compile ───────────────────────────────────────────────
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',   # binary since 2 classes
    metrics=['accuracy']
)

# ── 7. Callbacks ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=2, verbose=1)
]

# ── 8. Train ─────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
model.save('gru_sentiment_model.h5')
print("Model saved as gru_sentiment_model.h5 ✅")

# ── 9. Evaluate ──────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy : {test_acc*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ── 10. Predictions & Classification Report ──────────────────
y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Negative', 'Positive']))

# ── 11. Visualizations ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Accuracy curves
axes[0].plot(history.history['accuracy'],     label='Train Acc')
axes[0].plot(history.history['val_accuracy'], label='Val Acc')
axes[0].set_title('Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True)

# (b) Loss curves
axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True)

# (c) Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'], ax=axes[2])
axes[2].set_title('Confusion Matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('gru_results.png', dpi=150)
plt.show()

# (d) ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - GRU Sentiment Model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ── 12. Test on Custom Reviews ───────────────────────────────
def predict_sentiment(review_text: str) -> None:
    words   = review_text.lower().split()
    encoded = [word_index.get(w, 2) + 3 for w in words]
    padded  = pad_sequences([encoded], maxlen=MAX_LEN,
                             padding='post', truncating='post')
    prob    = model.predict(padded, verbose=0)[0][0]
    label   = "Positive ✅" if prob >= 0.5 else "Negative ❌"
    print(f"\nReview  : {review_text[:80]}...")
    print(f"Sentiment: {label} (confidence: {prob:.2%})")

predict_sentiment("This movie was absolutely fantastic! I loved every minute of it.")
predict_sentiment("Terrible film. Boring plot and awful acting. Complete waste of time.")
predict_sentiment("It was okay, nothing special but not bad either.")