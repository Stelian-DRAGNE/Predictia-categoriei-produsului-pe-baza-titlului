

import argparse
import joblib               # type: ignore
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

# -------------------------------
# Funcții auxiliare (identice cu cele folosite la antrenare)
# -------------------------------
def feature_frame_from_titles(titles: pd.Series):
    s = pd.Series(titles).fillna("")
    num_chars = s.str.len().astype(float)
    num_words = s.str.split().str.len().astype(float)
    has_digits = s.str.contains(r"\d").astype(int)
    has_caps_token = s.str.contains(r"\b[A-Z]{2,}\b").astype(int)
    longest_word = s.str.split().apply(lambda toks: max((len(t) for t in toks), default=0)).astype(float)
    return pd.DataFrame({
        "num_chars": num_chars,
        "num_words": num_words,
        "has_digits": has_digits,
        "has_caps_token": has_caps_token,
        "longest_word": longest_word
    })

def ft_transform(X):
    return feature_frame_from_titles(pd.Series(X))

# -------------------------------
# Script interactiv
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Interactive product category prediction")
    parser.add_argument("--model", type=str, default="model_product_category.pkl", help="Path to trained model .pkl")
    args = parser.parse_args()

    # Încarcă modelul
    pipe = joblib.load(args.model)
    print("✅ Model loaded. Type a product title (or 'quit' to exit).")

    # Loop interactiv
    while True:
        try:
            title = input("> ").strip()
        except EOFError:
            break
        if not title:
            continue
        if title.lower() in {"q", "quit", "exit"}:
            break

        # Pregătim inputul ca DataFrame (coloană Product Title)
        X = pd.DataFrame({"Product Title": [title]})
        pred = pipe.predict(X)
        print(f"Predicted category: {pred[0]}")

if __name__ == "__main__":
    main()
