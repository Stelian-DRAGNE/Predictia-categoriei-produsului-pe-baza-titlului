
import argparse
import pandas as pd
import joblib           # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score



def feature_frame_from_titles(titles: pd.Series) -> pd.DataFrame:
    s = titles.fillna("")
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


def build_pipeline():
    text_vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9, strip_accents="unicode", lowercase=True)
    feat_transformer = FunctionTransformer(ft_transform, validate=False)
    preprocess = ColumnTransformer([
        ("tfidf", text_vectorizer, "Product Title"),
        ("num", Pipeline([("eng", feat_transformer), ("scale", MinMaxScaler())]), "Product Title")
    ])
    model = LinearSVC(class_weight="balanced", random_state=42, max_iter=5000)
    return Pipeline([("preprocess", preprocess), ("clf", model)])


def main():
    parser = argparse.ArgumentParser(description="Train product title -> category classifier")
    parser.add_argument("--csv", type=str, default="products.csv", help="Path to products.csv")
    parser.add_argument("--out", type=str, default="model_product_category.pkl", help="Path to save trained model (.pkl)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Category Label":"Category_Label"})
    df = df.dropna(subset=["Product Title","Category_Label"]).copy()

    X_train, X_test, y_train, y_test = train_test_split(df[["Product Title"]], df["Category_Label"], test_size=0.2, random_state=42, stratify=df["Category_Label"])

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(pipe, args.out)
    print(f"Saved model to: {args.out}")



if __name__ == "__main__":
    main()
