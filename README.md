# Predictia categoriei produsului pe baza titlului

## Predicția categoriei produsulu pe baza titlului

## 🔍 Contextul și obiectivul
Când un produs nou este listat pe platformă, trebuie atribuită categoria corectă 
(ex. „Mobile Phones”, „Laptops”, „Washing Machines”). 
Până acum, această clasificare se făcea manual – un proces lent, predispus la erori 
și greu de scalat pentru zeci de mii de produse.

Scopul acestui proiect este să dezvoltăm un **model de învățare automată** 
care, pe baza titlului produsului, poate prezice automat categoria potrivită.
Astfel:
- accelerăm procesul de introducere a produselor,
- reducem erorile umane,
- îmbunătățim experiența utilizatorilor la căutare și filtrare.

---

## 🛠️ Toolbox și date
- **Set de date:** `products.csv` (peste 30.000 de produse)
  - `Product Title` (titlul produsului)  
  - `Category Label` (categoria țintă)  
  - + alte coloane: Merchant ID, Product Code, Views, Rating, Listing Date  

- **Biblioteci:** `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`

---

## 📊 Abordarea
1. **Explorarea datelor** – am analizat distribuția categoriilor, curățat valorile lipsă și standardizat anteturile.
2. **Feature engineering** – pe lângă titlul brut, am adăugat:
   - număr de caractere,
   - număr de cuvinte,
   - prezența cifrelor,
   - prezența token-urilor cu majuscule (USB, LED),
   - lungimea celui mai lung cuvânt.
3. **Vectorizare text** – `TfidfVectorizer` cu n-grame (1–2).
4. **Model** – `LinearSVC` cu `class_weight="balanced"`.
5. **Evaluare** – `accuracy`, `precision/recall/F1`, matrice de confuzie.

Rezultate:
- Accuracy pe setul de test: **~95%**
- Matricea de confuzie arată că modelul este foarte bun pe categoriile majore, dar poate confunda produse similare (ex. „Fridges” vs „Dishwashers”).

---

## 📂 Structura proiectului
```
.
├── train_model.py               # script pentru antrenare model
├── predict_category.py          # script interactiv pentru testare
├── model_product_category.pkl   # modelul antrenat (salvat)
├── products.csv                 # setul de date
├── evaluation_report.txt        # raport de evaluare
├── confusion_matrix_top15.png   # matrice confuzie (vizual)
├── README.md                    # documentația proiectului
└── notebooks/
    └── 01_exploration_training.ipynb  # analiza completă
```

---

## 🚀 Cum rulezi proiectul

### 1. Instalează dependențele
```bash
pip install scikit-learn pandas joblib matplotlib seaborn
```

### 2. Antrenează modelul
```bash
python train_model.py --csv products.csv --out model_product_category.pkl
```

### 3. Testează interactiv
```bash
python predict_category.py --model model_product_category.pkl
```
Apoi introdu titluri precum:
- `iphone 7 32gb gold`
- `olympus e m10 mark iii`
- `bosch wap28390gb 8kg 1400 spin`

### 4. Vezi rezultatele
- Accuracy și raportul: în `evaluation_report.txt`
- Matrice confuzie: `confusion_matrix_top15.png`

---

## 🌱 Ce putem îmbunătăți
- Adăugarea lematizării/stemming pentru titluri.
- Testarea altor algoritmi (Logistic Regression, Random Forest).
- Folosirea și a altor coloane (ex. `Merchant Rating`, `Number of Views`).
- Crearea unei interfețe web simple pentru predicții.

---

## 📌 Concluzie
Acest proiect arată cum putem rezolva o provocare reală de business cu ML: 
automatizarea clasificării produselor.  
Fluxul este complet, documentat și pregătit pentru a fi extins de echipă.  
