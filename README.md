# Real-and-Fake-News-Detection - Data Mining Project

This repository presents a comprehensive pipeline for detecting fake news using machine learning and deep learning techniques on a dataset of over 6,000 news articles.

## 📌 Project Goals
- Classify news articles as real or fake
- Apply both classical ML and modern DL models
- Compare TF-IDF vs Word2Vec representations
- Evaluate model performance with precision, recall, and F1-score

## 📊 Dataset
- **Shape:** 6,335 rows × 4 columns
- **Features:** title, text, subject, label (target)

## 🛠️ Techniques Used
- **Text Cleaning & Preprocessing**:
  - Stopword removal, lemmatization, punctuation removal
  - HTML tag removal, special character filtering
  - Language detection and duplicate handling

- **EDA**:
  - Sentiment Analysis, Title/Text Length Analysis
  - Named Entity Recognition (NER), Co-occurrence graphs
  - Topic Modeling (LDA)

- **Feature Engineering**:
  - **TF-IDF Vectorization**
  - **Word2Vec Embeddings**

## 🧪 Models Implemented
| Model | Representation | Accuracy |
|-------|----------------|----------|
| Logistic Regression | TF-IDF | 93% |
| Random Forest | TF-IDF | 92% |
| MLP (Deep Learning) | TF-IDF | **94%** |
| GRU / RNN / LSTM | TF-IDF | Up to 93% |
| Transformers | Word2Vec | 92% |

## 🔮 Results
- MLP (TF-IDF) achieved the best performance: **94% F1 Score**
- Transformer-based models showed strong contextual understanding
- Word2Vec performed better in DL models compared to classical ones

## 🚀 Future Enhancements
- Real-time news scraping and classification
- Domain-specific transformer fine-tuning
- Advanced interpretability tools (e.g., LIME, SHAP)

## 👨‍🏫 Credits
- Developed under the guidance of Prof. Roberta Siciliano and Prof. Giuseppe Longo
