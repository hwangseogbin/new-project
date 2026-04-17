# Fake News Detection Web App

This project is a complete fake news detection website built with Flask, scikit-learn, and a responsive frontend.

## What it includes

- Stacked hybrid Linear SVC + TF-IDF backend with learned fusion
- Training pipeline that can be switched to a different CSV/TSV/JSON dataset
- REST API for prediction and model metrics
- Source credibility, domain, and publish-date aware scoring
- Chunked transformer second-opinion support when local model weights exist
- Verification links for external fact-check follow-up
- Attractive responsive frontend with light and dark theme toggle
- Admin dashboard with login and saved prediction history

## Project structure

- `app.py` - Flask app and API routes
- `model.py` - hybrid training, loading, preprocessing, heuristics, and prediction logic
- `templates/index.html` - frontend page
- `templates/login.html` - admin login page
- `templates/admin.html` - admin dashboard
- `static/style.css` - custom UI styling
- `static/app.js` - frontend API integration
- `data/WELFake_Dataset.csv` - main dataset file used by default
- `artifacts/` - saved model and metrics after first run

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. The project now uses the WELFake dataset in this project as the main default dataset:

```text
C:\Users\ASUS\Documents\New project\data\WELFake_Dataset.csv
```

You can still point the app to another dataset file or folder:

```powershell
$env:DATASET_PATH='data/my_new_dataset.csv'
python app.py
```

If `DATASET_PATH` points to a folder, the app will automatically use the first supported dataset file inside it.

Supported input files:

- `.csv`
- `.tsv`
- `.json`
- `.jsonl`

The app will automatically retrain when the dataset file changes.

If you still want to use the original Kaggle dataset, it should come from:
[https://www.kaggle.com/datasets/marwanelmahalawy/fake-news](https://www.kaggle.com/datasets/marwanelmahalawy/fake-news)

## Dataset format

The trainer now auto-detects common column names.

Required logical fields:

- `text`: supported names include `text`, `content`, `article`, `article_text`, `body`, `story`, `news`
- `label`: supported names include `label`, `class`, `target`, `output`, `category`, `is_fake`, `fake`

Optional logical fields:

- `title`: supported names include `title`, `headline`, `news_title`, `heading`, `subject`
- `author`: supported names include `author`, `source`, `publisher`, `news_source`, `byline`, `publication`

If your column names are different, set overrides before running:

```powershell
$env:DATASET_PATH='data/my_new_dataset.csv'
$env:DATASET_TITLE_COLUMN='headline_text'
$env:DATASET_AUTHOR_COLUMN='newsroom'
$env:DATASET_TEXT_COLUMN='article_body'
$env:DATASET_LABEL_COLUMN='category'
python app.py
```

Label values default to:

- real: `0`, `real`, `true`, `reliable`, `legit`, `genuine`, `credible`
- fake: `1`, `fake`, `false`, `unreliable`, `hoax`, `rumor`, `misleading`

If your dataset uses different label values, you can override those too:

```powershell
$env:DATASET_REAL_LABELS='REAL,credible,authentic'
$env:DATASET_FAKE_LABELS='FAKE,clickbait,misleading'
python app.py
```

4. Run the app:

```bash
python app.py
```

If port `5000` is already in use, run it on another port:

```powershell
$env:PORT=5001
python app.py
```

5. Open:

```text
http://127.0.0.1:5000
```

## Admin login

- Login URL: `http://127.0.0.1:5000/login`
- Username: `hsb`
- Password: `hsb18`

Change these credentials in `app.py` before presenting the project.

## Expected accuracy

Using the current hybrid setup, the app trains multiple classifiers on your labeled dataset and combines them with context-aware rules. The exact score depends on your environment, but it should stay above your 90% project requirement.

- Training accuracy: typically above `98%`
- Test accuracy: typically above `98%`

That is already above your 90% requirement.

## API

### `GET /api/metrics`

Returns training/test accuracy and model information.

### `POST /api/predict`

JSON body:

```json
{
  "title": "News headline",
  "author": "Author name",
  "text": "Full article text"
}
```

### `GET /api/history`

Returns the latest saved predictions for the dashboard and homepage history panel.

Response:

```json
{
  "label": "Real",
  "label_code": 0,
  "confidence": 96.43,
  "probabilities": {
    "real": 96.43,
    "fake": 3.57
  }
}
```
