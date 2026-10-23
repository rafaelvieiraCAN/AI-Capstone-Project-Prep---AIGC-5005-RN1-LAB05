# Late Deliveries
## Project Overview
This project predicts whether a food delivery order will be late using machine learning.
---

## Setup

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python -m latedeliveries.train
```

---

## Check It

```bash
pytest
```

```bash
ruff check .
```

---

## Decisions

### Features Used

- distance_km
- items
- hour
- weather
- day_type
- vehicle
- courier_months

### Features Excluded

- order_id
- prep_minutes
- customer_rating

Reason: potential data leakage and unavailable information at prediction time.

### Models Compared

- DummyClassifier
- Logistic Regression
- Decision Tree
- Random Forest

### Selected Model

Logistic Regression

### Best Hyperparameters

```text
C = 0.01
solver = liblinear
```

### Threshold Selection

Business Cost Rules:

- Flagged order = $4
- Missed late delivery = $15

Best Threshold:

```text
0.60
```

Best Cost:

```text
1431
```