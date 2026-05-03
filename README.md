# Congressional Vote Classifier
A binary neural network classifier that predicts whether a US congressman is a **Democrat or Republican** based on their voting record on 16 key political issues.


## Dataset
**Source:** [UCI Machine Learning Repository — Congressional Voting Records](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)

- 435 congressmen (267 Democrats, 168 Republicans)
- 16 vote columns (yes/no/abstain on policy issues)
- 1984 US House of Representatives data
- Loaded directly from URL — no download needed


## Project Structure
```
congressional-vote-classifier/
├── train.py              # Data loading, cleaning, model training, saving
├── test.py               # Load saved model and predict on new data
├── politics_model.keras  # Saved trained model
└── README.md
```

## How It Works

### Data Cleaning
- Missing votes represented as `?` are replaced with `np.nan`
- `y` (yes) votes are encoded as `1`, `n` (no) as `0`
- Missing values are filled with `0.5` — a neutral position representing abstention
- Labels: `democrat = 1`, `republican = 0`


### Model Architecture
```
Input (16 features)
→ Dense(16, relu)
→ Dense(8, relu)
→ Dense(1, sigmoid)
```

- Binary classification using `sigmoid` output
- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Train/test split: 80/20


### Results
- Training accuracy: **97.4%**
- Validation accuracy: **97.7%**
- No overfitting — validation accuracy matches training accuracy closely


## How to Run
Install dependencies:
```bash
pip install tensorflow pandas numpy
```

Train the model:
```bash
python train.py
```

Test with new data:
```bash
python test.py
```

## Example Prediction
```python
# Typical Democrat voting pattern
new_congressman = np.array([[1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1]], dtype='float32')

# Output
Raw prediction: 0.97027296
Predicted party: Democrat  (97% confident)
```

## Requirements
- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
