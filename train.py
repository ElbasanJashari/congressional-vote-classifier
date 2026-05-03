from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import pandas as pd
import numpy as np

columns = [
    'party',
    'handicapped_infants', 'water_project', 'budget_resolution',
    'physician_fee_freeze', 'el_salvador_aid', 'religious_groups_in_schools',
    'anti_satellite_ban', 'aid_to_contras', 'mx_missile',
    'immigration', 'synfuels_cutback', 'education_spending',
    'superfund_right_to_sue', 'crime', 'duty_free_exports',
    'export_admin_act_south_africa'
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"
df = pd.read_csv(url, header=None, names=columns)

df = df.replace('?', np.nan)
df = df.replace({"y": 1, "n": 0})
df["party"] = df["party"].replace({"democrat": 1, "republican": 0})
df = df.fillna(0.5)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.drop('party', axis=1).values.astype('float32')
y = df['party'].values.astype('float32')

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

model = Sequential([
    Dense(16, activation='relu', input_shape=(16,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)

model.save('politics_model.keras')
print("Model saved!")
