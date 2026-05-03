import pandas as pd
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('politics_model.keras')

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
df = df.replace({'y': 1, 'n': 0})
df['party'] = df['party'].replace({'democrat': 1, 'republican': 0})
df = df.fillna(0.5)

# Grab the first real Republican row
republican_row = df[df['party'] == 0].iloc[0]
votes = republican_row.drop('party').values.astype('float32').reshape(1, 16)

print("Actual votes:", votes)
prediction = model.predict(votes)
print("Raw prediction:", prediction[0][0])

if prediction[0][0] >= 0.5:
    print("Predicted party: Democrat")
else:
    print("Predicted party: Republican")
