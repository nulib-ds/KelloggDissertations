import re
import pandas as pd
import plotly.io as pio
import os
from bertopic import BERTopic
import umap
import hdbscan
import re
import pandas as pd

# Prepare data
folder_path = r"/Users/hjr7324/Desktop/Kellogg_Dissertations"
if not os.path.exists(folder_path + '/results'): # create a results folder
    os.mkdir(folder_path + '/results')

# Load the CSV file
df = pd.read_csv(os.path.join(folder_path, 'matrix_full.csv'))
df['Department'] = df['Department'].str.strip()
df.set_index('GOID', inplace=True)
year = df['Year'].tolist()
df.drop(['Year'], axis=1, inplace=True)
department = df['Department']
unique_classes = department.unique()

documents = []
for _, row in df.iloc[:, 1:].iterrows():
    doc = ' '.join([f"{word} " * freq for word, freq in row.items() if freq > 0])
    documents.append(doc)

# # Create UMAP and HDBSCAN with a fixed random seed
# umap_model = umap.UMAP()
# hdbscan_model = hdbscan.HDBSCAN()

# Create BERTopic model with custom UMAP and HDBSCAN
topic_model = BERTopic(verbose=True)

topics, probs = topic_model.fit_transform(documents)

topics_over_time = topic_model.topics_over_time(documents, year, global_tuning=True, evolution_tuning=True)

plotly_fig = topic_model.visualize_topics_over_time(topics_over_time)

pio.show(plotly_fig)
# Save the Plotly figure as an HTML file
pio.write_html(plotly_fig, f'{folder_path}/results/topic_overtime.html')
