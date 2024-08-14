import pandas as pd
from bertopic import BERTopic
from datetime import datetime
from sentence_transformers import SentenceTransformer
import os
import plotly.io as pio
from umap import UMAP

# Get the current time in UTC
time_start = datetime.now()

folder_path = r"/Users/hjr7324/Desktop/Kellogg_Dissertations"
if not os.path.exists(folder_path + '/results'): # create a results folder
    os.mkdir(folder_path + '/results')

# Load the CSV file
df = pd.read_csv(os.path.join(folder_path, 'matrix_full.csv'))
df['Department'] = df['Department'].str.strip()
df.set_index('GOID', inplace=True)
year = df['Year']
df.drop(['Year'], axis=1, inplace=True)
department = df['Department']
unique_classes = department.unique()

documents = []
for _, row in df.iloc[:, 1:].iterrows():
    doc = ' '.join([f"{word} " * freq for word, freq in row.items() if freq > 0])
    documents.append(doc)

bertopic_models = {}
# Save the topics to a text file
from plotly.subplots import make_subplots
visualizations = []

with open(os.path.join(folder_path, f'results/bertopic_topics.txt'), 'w') as file:
    for class_label in unique_classes:
        print(f'Processing {class_label}')
        dep_df = df[df['Department'] == class_label]
        # document_ids = dep_df['ID'].tolist()
        documents = []
        for _, row in df.iloc[:, 1:].iterrows():
            doc = ' '.join([f"{word} " * freq for word, freq in row.items() if freq > 0])
            documents.append(doc.strip())
        # Create a dictionary to map document IDs to their texts
        # document_texts = dict(zip(document_ids, documents))
        # vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
        # Load pre-trained Sentence Transformer model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Generate embeddings for the documents
        # embeddings = model.encode(class_sub.tolist(), show_progress_bar=True)
        embeddings = model.encode(documents, show_progress_bar=True)

        # Initialize the BERTopic model
        topic_model = BERTopic(
            # vectorizer_model=vectorizer_model,
                               language='english', calculate_probabilities=True,
                               verbose=True)
        # Fit the model to your text data
        topics, probs = topic_model.fit_transform(documents, embeddings)
        # Get the topics
        topic_info = topic_model.get_topic_info()

        # print(f"Subreddit: {cls}")
        file.write(f'Department: {class_label}:\n')
        for index, row in topic_info.iterrows():
            file.write(f"Topic {row['Topic']} - Count {row['Count']} : {row['Name']}\n")
            topic_words = topic_model.get_topic(row['Topic'])
            # topic_model.visualize_barchart()
            for word, _ in topic_words:
                file.write(f"  {word}: {_}\n")
            file.write("\n---------------\n")
            file.write("\n")

        plotly_fig = topic_model.visualize_barchart()
        # plotly_fig = topic_model.visualize_documents(documents, embeddings=embeddings)
        # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
        # plotly_fig = topic_model.visualize_documents(class_sub, reduced_embeddings=reduced_embeddings)
        # plotly_fig.update_layout({class_label})
        # fig.update_layout(title_text=f"Topic Model for Submissions - {class_label}")
        html_str = pio.to_html(plotly_fig, full_html=False)
        visualizations.append([class_label, html_str])
        # fig.show()
        # fig.write_html(f"{token_path}/bertopic_sub_{class_label}.html")
        # fig.write_html(f"{token_path}/bertopic_submission.html")


html_content = """
<!DOCTYPE html>
<html>
<head>
    <style>
        .container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }}
        .item {{
            flex: 1;
            min-width: 300px;
            max-width: 500px;
        }}
    </style>
</head>
<body>
    <div class="container">
"""


# Embed each visualization
for cls, html_string in visualizations:
    html_content += f"""
    <h2>BERTopic Visualization for {cls}</h2>
    {html_string}
    """

html_content += """
    </div>
</body>
</html>
"""

# Save the combined HTML file
with open(f"{folder_path}/results/BERTopic_vis_bar.html", "w") as f:
    f.write(html_content)

print(f"BERTopic topics saved to bertopic_topics.txt")

# Get the current time in UTC
time_now = datetime.now()
print(f"Execution end: {time_now}")
print(f"Time taken: {time_now - time_start}")
