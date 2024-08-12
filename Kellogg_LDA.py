import pandas as pd
from gensim import corpora, models
import os

# package requirements
# conda install gensim
# conda install pyLDAvis
# conda install "scipy<1.13"

# Load the dataset
result_path = r"/Users/hjr7324/Desktop/Kellogg_Dissertations"  #replace the path with your file folder
df = pd.read_csv(os.path.join(result_path, 'matrix_full.csv'))
df.set_index('GOID', inplace=True)
# Save the year result
year = df['Year']
df.drop(['Year'], axis=1, inplace=True)

department = df['Department']
unique_classes = department.unique()

# Dictionary to hold LDA models for each department
lda_models = {}
dictionaries = {}
corpora_data = {}
# Iterate department results
for class_label in unique_classes:
    print(f'Processing {class_label}')
    dep_df = df[df['Department'] == class_label]
    # Extract words (column headers) and prepare the dictionary
    words = dep_df.columns[2:]  # Exclude the GOID and Department column
    dictionary = corpora.Dictionary()
    dictionary.add_documents([[word] for word in words])
    corpus = []
    for index, row in dep_df.iterrows():
        # Extract word frequencies and convert to (word_id, frequency) format
        document = [(dictionary.token2id[word], freq) for word, freq in zip(words, row[2:])]
        corpus.append(document)  # corpus is in the [(idx, frequency), (idx, frequency) ... ] format

    # Train the LDA model
    lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15,  random_state=42)
    lda_models[class_label] = lda_model
    dictionaries[class_label] = dictionary
    corpora_data[class_label] = corpus

    # Print topics
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)


# Display the topics for each department
with open(f'{result_path}/lda_topics.txt', 'w') as file:
    for cls, lda_model in lda_models.items():
        print(f"Department: {cls} \n")
        file.write(f'Department: {cls}:')
        for idx, topic in lda_model.print_topics(-1):
            file.write(f'Topic: {idx}\n')
            words_probs = topic.split(" + ")
            for wp in words_probs:
                prob, word = wp.split("*")
                word = word.strip('"')
                file.write(f"  {word}: {prob}\n")

            print(f'Topic: {idx} \nWords: {topic}')
            # file.write(f'Topic: {idx} \nWords: {topic}')
            file.write("\n---------------\n")
            file.write("\n")


import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
# Prepare visualizations
html_files = []
for cls, lda_model in lda_models.items():
    dictionary = dictionaries[cls]
    corpus = corpora_data[cls]
    lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)
    html_string = pyLDAvis.prepared_data_to_html(lda_vis)
    html_files.append((cls, html_string))

# Create the combined HTML file
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LDA Visualizations for Departments</title>
    <script src="https://cdn.jsdelivr.net/npm/pyldavis@3.0.0/lib/d3.v3.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/pyldavis@3.0.0/lib/ldavis.js"></script>
</head>
<body>
    {visualizations}
</body>
</html>
"""

visualizations = ""
for cls, html_string in html_files:
    visualization = f"""
    <h2>Visualization for {cls}</h2>
    {html_string}
    """
    visualizations += visualization

final_html = html_template.format(visualizations=visualizations)

# Save to a file
with open(f'{result_path}/LDA_vis.html', 'w') as file:
    file.write(final_html)
