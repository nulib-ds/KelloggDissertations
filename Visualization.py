from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Example DataFrame with word frequencies for different years
result_path = r"/Users/hjr7324/Desktop/Kellogg_Dissertations"  #replace the path with your file folder
df = pd.read_csv(os.path.join(result_path, 'matrix_full.csv'))
df['Department'] = df['Department'].str.strip()
department = df['Department']
unique_classes = department.unique()
years = df['Year']
unique_year = years.sort_values(ascending=True).unique()
num_years = len(unique_year)

rows = 4
cols = (num_years + 1) // rows  # Calculate number of columns needed
# plot_type = 'wordcloud'
plot_type = 'bar' # wordcloud

for dep in unique_classes:
    # Create subplots with two rows
    fig, axs = plt.subplots(rows, cols, figsize=(36, 26)) # plot results in a same figure with 4 rows
    axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easy iteration
    dep_df = df[df['Department'] == dep]
    for i, year in enumerate(unique_year):
        year_df = dep_df[dep_df['Year'] == year]
        if len(year_df) == 0: # keep empty in the plot if no records in this year
            axs[i].axis('off')
        else:
            year_df.drop(['Year', 'GOID', 'Department'], axis=1, inplace=True)
            word_freq = year_df.sum().sort_values(ascending=False)

            if plot_type == 'wordcloud': # generate wordcloud for different department in different years
                wordcloud = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(word_freq)
                axs[i].imshow(wordcloud, interpolation='bilinear')
                axs[i].axis('off')
                plotname ='Word Cloud'
            elif plot_type == 'bar': # generate bar plot
                sns.barplot(x=word_freq.index[:20], y=word_freq.values[:20], ax=axs[i], palette='rocket')
                # axs[i].set_xlabel('Words')
                # axs[i].set_ylabel('Frequency')
                sns.despine(bottom = True, left = True) # remove borders in the subfig
                axs[i].tick_params(axis='x', rotation=45)
                plotname ='Barplot'

            axs[i].set_title(f'{year}', fontsize=16, fontweight='bold')

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    fig.suptitle(f'{plotname} of {dep}', fontsize=30, fontweight='bold')
    fig.tight_layout()
    plt.show()

    dep = dep.replace("/", " and ")  # "Management and Organizations/Sociology", / can't be part of the file name.
    fig.savefig(os.path.join(result_path, 'results', f'{plotname}_{dep}.png'), dpi=300)