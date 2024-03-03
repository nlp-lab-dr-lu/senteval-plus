import numpy as np
import pandas as pd
from itertools import combinations
import logging

splitted_datasets = ["yelpp", "imdb", "agnews", "yelpf", "trec", "sstf", "mrpc", "mrpc1", "mrpc2"]
unsplitted_datasets = ["mr", "cr", "subj", "mpqa", "bbbp"]
similarity_datasets = ["sts1", "sts2"]

def save_embeddings(embeddings, dataset, model_name, dataset_name, split):
    logging.info('saving data embeddings')
    embeddings = np.array(embeddings)
    data = pd.concat([pd.DataFrame(dataset), pd.DataFrame(embeddings)], axis=1)
    if(dataset_name in ['sts1', 'sts2', 'mrpc1', 'mrpc2']):
        dataset_name = dataset_name[:-1]

    path = f'embeddings/{dataset_name}/{model_name}_{split}_embeddings.csv'
    # path = f'embeddings/sts/{self.model_name}_{split}_embeddings.csv'
    data.to_csv(path, sep='\t', index=False)

def get_layers_mean(model_name, dataset, embeddings_list):
    # numbers = list(range(1, 33))
    # all_tuples = list(combinations(numbers, 2))
    # datasets = ["sts1", "sts2"]
    # model_names = ["llama2-7B"]
    # for model_name in model_names:
    #     for dataset in datasets:
    #         for tpl in all_tuples:
    #             get_layers_mean(model_name, dataset, [tpl[0], tpl[1]])
    base_path = 'results/embeddings_sts/llama_layers'
    path1 = f'{base_path}/{model_name}-layer-{embeddings_list[0]}_{dataset}_test_embeddings.csv'
    path2 = f'{base_path}/{model_name}-layer-{embeddings_list[1]}_{dataset}_test_embeddings.csv'

    df1 = pd.read_csv(path1, sep='\t')
    df2 = pd.read_csv(path2, sep='\t')
    sentences = df1['text'] # save sentences for later
    df1 = df1.drop('text', axis=1)
    df2 = df2.drop('text', axis=1)

    average_df = pd.DataFrame((df1.values + df2.values) / 2)
    average_df = pd.concat([sentences, average_df], axis=1)
    average_df.to_csv(f'results/embeddings_sts/llama_pair/{model_name}-pair-{embeddings_list[0]}-and-{embeddings_list[1]}_{dataset}_test_embeddings.csv', index=False, sep='\t')
    print(f'saved average of {embeddings_list[0]} and {embeddings_list[1]} for {dataset}')
    return

def draw_cm(self, cm_data, dataset, whitening_method, encoder, download_font=False):
    if download_font:
        # add the font to Matplotlib
        # create the directory for fonts if it doesn't exist
        font_dir = "font/"
        os.makedirs(font_dir, exist_ok=True)

        # Download the font
        font_url = "http://foroogh.myweb.cs.uwindsor.ca/Times_New_Roman.ttf"
        font_path = os.path.join(font_dir, "times_new_roman.ttf")
        if not os.path.exists(font_path):
            os.system(f"wget -P {font_dir} {font_url}")

    font_files = font_manager.findSystemFonts(fontpaths=['font/'])
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    # Verify the font is recognized by Matplotlib
    font_name = "Times New Roman"
    if font_name in font_manager.get_font_names():
        print(f"'{font_name}' font successfully added.")
        # Set default font to Times New Roman
        matplotlib.rc('font', family=font_name)
    else:
        print(f"'{font_name}' font not found. Please check the font installation.")

    cm = confusion_matrix(cm_data['y_test'], cm_data['y_pred'], labels=np.unique(cm_data['y_test']))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(cm_data['y_test']))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_axis_off()
    ax.set_title(whitening_method.upper())
    plt.rcParams.update({'font.size': 30})
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    path = f'{self.RESULTS_PATH}/{dataset}_eval/cm/{encoder}-{whitening_method}-cm.pdf'
    print('svaing cm plot',path)
    fig.savefig(path, format='pdf')

def check_and_reorder_dataframe(df):
    # print('col names:',df.columns.tolist())
    if('SMILES' in df.columns.tolist()):
        df = df.rename(columns={'SMILES': 'text'}) 

    # Reorder DataFrame to ensure 'text' is the first column and 'label' is the second
    column_order = ['text', 'label'] + [col for col in df.columns if col not in ['text', 'label']]
    df = df[column_order]
    df['label'] = df['label'].astype(int)

    # Check if 'text' column exists and is of string type
    if df['text'].dtype != object:
        raise Exception("'text' column must be of string type")

    # Check if 'label' column exists and is of integer type
    if not pd.api.types.is_integer_dtype(df['label']):
        raise Exception("'label' column must be of integer type")

    # Check if the rest of the columns are float
    for col in df.columns[2:]:
        if not pd.api.types.is_float_dtype(df[col]):
            raise Exception(f"'{col}' column must be of float type")

    return df