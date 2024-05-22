import os
import glob
import numpy as np
import pandas as pd
from itertools import combinations
import logging
import matplotlib
from matplotlib import font_manager
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('SentEval+')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

splitted_datasets = ["yelpp", "imdb", "agnews", "yelpf", "trec", "sstf", "mrpc", "mrpc1", "mrpc2"]
unsplitted_datasets = ["mr", "cr", "subj", "mpqa"]
similarity_datasets = ["stsb1", "stsb2", "sts121", "sts122", "sts131", "sts132", "sts141", "sts142", "sts151", "sts152", "sts161", "sts162"]
clustering_datasets = ["twentynewsgroups"]
bio_datasets = ["bbbp", "bace", "clintox",'sider','tox21','freesolv','lipo','delaney','hiv']

def save_embeddings(embeddings, dataset, model_name, dataset_name, split):
    logging.info('saving data embeddings')
    embeddings = np.array(embeddings)
    data = pd.concat([pd.DataFrame(dataset), pd.DataFrame(embeddings)], axis=1)
    if(dataset_name in ['mrpc1', 'mrpc2'] or dataset_name in similarity_datasets):
        dataset_name = dataset_name[:-1]

    path = f'embeddings/{dataset_name}/{model_name}_{split}_embeddings.csv'
    dir_path = f'embeddings/{dataset_name}/'
    if not os.path.exists(dir_path):
        # Create the directory
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' was created.")

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

def download_fonts():
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

def draw_cm(self, cm_data, dataset, whitening_method, encoder, download_font=False):
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

def sep_vecs_txtlbl(file):
    print(file)
    df = pd.read_csv(file, sep='\t')
    # df = df.drop(["Unnamed: 0.1", "Unnamed: 0"], axis=1)
    # df = df.sort_index(axis=1, ascending=False)
    vecs = df.iloc[:,2:]
    txt_labels = df.iloc[:,0:2]

    cols = vecs.columns.values.tolist()
    cols = [eval(i) for i in cols]
    cols.sort()
    cols = map(str, cols)
    vecs = vecs[cols]

    splited_path = file.split("/")

    vecs_path = splited_path[3].replace('embeddings', 'vecs')
    vecs.to_csv(f'./results/embeddings_cleaned/{vecs_path}', index=False, sep='\t')

    txt_labels_path = splited_path[3].replace('embeddings', 'txtlbl')
    txt_labels.to_csv(f'./results/embeddings_cleaned/{txt_labels_path}', index=False, sep='\t')

def sort_vecs(file):
    print(file)
    df = pd.read_csv(file, sep='\t')
    vecs = df.iloc[:,2:]
    txt_labels = df.iloc[:,0:2]

    cols = vecs.columns.values.tolist()
    cols = [eval(i) for i in cols]
    cols.sort()
    cols = map(str, cols)
    vecs = vecs[cols]
    df = pd.concat([txt_labels, vecs], axis=1)
    print(df.head())

    splited_path = file.split("/")

    df_path = splited_path[3]
    df.to_csv(f'./results/embeddings_cleaned/{df_path}', index=False, sep='\t')


def select_n_instances(file, n):
    df = pd.read_csv(file, sep='\t')
    print(file)
    print(df)
    per_class = np.round(n / df.label.nunique()).astype(int)
    print(per_class)
    df = df.groupby('label').head(per_class).sample(frac=1).reset_index(drop=True)

    splited_path = file.split("/")

    df_path = splited_path[3]
    df.to_csv(f'./results/embeddings/{df_path}', index=False, sep='\t')
    print(df)

def join_vecs_txtlbl(file_txtlbl, file_vecs):
    df_txtlbl = pd.read_csv(file_txtlbl, sep='\t')
    print(df_txtlbl)
    df_vecs = pd.read_csv(file_vecs, sep='\t')
    print(df_vecs)

    df = pd.concat([df_txtlbl, df_vecs], axis=1)
    df.to_csv(f'./results/embeddings/llama2-7B_yelpf_test_embeddings.csv', index=False, sep='\t')
    print(df)

# for file in glob.iglob('./results/embeddings/*.csv'):
#     sep_vecs_txtlbl(file)
    # sep_vecs_txtlbl(file)
    # sort_vecs(file)
    # select_n_instances(file, 4500)
    # join_vecs_txtlbl(file_txtlbl, file_vecs)