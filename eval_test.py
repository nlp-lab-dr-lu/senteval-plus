from eval.eval_cls import Evaluation
from eval.eval_cls_whitening import WhiteningEvaluation
# from eval.eval_mrpc import Evaluation
# from eval.eval_sts import Evaluation
from eval.clustering import ClusteringEvaluation

datasets = ["mrpc"]
classifiers = ["mlp", "rf", "svm", "nb"]
EMBEDDINGS_PATH = 'embeddings/'
RESULTS_PATH = 'results' # where you want to save the results of evaluation

# config = {
#     'EMBEDDINGS_PATH': EMBEDDINGS_PATH,
#     'RESULTS_PATH': RESULTS_PATH,
#     'drawcm': False,
#     'classifier': 'svm',
#     'kfold': 5,
#     'datasets': datasets,
#     'encoders': ["text-embedding-3-small"]
# }

# # eval = Evaluation(config)
# eval = WhiteningEvaluation(config)
# eval.run()


config = {
    'EMBEDDINGS_PATH': EMBEDDINGS_PATH,
    'RESULTS_PATH': RESULTS_PATH,
    'eval_whitening': True,
    'datasets': datasets,
}
eval = ClusteringEvaluation(config)
eval.run()

# config = {
#     'EMBEDDINGS_PATH': EMBEDDINGS_PATH,
#     'SCORES_PATH': 'data',
#     'RESULTS_PATH': RESULTS_PATH,
#     'eval_whitening': True,
#     'datasets': ["twentynewsgroups"],
#     # 'encoders': encoders
# }

# config = {
#     'EMBEDDINGS_PATH': EMBEDDINGS_PATH,
#     'RESULTS_PATH': RESULTS_PATH,
#     'drawcm': False,
#     'classifier': 'mlp',
#     'kfold': 5,
#     'encoders': encoders,
#     'datasets': datasets
# }

