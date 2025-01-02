import optuna
from optuna.visualization import plot_pareto_front

# db_path = "sqlite:////Users/wenyidai/Development/graduation_projects/data_and_results_backup/search_mbv3_remote_tx2_gpu_240_360_480_600_720.db"
db_path = "sqlite:////Users/wenyidai/Development/graduation_projects/data_and_results_backup/search_resnet50_remote_tx2_gpu_240_360_480_600_720.db"
studies = optuna.study.get_all_study_summaries(storage=db_path)

# 打印所有study的名称
for study in studies:
    print(f"Study name: {study.study_name}")
# study = optuna.load_study(study_name="search_mbv3_w12_remote_tx2", storage=db_path)
study = optuna.load_study(study_name="search_resnet50_remote_tx2", storage=db_path)

fig = plot_pareto_front(study)
fig.show()
