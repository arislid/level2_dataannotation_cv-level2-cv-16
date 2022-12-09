from torch import cuda

class Config:
    data_dir='../input/data/ICDAR19_MIXPASS'
    model_dir='trained_models'
    device='cuda' if cuda.is_available() else 'cpu'
    num_workers=4
    image_size=1024
    input_size=512
    batch_size=16
    learning_rate=1e-3
    max_epoch=100
    save_interval=5
    optimizer='Adam'
    early_stopping=5
    expr_name='icdar19_aug_1_continue'
    resume_from='./trained_models/best_mean_loss.pth' # 이어서 학습 할 .pth 경로 없으면 ''
    save_point=[38, 94] # 저장하고 싶은 epoch