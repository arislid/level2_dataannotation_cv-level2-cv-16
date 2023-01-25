from torch import cuda

class Config:
    data_dir='../input/data/ICDAR19_MIXPASS'
    use_val = True
    val_dir = '../input/data/ICDAR17_MIXPASS'
    model_dir='trained_models'
    device='cuda' if cuda.is_available() else 'cpu'
    num_workers=8
    image_size=1024
    input_size=512
    batch_size=16
    learning_rate=1e-3
    max_epoch=200
    save_interval=1
    optimizer='Adam'
    early_stopping=100
    expr_name='icdar19_crop_1_continue_val_2'
    resume_from='./trained_models/real-latest.pth' # 이어서 학습 할 .pth 경로 없으면 ''
    save_point=[50, 75, 100, 125, 150, 175, 200] # 저장하고 싶은 epoch
    seed = 42