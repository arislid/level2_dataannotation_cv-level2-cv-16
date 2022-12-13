from torch import cuda

class Config:
    data_dir='../input/data/SimpleOCR_atom'
    use_val = True
    val_dir = '../input/data/ICDAR17_Korean'
    model_dir='trained_models'
    device='cuda' if cuda.is_available() else 'cpu'
    num_workers=8
    image_size=1024
    input_size=512
    batch_size=8
    learning_rate=1e-3
    max_epoch=100
    save_interval=5
    optimizer='Adam'
    early_stopping=5
    expr_name='SimpleOCR_atom_local'
    resume_from='' # 이어서 학습 할 .pth 경로 없으면 ''
    save_point=[38, 94] # 저장하고 싶은 epoch
    seed = 24