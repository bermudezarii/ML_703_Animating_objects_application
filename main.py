import os
from model.Classifier.code.inference_1image import make_prediction

def get_class(pth):
    res = make_prediction(pth)
    class_name = 'Rotation' if res == 1 else "Periodic"
    print(class_name)
    return res

#FOMM model
def train_FOMM_rotation():
    os.system('CUDA_VISIBLE_DEVICES=0 python model/first-order-model/run.py --config model/first-order-model/config/our_motion_ds_256_rotation.yaml --device_ids 0 --checkpoint model/first-order-model/checkpoints/rotation_00001450-checkpoint.pth.tar')

def train_FOMM_periodic():
    os.system('CUDA_VISIBLE_DEVICES=0 python model/first-order-model/run.py --config model/first-order-model/config/our_motion_ds_256_periodic.yaml --device_ids 0 --checkpoint model/first-order-model/checkpoints/periodic_00001450-checkpoint.pth.tar')

def generate_FOMM_rotation(imgpath = 'assets/Rotation/rotation_1.jpeg'):
    command = 'CUDA_VISIBLE_DEVICES=0 python model/first-order-model/demo.py --config model/first-order-model/config/our_motion_ds_256_rotation.yaml --checkpoint model/first-order-model/checkpoints/rotation_00001450-checkpoint.pth.tar  --source_image ' + imgpath + ' --driving_video assets/Rotation/rotation_pumpkin.mp4'
    os.system(command)

def generate_FOMM_periodic(imgpath = 'assets/Periodic/periodic_1.jpeg'):
    command = 'CUDA_VISIBLE_DEVICES=0 python model/first-order-model/demo.py --config model/first-order-model/config/our_motion_ds_256_periodic.yaml --checkpoint model/first-order-model/checkpoints/periodic_00001450-checkpoint.pth.tar --source_image ' + imgpath + ' --driving_video assets/Periodic/periodic_metronome.mp4'
    os.system(command)

#TPSM model
def train_TPSM_rotation():
    os.system('CUDA_VISIBLE_DEVICES=0 python model/thin-plate-spline-motion-model/run.py --config model/thin-plate-spline-motion-model/config/tps_rotation.yaml --device_ids 0 --checkpoint model/thin-plate-spline-motion-model/checkpoint/rotation-00000099-epoch.pth.tar')

def train_TPSM_periodic():
    os.system('CUDA_VISIBLE_DEVICES=0 python model/thin-plate-spline-motion-model/run.py --config model/thin-plate-spline-motion-model/config/tps_periodic.yaml --device_ids 0 --checkpoint model/thin-plate-spline-motion-model/checkpoint/periodic-00000099-epoch.pth.tar')

def generate_TPSM_rotation(imgpath = 'assets/Rotation/rotation_1.jpeg'):
    command = 'CUDA_VISIBLE_DEVICES=0 python model/thin-plate-spline-motion-model/demo.py --config model/thin-plate-spline-motion-model/config/tps_rotation.yaml --checkpoint model/thin-plate-spline-motion-model/checkpoint/rotation-00000099-epoch.pth.tar  --source_image ' + imgpath + ' --driving_video assets/Rotation/rotation_pumpkin.mp4'
    os.system(command)

def generate_TPSM_periodic(imgpath = 'assets/Periodic/periodic_1.jpeg'):
    command = 'CUDA_VISIBLE_DEVICES=0 python model/thin-plate-spline-motion-model/demo.py --config model/thin-plate-spline-motion-model/config/tps_periodic.yaml --checkpoint model/thin-plate-spline-motion-model/checkpoint/rotation-00000099-epoch.pth.tar --source_image ' + imgpath + ' --driving_video assets/Periodic/periodic_metronome.mp4'
    os.system(command)