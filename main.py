'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
'''

import torch
import os
import pandas
import numpy as np
import tqdm
import glob
import sys
import yaml
from PIL import Image

from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize
from utils.fusion import apply_fusion
from networks import create_architecture, load_weights

import torchvision.transforms as transforms
from networks.openclipnet import OpenClipLinear
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from torchvision.transforms.functional import InterpolationMode
import random

# from memory_profiler import profile

# podiva se napr do souboru weights/clipdet_latent10k/config.yaml
def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        # nacte data z configu
        data = yaml.load(fid, Loader=yaml.FullLoader)
    # weights.pth
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    # clipdet_latent10k, weights.pth, opencliplinearnext_clipL14commonpool, clip, Clip224
    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']

# @profile
def runnig_tests(input_csv, weights_dir, models_list, device, batch_size = 1):
    table = pandas.read_csv(input_csv)[['filename',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    
    models_dict = dict()
    transform_dict = dict()
    print("Models:")
    for model_name in models_list:
        print(model_name, flush=True)
        _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)
        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()

        transform = list()
        if patch_size is None:
            print('input none', flush=True)
            transform_key = 'none_%s' % norm_type
        elif patch_size=='Clip224':
            print('input resize:', 'Clip224', flush=True)
            transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
            transform.append(CenterCrop((224, 224)))
            transform_key = 'Clip224_%s' % norm_type
        elif isinstance(patch_size, tuple) or isinstance(patch_size, list):
            print('input resize:', patch_size, flush=True)
            transform.append(Resize(*patch_size))
            transform.append(CenterCrop(patch_size[0]))
            transform_key = 'res%d_%s' % (patch_size[0], norm_type)
        elif patch_size > 0:
            print('input crop:', patch_size, flush=True)
            transform.append(CenterCrop(patch_size))
            transform_key = 'crop%d_%s' % (patch_size, norm_type)
        
        transform.append(make_normalize(norm_type))
        transform = Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)
        print(flush=True)

    ### test
    with torch.no_grad():
        print("!!TEST TEST TEST!!")
        do_models = list(models_dict.keys())
        do_transforms = set([models_dict[_][0] for _ in do_models])
        
        print("Running the Tests")
        batch_img = {k: list() for k in transform_dict}
        batch_id = list()
        last_index = table.index[-1]
        for index in tqdm.tqdm(table.index, total=len(table)):
            filename = os.path.join(rootdataset, table.loc[index, 'filename'])
            cont = False
            for k in transform_dict:
                try:
                    batch_img[k].append(transform_dict[k](Image.open(filename).convert('RGB')))
                except FileNotFoundError:
                    print(f"File not found: {filename}. Skipping this file.")
                    cont = True
            if cont:
                continue

            batch_id.append(index)

            # process batch
            if (len(batch_id) >= batch_size) or (index==last_index):
                for k in do_transforms:
                    batch_img[k] = torch.stack(batch_img[k], 0)

                # clipdet, corvi
                for model_name in do_models:
                    out_tens = models_dict[model_name][1](batch_img[models_dict[model_name][0]].clone().to(device)).cpu().numpy()
                    
                    image_tensor = batch_img[models_dict[model_name][0]].clone().to(device)[0]
                    print("IMAGE TENSOR")
                    print(image_tensor)
                    adversarial_image_np = image_tensor.cpu().detach().numpy().transpose(1, 2, 0)
                    adversarial_image_pil = Image.fromarray((adversarial_image_np * 255).astype(np.uint8))  # Convert to [0, 255] range
                    adversarial_image_pil.save('adversarial_image.jpg')

                    # adversarial_image_np = batch_img[models_dict[model_name][0]].cpu().detach().numpy().transpose(1, 2, 0)  # Convert to HWC format
                    # adversarial_image_pil = Image.fromarray((adversarial_image_np * 255).astype(np.uint8))  # Convert to [0, 255] range
                    # adversarial_image_pil.save('adversarial_image.jpg')
                    if out_tens.shape[1] == 1:
                        out_tens = out_tens[:, 0]
                    elif out_tens.shape[1] == 2:
                        out_tens = out_tens[:, 1] - out_tens[:, 0]
                    else:
                        assert False
                    
                    if len(out_tens.shape) > 1:
                        logit1 = np.mean(out_tens, (1, 2))
                    else:
                        logit1 = out_tens

                    for ii, logit in zip(batch_id, logit1):
                        table.loc[ii, model_name] = logit

                batch_img = {k: list() for k in transform_dict}
                batch_id = list()

            assert len(batch_id)==0

    return table

def clip_transform():
    return Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

# na jakych datech trenovat, navic real/synthetic musi byt rovnomerne rozlozene?
# bfgs
# git

def train_linear(input_csv, device):
    files = pandas.read_csv(input_csv)[['filename',]]
    typs = pandas.read_csv(input_csv)[['typ',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    model = OpenClipLinear().to(device)
    for param in model.bb[0].parameters():
        param.requires_grad = False
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    num_epochs = 1
    batch_size = len(files)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        skipped = 0
        batch_images = []
        batch_labels = []
        for index in tqdm.tqdm(files.index, total=len(files)):
            filename = os.path.join(rootdataset, files.loc[index, 'filename'])
            label_path = os.path.join(rootdataset, typs.loc[index, 'typ'])
            label_str = os.path.basename(label_path)
            label_num = 0 if label_str == 'real' else 1
            cont = False
            try:
                image = clip_transform()(Image.open(filename).convert('RGB'))
                batch_images.append(image)
                batch_labels.append(label_num)
            except FileNotFoundError:
                print(f"File not found: {filename}. Skipping this file.")
                cont = True
                skipped += 1
            if cont:
                continue

            if len(batch_images) == batch_size or index == files.index[-1]:
                batch_images = torch.stack(batch_images).to(device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(1).to(device)
                # Forward
                logits = model(batch_images)
                loss = criterion(logits, batch_labels)
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batch_images = []
                batch_labels = []

        print(f"Epoch {epoch + 1}, Loss: {running_loss/(len(files)-skipped) }")

    torch.save(model.state_dict(), 'weights_linear_final.pth')
    print("weights saved")


def train_adversarial(input_csv, device, step_w, step_data):
    files = pandas.read_csv(input_csv)[['filename']]
    typs = pandas.read_csv(input_csv)[['typ']]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))

    # Preload dataset
    print("Loading dataset...")
    skipped = 0
    dataset = {}
    # for index in tqdm.tqdm(files.index, total=len(files)):
    for index in tqdm.tqdm(range(1100,2000),total=900):
        filename = os.path.join(rootdataset, files.loc[index, 'filename'])
        label_path = os.path.join(rootdataset, typs.loc[index, 'typ'])
        label_str = os.path.basename(label_path)
        label_num = 0 if label_str == 'real' else 1
        try:
            # Preload images into memory as tensors
            dataset[filename] = (clip_transform()(Image.open(filename).convert('RGB')),label_num)
        except FileNotFoundError:
            print(f"File not found: {filename}. Skipping.")
            skipped += 1
    
    model = OpenClipLinear().to(device)
    for param in model.bb[0].parameters():
        param.requires_grad = False  # Freeze CLIP backbone weights

    # torch.save(model.bb[0].state_dict(), 'backbone_weights_initial.pth')
    # torch.save(model.state_dict(), 'weights_linear_initial.pth')
    # print("Initial weights saved")

    criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    num_epochs = 1
    batch_size = 1

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0

        # Shuffle the data
        filenames = list(dataset.keys())
        random.shuffle(filenames)

        for batch_start in tqdm.tqdm(range(0, len(filenames), batch_size)):
            # Prepare batch
            batch_filenames = filenames[batch_start:batch_start + batch_size]
            batch_images = torch.stack([dataset[fn][0] for fn in batch_filenames]).to(device).squeeze(1)
            print("IMAGE")
            print(batch_images[0].shape)
            print(batch_images[0])
            
            print(batch_filenames[0])
            image_np = batch_images[0].cpu().detach().numpy().transpose(1, 2, 0)
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # Convert to [0, 255] range
            image_pil.save('image.jpg')
            batch_labels = torch.tensor([dataset[fn][1] for fn in batch_filenames], dtype=torch.float32).unsqueeze(1).to(device)

            # Enable gradient computation for adversarial images
            batch_images.requires_grad = True

            # 1. UPDATE WEIGHTS
            # optimizer.zero_grad()
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()  # Compute gradients for weights and also for the images?
            # optimizer.step()  # Update the model's weights

            with torch.no_grad():
                for param in model.fc.parameters(): # weights (768) + bias (1)
                    param -= step_w * param.grad # zprumerovany pres batch_size .mean()?

            avg_grad = batch_images.grad.mean(dim=0, keepdim=True)
            print("AVERAGED IMAGE GRAD")
            print(avg_grad.shape)
            print(avg_grad)

            # 2. UPDATE IMAGES
            with torch.no_grad():
                perturbation = step_data * avg_grad
                perturbation = perturbation.to(device)
                for filename, (img_tensor, label_num) in dataset.items():
                    img_tensor = img_tensor.to(device)
                    updated_img_tensor = img_tensor + perturbation
                    dataset[filename] = (updated_img_tensor, label_num)


            print("IMAGE UPDATED")
            print(dataset[filenames[0]][0])

            image_np = dataset[filenames[0]][0].squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # Convert to [0, 255] range
            image_pil.save('image2.jpg')

            running_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {running_loss / len(filenames)-skipped}")

    # torch.save(model.bb[0].state_dict(), 'backbone_weights_final.pth')
    torch.save(model.state_dict(), 'weights_linear_final.pth')
    print("weights saved")

    print("Training complete.")


def test_linear(input_csv, device):
    files = pandas.read_csv(input_csv)[['filename',]]
    typs = pandas.read_csv(input_csv)[['typ',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    model = OpenClipLinear().to(device)
    
    model.load_state_dict(torch.load('weights_linear_final.pth'))
    
    model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    skipped = 0

    with torch.no_grad():  # No need to track gradients during testing
        for index in tqdm.tqdm(files.index, total=len(files)):
            filename = os.path.join(rootdataset, files.loc[index, 'filename'])
            label_path = os.path.join(rootdataset, typs.loc[index, 'typ'])
            label_str = os.path.basename(label_path)
            label_num = 0 if label_str == 'real' else 1
            cont = False
            try:
                image = clip_transform()(Image.open(filename).convert('RGB')).to(device)
            except FileNotFoundError:
                print(f"File not found: {filename}. Skipping this file.")
                cont = True
                skipped += 1
            if cont:
                continue

            # Forward pass
            logit = model(image.unsqueeze(0))  # Unsqueeze to make it a batch of size 1
            loss = criterion(logit, torch.tensor([[label_num]], dtype=torch.float32).to(device))
            running_loss += loss.item()

            # Calculate predictions
            predicted_label = 1 if torch.sigmoid(logit) > 0.5 else 0  # Apply sigmoid and threshold to get binary prediction
            print("predicted label")
            print(predicted_label)
            print("label num")
            print(label_num)
            print()
            correct_predictions += (predicted_label == label_num)
            print("correct predictions")
            print(correct_predictions)
            total_predictions += 1

    # Print loss and accuracy after testing
    print(f"Test Loss: {running_loss / (len(files) - skipped)}")
    print(f"Accuracy: {correct_predictions / total_predictions * 100:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # data/synthwildx/list.csv ?
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="./results.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="./weights")
    parser.add_argument("--models"     , '-m', type=str, help="List of models to test", default='clipdet_latent10k_plus,Corvi2023')
    parser.add_argument("--fusion"     , '-f', type=str, help="Fusion function", default='soft_or_prob')
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cpu')
    args = vars(parser.parse_args())
    
    if args['models'] is None:
        args['models'] = os.listdir(args['weights_dir'])
    else:
        args['models'] = args['models'].split(',')

    # train_linear(args['in_csv'], args['device'])
    train_adversarial(args['in_csv'], args['device'], 1, 1000) # the last two parameters: step_size_vahy, step_size_data
    test_linear(args['in_csv'], args['device'])

    # checkpoint = torch.load('weights_linear_final.pth')

    # Print the contents of the checkpoint to see what's inside
    # print("Contents of the .pth file:")
    # print(checkpoint)
    
    # table = runnig_tests(args['in_csv'], args['weights_dir'], args['models'], args['device'])
    # if args['fusion'] is not None:
    #     table['fusion'] = apply_fusion(table[args['models']].values, args['fusion'], axis=-1)
    
    # output_csv = args['out_csv']
    # os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    # table.to_csv(output_csv, index=False)  # save the results as csv file

