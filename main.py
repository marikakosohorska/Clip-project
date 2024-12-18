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

from networks.resnet_mod import ResNet

# podiva se napr do souboru weights/clipdet_latent10k/config.yaml
def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        # nacte data z configu
        data = yaml.load(fid, Loader=yaml.FullLoader)
    # weights.pth
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    # clipdet_latent10k, weights.pth, opencliplinearnext_clipL14commonpool, clip, Clip224
    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']

def runnig_tests(input_csv, weights_dir, models_list, device, batch_size = 1):
    table = pandas.read_csv(input_csv)[['filename',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    
    models_dict = dict()
    transform_dict = dict()
    print("Models:")
    for model_name in models_list:
        print(model_name, flush=True)
        print(model_name)
        print(weights_dir)
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

    ### test
    with torch.no_grad():
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

            assert len(batch_id) == 0

    return table

def clip_transform():
    return Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

def train_linear(input_csv, device):
    files = pandas.read_csv(input_csv)[['filename',]]
    typs = pandas.read_csv(input_csv)[['typ',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    model = OpenClipLinear().to(device)
    for param in model.bb[0].parameters():
        param.requires_grad = False
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    num_epochs = 3
    batch_size = 16

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
            try:
                image = clip_transform()(Image.open(filename).convert('RGB'))
                batch_images.append(image)
                batch_labels.append(label_num)
            except FileNotFoundError:
                print(f"File not found: {filename}. Skipping this file.")
                skipped += 1
                continue

            if len(batch_images) == batch_size or index == files.index[-1]:
                batch_images = torch.stack(batch_images).to(device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(1).to(device)

                batch_images.requires_grad = True

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

    torch.save(model.state_dict(), 'weights_linear.pth')
    print("weights saved")

def train_adversarial(input_csv, device, step_w, step_data, gamma, epsilon):
    files = pandas.read_csv(input_csv)[['filename']]
    typs = pandas.read_csv(input_csv)[['typ']]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    
    model = OpenClipLinear().to(device)

    for param in model.bb[0].parameters():
        param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 5
    batch_size = 16

    model.train()

    perturbations_dict = {}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        skipped = 0

        filenames = list(files['filename'])
        # random.shuffle(filenames)

        for batch_start in tqdm.tqdm(range(1000,2000, batch_size)):
            # 1. DRAW A BATCH
            batch_filenames = filenames[batch_start:batch_start + batch_size]
            batch_images = []
            cont = True
            for filename in batch_filenames:
                img_path = os.path.join(rootdataset, filename)
                try:
                    image = Image.open(img_path).convert('RGB')
                    batch_images.append(clip_transform()(image))
                    cont = False
                except FileNotFoundError:
                    print(f"File not found: {filename}. Skipping.")
                    skipped += 1
            if cont: # aspon 1 obrazek v batchi
                continue

            batch_images = torch.stack(batch_images).to(device)
            batch_labels = torch.tensor([0 if typs.loc[files['filename'] == filename, 'typ'].values[0] == 'real' else 1 for filename in batch_filenames],
                                        dtype=torch.float32).unsqueeze(1).to(device)

            print("orinal image")
            print(batch_images[0].shape)
            print(batch_images[0])

            # 2. COMPUTE GRADIENTS APPROXIMATIONS
            batch_perturbations = []
            for filename in batch_filenames:
                if filename not in perturbations_dict:
                    perturbations_dict[filename] = np.zeros_like(batch_images[0].cpu().detach().numpy())
                batch_perturbations.append(torch.tensor(perturbations_dict[filename]).to(device))
            batch_perturbations = torch.stack(batch_perturbations).to(device)
            batch_images_with_perturbations = batch_images + batch_perturbations
            batch_images_with_perturbations.requires_grad = True

            print("perturbations")
            print(batch_perturbations.shape)
            print(batch_perturbations)

            print("new image")
            print(batch_images_with_perturbations.shape)
            print(batch_images_with_perturbations)

            logits = model(batch_images_with_perturbations)
            loss = criterion(logits, batch_labels)
            norm_loss = torch.norm(batch_images - batch_images_with_perturbations, p=2)**2
            total_loss = loss - gamma * norm_loss
            total_loss.backward()

            print("gradient")
            print(batch_images_with_perturbations.grad.shape)
            print(batch_images_with_perturbations.grad)

            # 3. UPDATE WEIGHTS
            with torch.no_grad():
                for param in model.fc.parameters(): # weights (768) + bias (1)
                    param -= step_w * param.grad # zprumerovany pres batch_size .mean()?

            # 4. UPDATE PERTURBATIONS
            with torch.no_grad():
                perturbation = step_data * batch_images_with_perturbations.grad
                for idx, filename in enumerate(batch_filenames):
                    perturbations_dict[filename] += perturbation[idx].cpu().detach().numpy()

                    perturbations_dict[filename] = torch.clamp(
                        torch.tensor(perturbations_dict[filename]),
                        min=-epsilon,
                        max=epsilon).numpy()

            running_loss += loss.item()
            image.close()

        print(f"Epoch {epoch + 1} Loss: {running_loss / len(filenames)-skipped}")

    torch.save(model.state_dict(), 'weights_linear_adversarial.pth')
    np.save('perturbations.npy', perturbations_dict)

    for filename, perturbation in perturbations_dict.items():
        print(f"Filename: {filename}")
        print(f"Perturbation shape: {perturbation.shape}")
        print(f"Perturbation values: {perturbation}\n")

    print("Training complete.")


def test_linear(input_csv, device, input_type, model_type, data_from, data_to):
    files = pandas.read_csv(input_csv)[['filename',]]
    typs = pandas.read_csv(input_csv)[['typ',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    if input_type == 'italy_pgd':
        perturbations_path = os.path.join(os.path.abspath('pgd_italy_perturbations.npy'))
        perturbations_dict = np.load(perturbations_path, allow_pickle=True).item()
    elif input_type == 'italy_train':
        pass
    elif input_type == 'marika_pgd':
        perturbations_path = os.path.join(os.path.abspath('marika_pgd_perturbations.npy'))
        perturbations_dict = np.load(perturbations_path, allow_pickle=True).item()
    elif input_type == 'marika_train':
        perturbations_path = os.path.join(os.path.abspath('perturbations.npy'))
        perturbations_dict = np.load(perturbations_path, allow_pickle=True).item()
        

    model = OpenClipLinear().to(device)

    if model_type == 'italy':
        _, model_path, arch, norm_type, patch_size = get_config('clipdet_latent10k_plus', weights_dir='./weights')
        model = load_weights(create_architecture(arch), model_path)
    elif model_type == 'resnet':
        _, model_path, arch, norm_type, patch_size = get_config('Corvi2023', weights_dir='./weights')
        model = load_weights(create_architecture(arch), model_path)
    elif model_type == 'marika':
        model.load_state_dict(torch.load('weights_linear_adversarial.pth'))

    model = model.to(device).eval()
    
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    skipped = 0

    with torch.no_grad():
        for index in tqdm.tqdm(range(data_from,data_to)):
            filename = os.path.join(rootdataset, files.loc[index, 'filename'])
            label_path = os.path.join(rootdataset, typs.loc[index, 'typ'])
            label_str = os.path.basename(label_path)
            label_num = 0 if label_str == 'real' else 1
            try:
                image = clip_transform()(Image.open(filename).convert('RGB')).to(device)
            except FileNotFoundError:
                print(f"File not found: {filename}. Skipping this file.")
                skipped += 1
                continue

            if input_type == 'italy_pgd' or input_type == 'marika_pgd' or input_type == 'italy_train' or input_type == 'marika_train':
                perturbation = torch.tensor(perturbations_dict[files.loc[index, 'filename']]).to(device)
                adversarial_image = image + perturbation
                logit = model(adversarial_image.unsqueeze(0))
            else:
                logit = model(image.unsqueeze(0))

            loss = criterion(logit, torch.tensor([[label_num]], dtype=torch.float32).to(device))
            running_loss += loss.item()

            predicted_label = 1 if torch.sigmoid(logit) > 0.5 else 0
            print("predicted label")
            print(predicted_label)
            print("correct label")
            print(label_num)
            correct_predictions += (predicted_label == label_num)
            print("correct predictions")
            print(correct_predictions)
            total_predictions += 1

    accuracy = correct_predictions / total_predictions * 100
    loss = running_loss / (len(files) - skipped)
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {loss}")
    return accuracy, loss


def pgd_attack_generate(input_csv, device, model_type, epsilon=0.03, alpha=0.005, num_iter=40):
    files = pandas.read_csv(input_csv)[['filename',]]
    typs = pandas.read_csv(input_csv)[['typ',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))

    if model_type == 'italy':
        _, model_path, arch, norm_type, patch_size = get_config('clipdet_latent10k_plus', weights_dir='./weights')
        model = load_weights(create_architecture(arch), model_path)
    elif model_type == 'resnet':
        _, model_path, arch, norm_type, patch_size = get_config('Corvi2023', weights_dir='./weights')
        model = load_weights(create_architecture(arch), model_path)
    elif model_type == 'marika':
        model = OpenClipLinear().to(device)
        model.load_state_dict(torch.load('weights_linear_adversarial.pth'))

    model = model.to(device).eval()
    perturbations_dict = {}

    for index in tqdm.tqdm(range(len(files))):
        filename = os.path.join(rootdataset, files.loc[index, 'filename'])
        label_path = os.path.join(rootdataset, typs.loc[index, 'typ'])
        label_str = os.path.basename(label_path)
        label_num = 0 if label_str == 'real' else 1

        try:
            image = clip_transform()(Image.open(filename).convert('RGB')).to(device)
        except FileNotFoundError:
            print(f"File not found: {filename}. Skipping this file.")
            continue

        perturbation, flipped = pgd_attack(
            model=model,
            image=image.unsqueeze(0),
            label=torch.tensor([[label_num]], dtype=torch.float32).to(device),
            device=device,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter
        )
        perturbations_dict[files.loc[index, 'filename']] = perturbation.squeeze(0).cpu().numpy()
        
        if flipped:
            print(f"Adversarial perturbation successfully generated and saved for {filename}")
        else:
            print(f"Perturbation generated for {filename}, but no label flip occurred.")

    if model_type == 'italy':
        np.save('italy_pgd_perturbations.npy', perturbations_dict)
    else:
        np.save('marika_pgd_perturbations.npy', perturbations_dict)


def pgd_attack(model, image, label, device, epsilon=0.03, alpha=0.005, num_iter=40):
    perturbed_image = image.clone().detach().to(device)
    perturbed_image.requires_grad = True
    original_prediction = (torch.sigmoid(model(image)) > 0.5).item()

    for i in range(num_iter):
        output = model(perturbed_image)
        loss = torch.nn.BCEWithLogitsLoss()(output, label)

        loss.backward()

        perturbed_image = perturbed_image + alpha * perturbed_image.grad.sign()

        perturbed_image = torch.clamp(perturbed_image, min=image - epsilon, max=image + epsilon).detach()
        # perturbed_image = torch.clamp(perturbed_image, min=0, max=1).detach()
        perturbed_image.requires_grad = True

        new_prediction = (torch.sigmoid(model(perturbed_image)) > 0.5).item()
        if new_prediction != original_prediction:
            print(f"Label flipped at iteration {i+1}.")
            perturbation = (perturbed_image - image).detach()
            return perturbation, True

    print("No label flip occurred.")
    perturbation = (perturbed_image - image).detach()
    return perturbation, False

def run_experiments(in_csv, out_csv, device):
    import csv
    # datasety - ted fixni
    # num_epochs - ted fixni
    # batch_size - ted fixni
    # step size wahy, step size data, gamma, epsilon
    headers = ['epsilon', 'step_size_w', 'step_size_data', 
            'genuine_loss', 'genuine_acc',
            'italy_pgd_loss', 'italy_pgd_acc',
            'marika_pgd_loss', 'marika_pgd_acc',
            'marika_train_loss', 'marika_train_acc']
    with open(out_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        step_size_w = 0.001
        while step_size_w < 1:
            step_size_data = 0.01
            while step_size_data < 10:
                epsilon = 0.1
                while epsilon < 0.6:
                    train_adversarial(in_csv, device, step_size_w, step_size_data, 0.1, epsilon)
                    pgd_attack_generate(in_csv, device, 'marika')
                    acc1, loss1 = test_linear(in_csv, device, 'genuine', 'marika', 0, 2000)
                    acc2, loss2 = test_linear(in_csv, device, 'italy_pgd', 'marika', 0, 2000)
                    acc3, loss3 = test_linear(in_csv, device, 'marika_pgd', 'marika', 0, 2000)
                    acc4, loss4 = test_linear(in_csv, device, 'marika_train', 'marika', 1000, 2000)
                    writer.writerow([
                        epsilon, step_size_w, step_size_data,
                        loss1, acc1,
                        loss2, acc2,
                        loss3, acc3,
                        loss4, acc4
                    ])
                    epsilon += 0.2
                step_size_data *= 10
            step_size_w *= 10
    print(f"Results have been saved to {out_csv}")




    


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
    train_adversarial(args['in_csv'], args['device'], 0.001, 0.01, 0.1, 0.1) # step_size_vahy, step_size_data, gamma, epsilon
    # test_italy(args['in_csv'], args['device'])

    pgd_attack_generate(args['in_csv'], args['device'], 'marika')
    test_linear(args['in_csv'], args['device'], 'marika_pgd', 'marika', 0, 2000)
    # run_experiments(args['in_csv'], 'experiments.csv', args['device'])

    # checkpoint = torch.load('perturbations.npy')

    # Print the contents of the checkpoint to see what's inside
    # print("Contents of the .pth file:")
    # print(checkpoint)
    
    # table = runnig_tests(args['in_csv'], args['weights_dir'], args['models'], args['device'])
    # if args['fusion'] is not None:
    #     table['fusion'] = apply_fusion(table[args['models']].values, args['fusion'], axis=-1)
    
    # output_csv = args['out_csv']
    # os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    # table.to_csv(output_csv, index=False)  # save the results as csv file

