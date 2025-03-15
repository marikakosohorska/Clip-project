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
from PIL import Image, UnidentifiedImageError

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
import torch.optim as optim

from networks.resnet_mod import ResNet
from tensorboardX import SummaryWriter
import math
from torch.utils.data import DataLoader, Dataset, Sampler

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

class ImageDataset(Dataset):
    def __init__(self, input_csv, transform=None):
        df = pandas.read_csv(input_csv)[['filename', 'typ']]
        self.rootdataset = os.path.dirname(os.path.abspath(input_csv))  
        self.real_files = df[df['typ'] == 'real']
        self.synthetic_files = df[df['typ'] != 'real']
        self.transform = transform

    def __len__(self):
        return len(self.real_files) + len(self.synthetic_files)

    def __getitem__(self, idx):
        if idx < 1500:
            filename = self.synthetic_files.loc[idx, 'filename']
            img_path = os.path.join(self.rootdataset, filename)
            label = 1
        else:
            filename = self.real_files.loc[idx, 'filename']
            img_path = os.path.join(self.rootdataset, filename)
            label = 0
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return (image, torch.tensor(label, dtype=torch.float32),filename)
        except FileNotFoundError:
            print(f"File not found: {img_path}. Skipping.")
            return None

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.synthetic_indices = list(range(0, 400)) + list(range(500, 900)) + list(range(1000, 1400))
        self.real_indices = list(range(1500,1900))
        self.batch_size = batch_size
        self.half_batch = batch_size // 2

    def __iter__(self):
        random.shuffle(self.real_indices)
        random.shuffle(self.synthetic_indices)
        batches = []
        real_counter = 0
        for i in range(0, len(self.synthetic_indices), self.half_batch):
            batch = self.real_indices[real_counter:real_counter+self.half_batch] + self.synthetic_indices[i:i+self.half_batch]
            real_counter += self.half_batch
            if real_counter >= 400:
                real_counter = 0
                random.shuffle(self.real_indices)
            random.shuffle(batch)
            batches.append(batch)
        random.shuffle(batches)
        return iter([idx for batch in batches for idx in batch])

    def __len__(self):
        return 2400 # pocet vsech trenovacich obrazku


def train_nonadversarial(input_csv, device):
    dataset = ImageDataset(input_csv, transform=clip_transform())
    sampler = BalancedBatchSampler(dataset, batch_size=16)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler, drop_last=True, num_workers=0)

    model = OpenClipLinear().to(device)
    for param in model.bb[0].parameters():
        param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = 10

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        for batch in tqdm.tqdm(dataloader):
            if batch is None:
                print("No batch")
                continue

            batch_images,batch_labels,batch_filenames = batch
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.unsqueeze(1).to(device)

            # Forward
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss/(len(dataloader)) }")

    torch.save(model.state_dict(), 'weights_linear.pth')
    print("weights saved")




def train_adversarial(input_csv, device, step_w, step_data, gamma, epsilon):
    dataset = ImageDataset(input_csv, transform=clip_transform())
    sampler = BalancedBatchSampler(dataset, batch_size=16)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler, drop_last=True, num_workers=0)

    model = OpenClipLinear().to(device)
    for param in model.bb[0].parameters():
        param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for i, param in enumerate(model.parameters()):
        print(f"Param {i}: {param.shape}")

    model.train()

    num_epochs = 15
    perturbations_dict = {}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0

        for batch in tqdm.tqdm(dataloader):
            if batch is None:
                print("No batch")
                continue

            batch_images,batch_labels,batch_filenames = batch
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.unsqueeze(1).to(device)
            print(batch_filenames)
            print(batch_labels)

            # PGD ATTACK FOR BATCH IMAGES
            batch_perturbations = []
            for index, image in enumerate(batch_images):
                image = image.unsqueeze(0)
                initial_perturbation = torch.empty_like(image).uniform_(-epsilon, epsilon).to(device)
                perturbation, success, iter = pgd_attack(
                    model, image, device, batch_labels[index], epsilon, step_data, gamma, initial_perturbation=initial_perturbation
                )
                batch_perturbations.append(perturbation)
                if epoch == num_epochs - 1:
                    perturbations_dict[batch_filenames[index]] = perturbation.squeeze().cpu().numpy()

            batch_perturbations = torch.stack(batch_perturbations).to(device)
            batch_images_with_perturbations = batch_images + batch_perturbations.squeeze(1)

            for param in model.parameters():
                param.requires_grad = True

            # UPDATE WEIGHTS
            logits = model(batch_images_with_perturbations)

            loss = criterion(logits, batch_labels)
            norm_loss = 0.0
            for i in range(batch_images.size(0)):
                norm_loss += torch.norm(batch_images[i] - batch_images_with_perturbations[i], p=2) ** 2
            total_loss = loss - gamma * norm_loss / batch_images.size(0)

            model.zero_grad()
            optimizer.zero_grad()

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss/(len(dataloader)) }")

    np.save('perturbations.npy', perturbations_dict)
    torch.save(model.state_dict(), 'weights_linear_adversarial.pth')
    print("weights saved")
    



def evaluate(input_csv, device, input_type, model_type, ranges):
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    df = pandas.read_csv(input_csv)
    files = df['filename'].tolist()
    typs = df['typ'].tolist()
    selected_indices = [i for sublist in ranges for i in sublist]
    selected_files = [files[i] for i in selected_indices]
    selected_typs = [typs[i] for i in selected_indices]
    train_indices = list(range(0, 400))+list(range(500, 900))+list(range(1000, 1400))


    if input_type == 'italy_pgd':
        perturbations_path = os.path.join(os.path.abspath('italy_pgd_perturbations.npy'))
        perturbations_dict = np.load(perturbations_path, allow_pickle=True).item()
    elif input_type == 'italy_train':
        pass
    elif input_type == 'marika_pgd':
        perturbations_path = os.path.join(os.path.abspath('marika_pgd_perturbations.npy'))
        perturbations_dict = np.load(perturbations_path, allow_pickle=True).item()
    elif input_type == 'marika_train':
        perturbations_path = os.path.join(os.path.abspath('perturbations.npy'))
        perturbations_dict = np.load(perturbations_path, allow_pickle=True).item()

    if model_type == 'italy':
        # _,model_path,arch,_,_ = get_config('clipdet_latent10k_plus', weights_dir='./weights')
        # model = load_weights(create_architecture(arch), model_path).to(device)
        model = OpenClipLinear().to(device)
        model.load_state_dict(torch.load('weights_linear.pth'))
    elif model_type == 'resnet':
        _,model_path,arch,_,_ = get_config('Corvi2023', weights_dir='./weights')
        model = load_weights(create_architecture(arch), model_path)
    elif model_type == 'marika':
        model = OpenClipLinear().to(device)
        model.load_state_dict(torch.load('weights_linear_adversarial.pth'))
    model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    results = []
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for index,filename,typ in tqdm.tqdm(zip(selected_indices,selected_files,selected_typs),total=len(selected_files)):
            print(f"Index: {index}, Filename: {filename}, Type: {typ}")
            full_filename = os.path.join(rootdataset, filename)
            label_num = 0 if typ == 'real' else 1
            try:
                image = clip_transform()(Image.open(full_filename).convert('RGB')).to(device)
            except UnidentifiedImageError:
                print(f"Cannot identify image file: {full_filename}.")
                continue
            except OSError:
                print(f"Error loading image: {full_filename}.")
                continue
            except FileNotFoundError:
                print(f"File not found: {full_filename}.")
                continue

            if input_type == 'italy_pgd' or input_type == 'marika_pgd' or input_type == 'italy_train' or input_type == 'marika_train':
                perturbation = torch.tensor(perturbations_dict[filename]).to(device)
                adversarial_image = image + perturbation
                logit = model(adversarial_image.unsqueeze(0))
            else:
                logit = model(image.unsqueeze(0))

            loss = criterion(logit, torch.tensor([[label_num]], dtype=torch.float32).to(device))
            running_loss += loss.item()

            predicted_label = 1 if logit >= 0 else 0
            correct_predictions += (predicted_label == label_num)
            total_predictions += 1

            print("predicted label")
            print(predicted_label)
            print("correct label")
            print(label_num)
            print("correct predictions")
            print(correct_predictions)

            true_labels.append(label_num)
            predicted_labels.append(predicted_label)

            results.append({
                "filename": filename,
                "label": "real" if label_num == 0 else "generated",
                "p(real)": 1-torch.sigmoid(logit).item(),
                "p(generated)": torch.sigmoid(logit).item(),
                "train_test": "train" if index in train_indices else "test"
            })

    results_df = pandas.DataFrame(results)
    results_df.to_csv("results.csv", index=False)
    # accuracy = correct_predictions / total_predictions
    loss = running_loss / total_predictions

    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    accuracy = accuracy_score(true_labels, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Balanced accuracy: {balanced_accuracy:.2f}%")
    print(f"Test Loss: {loss}")
    return accuracy, balanced_accuracy, loss


def pgd_attack_generate(input_csv, device, model_type, epsilon, step_data, gamma, num_iter=1):
    files = pandas.read_csv(input_csv)[['filename',]]
    typs = pandas.read_csv(input_csv)[['typ',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))

    if model_type == 'italy':
        # _,model_path,arch,_,_ = get_config('clipdet_latent10k_plus', weights_dir='./weights')
        # model = load_weights(create_architecture(arch), model_path)
        model = OpenClipLinear()
        model.load_state_dict(torch.load('weights_linear.pth'))
    elif model_type == 'resnet':
        _,model_path,arch,_,_ = get_config('Corvi2023', weights_dir='./weights')
        model = load_weights(create_architecture(arch), model_path)
    elif model_type == 'marika':
        model = OpenClipLinear()
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
        perturbation, flipped, _ = pgd_attack(
            model,
            image.unsqueeze(0),
            device,
            torch.tensor([[label_num]], dtype=torch.float32).to(device),
            epsilon,
            step_data,
            gamma,
            num_iter=num_iter
        )
        perturbations_dict[files.loc[index, 'filename']] = perturbation.squeeze(0).cpu().numpy()
        
        if flipped:
            print(f"Adversarial perturbation successfully generated and saved for {filename}")
        else:
            print(f"Perturbation generated for {filename}, but no label flip occurred.")

    if model_type == 'italy':
        np.save('italy_pgd_perturbations.npy', perturbations_dict)
    elif model_type == 'marika':
        np.save('marika_pgd_perturbations.npy', perturbations_dict)


def pgd_attack(model, image, device, true_label, epsilon, step_data, gamma, initial_perturbation=None, num_iter=1):
    print(true_label)

    if initial_perturbation is None:
        perturbed_image = image.clone().detach().to(device)
    else:
        perturbed_image = image.clone().detach().to(device) + initial_perturbation

    target = torch.tensor([[true_label]], dtype=torch.float32, device=device)

    for param in model.parameters():
        param.requires_grad = False

    for i in range(num_iter):
        perturbed_image.requires_grad = True
        
        logit = model(perturbed_image)
        current_prediction = (torch.sigmoid(logit) > 0.5).item()

        if current_prediction != true_label:
            print(f"Label flipped at iteration {i + 1}.")
            perturbation = (perturbed_image - image).detach()
            return perturbation, True, i

        loss = torch.nn.BCEWithLogitsLoss()(logit, target)
        norm_loss = torch.norm(image - perturbed_image, p=2) ** 2
        total_loss = loss - gamma * norm_loss
        model.zero_grad()
        # if perturbed_image.grad is not None:
        #     perturbed_image.grad.zero()
        total_loss.backward()

        with torch.no_grad():
            perturbed_image = perturbed_image + step_data * perturbed_image.grad.sign()
            perturbed_image = torch.clamp(perturbed_image, min=image - epsilon, max=image + epsilon)
        
        perturbed_image = perturbed_image.detach()

    print("No label flip occurred.")
    perturbation = (perturbed_image - image).detach()
    return perturbation, False, 1



def save_img(input_csv, device, index):
    files = pandas.read_csv(input_csv)[['filename',]]
    typs = pandas.read_csv(input_csv)[['typ',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    perturbations_path = os.path.join(os.path.abspath('perturbations.npy'))
    perturbations_dict = np.load(perturbations_path, allow_pickle=True).item()
    filename = os.path.join(rootdataset, files.loc[index, 'filename'])

    img = clip_transform()(Image.open(filename).convert('RGB')).to(device)
    print(img)
    img_rounded = torch.round(img * 10) / 10
    img_clipped = torch.clamp(img_rounded, -1.3, 1.3)
    img_scaled = (img_clipped + 1.3) / 2.6
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img_scaled)
    img_pil.save("image.png")

    perturbation = torch.tensor(perturbations_dict[files.loc[index, 'filename']]).to(device)
    print(perturbation)
    # perturbation_rounded = torch.round(perturbation * 10) / 10
    # perturbation_clipped = torch.clamp(perturbation_rounded, -0.1, 0.3)
    perturbation_scaled = (perturbation + 0.1) / 0.2
    to_pil = transforms.ToPILImage()
    perturbation_pil = to_pil(perturbation_scaled)
    perturbation_pil.save("perturbation.png")


    adversarial_image = img + perturbation
    print(adversarial_image)
    img_rounded = torch.round(adversarial_image * 10) / 10
    img_clipped = torch.clamp(img_rounded, -1.3, 1.3)
    img_scaled = (img_clipped + 1.5) / 3.0
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img_scaled)
    img_pil.save("image_adv.png")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # data/synthwildx/list.csv
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

    step_weights = 0.01
    step_data = 0.1
    epsilon = 0.1
    gamma = 0
    train = [range(0, 400),range(500, 900),range(1000, 1400),range(1500,1900)]
    test = [range(400,500),range(900,1000),range(1400,1500),range(1900,2000)]


    # train_nonadversarial(args['in_csv'], args['device'])
    train_adversarial(args['in_csv'], args['device'], step_weights, step_data, gamma, epsilon) # step_size_vahy, step_size_data, gamma, epsilon


    # pgd_attack_generate(args['in_csv'], args['device'], 'marika', epsilon, step_data, gamma, 1)
    # evaluate(args['in_csv'], args['device'], 'marika_pgd', 'marika', train)
    # run_experiments(args['in_csv'], 'experiments.csv', args['device'])

    # checkpoint = np.load('perturbations.npy',allow_pickle=True)
    # print(checkpoint)

    # save_img(args['in_csv'],args['device'],1788)







    
    # table = runnig_tests(args['in_csv'], args['weights_dir'], args['models'], args['device'])
    # if args['fusion'] is not None:
    #     table['fusion'] = apply_fusion(table[args['models']].values, args['fusion'], axis=-1)
    
    # output_csv = args['out_csv']
    # os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    # table.to_csv(output_csv, index=False)  # save the results as csv file


    # test_italy(args['in_csv'], args['device'])
