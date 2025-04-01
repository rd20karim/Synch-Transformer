import logging
import os
from datasets.visualization import decode_predictions_and_compute_bleu_score
from architectures.Transformer import create_model
from evaluate_transformer import evaluate
from datasets.loader import build_data
import torch
import torch.nn as nn
import random
# import numpy as np
# import math
# import time
import wandb
# import pprint
import argparse
import yaml
# from PIL import Image
# import pandas as pd
# import seaborn as sns
# from torchtext.data.metrics import bleu_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''A wrapper class for scheduled optimizer '''
import numpy as np
import matplotlib.pyplot as plt
# from io import BytesIO
# from matplotlib import patches
# import threading
# from concurrent.futures import ThreadPoolExecutor
# Set matplotlib to use the 'Agg' backend
plt.switch_backend('Agg')
#matplotlib.use('Agg')

def learning_rate_schedule(d_model, warmup_steps, step_num,Lr_max=None,a=None,total_steps=None):
    if Lr_max:
        a = (Lr_max)**(-2) / (warmup_steps*d_model)
    learning_rate = np.minimum((a*step_num)**(-0.5), (warmup_steps*a)**(-1.5) *(a*step_num) ) * (d_model ** -0.5)
    lr_for_plot  = lambda step_num : np.minimum((a*step_num)**(-0.5), (warmup_steps*a)**(-1.5) *(a*step_num) ) * (d_model ** -0.5)
    if step_num==1:
        logging.info("Displaying/Saving scheduling curve")
        # Calculate learning rates over the specified number of steps
        learning_rates = [lr_for_plot(step) for step in
                          range(1, total_steps + 1)]

        # Plot the learning rate schedule
        fig,ax = plt.subplots()

        ax.plot(range(1, total_steps + 1), learning_rates)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Transformer Learning Rate Schedule')
        fig.tight_layout()
        fig.savefig("scheduling_lr_curve.png")
    return learning_rate


def step_LR(lr,step_num,step_size=15,gamma=0.1,cyclic=False,lr_init=0.0001):
    if step_num % step_size == 0:
        return lr*gamma
    elif not cyclic or lr >=1e-6 : return lr #>=1e-6
    else : return lr_init

def exp_LR(lr,step_num,gamma=0.5,cyclic=False,lr_init=0.0001,shift=0):
    if  not cyclic or lr >=1e-6 :
      return lr_init * np.exp(-gamma*(step_num-shift)),step_num
    else : return lr_init,step_num


# Example and plotting over steps
d_model = 256  # Dimension of the model
warmup_steps = 50 # Number of warmup steps
total_steps = 500  # Total number of training steps

def align_schedule(epoch,L_max=1000,L_start=15,L_end=55,mode='exp'):
    if mode =='exp':
        return min(max(0,L_max*(epoch-L_start)/(L_end-L_start) * (epoch/L_end)**2),L_max)
    elif mode == 'linear':
        return min(max(0, L_max * (epoch - L_start) / (L_end - L_start)), L_max)
    else:
        print("mode not found from available choices 'exp' / 'linear'  ")
        raise ValueError


import threading

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)






def train(model, iterator, optimizer, criterion, clip,num_grams=4,config=None,epoch=None):
    model.train()
    epoch_loss = 0
    BLEU_scores = []
    for i, batch in enumerate(iterator):
        src = batch[0].to(device).type(torch.float)
        trg = batch[1].to(device)
        src_len = torch.tensor(batch[2]).unsqueeze(1)
        trg_len = batch[3]
        optimizer.zero_grad()
        output, _ = model(src, trg[:,:-1],src_len)
        #output, _ = model(src, trg,src_len)

        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
        output_dim = output.shape[-1]
        output_logits = output
        trg_out = trg
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
        L0 = config.L0 if not config.align_schedule else align_schedule(epoch,L_max=1,L_end=1000,L_start=150,mode='exp') #L_max=50/1000,L_end=200/150,L_start=20)
        L1 = config.L1 if not config.align_schedule else align_schedule(epoch,L_max=1000,L_end=500,L_start=150,mode='exp')
        L2 = config.L2
        order_loss = model.decoder.order_loss_
        align0_loss = model.decoder.align0_loss_
        alignf_loss = model.decoder.alignf_loss_

        _loss = criterion(output, trg)
        loss = _loss + L0 * align0_loss + L1 * order_loss + L2 *alignf_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        #print(str(i)+"/"+str(len(iterator))+" loss batch ---> "+str(loss.item())+"\r",end='')
        epoch_loss += _loss.item()
        vocab_obj = iterator.dataset.lang
        bleu_score, _, _ = decode_predictions_and_compute_bleu_score(output_logits, trg_out, vocab_obj,
                                                                     num_grams=num_grams,batch_first=True)
        BLEU_scores += [bleu_score]
        print(f"Loss/train_batch %d --> %.3f BLEU score_batch %.3f\n" % (i, loss, bleu_score),end='')
        print(f"--------> {L0}*align0_loss %.3f"% (L0*align0_loss.item()))
        print(f"--------> {L1}*orderloss %.3f"% (L1*order_loss.item()))
        print(f"--------> {L2}*alignf_loss %.3f"% (L2*alignf_loss.item()))
        print(f"-------->  loss normal %.3f" % (_loss.item()))
    bleu_epoch = (sum(BLEU_scores) / len(BLEU_scores))
    print("\033[1;32m BLEU TRAIN ---- > %.3f" % bleu_epoch)
    return epoch_loss / len(iterator), bleu_epoch




def train_tune():

    with wandb.init(dir = "./wlogs") as run:
        print(f"D: {config.D} / r : {config.r}")
        PF_DIM = 2 * config.hid_dim # TODO TUNE THE FACTOR '2'
        max_length = config.max_length # In all previous runs it was 500

        project_path = r"C:\Users\karim\PycharmProjects\SemMotion"
        aug_path = r"C:\Users\karim\PycharmProjects\HumanML3D"
        if "kit" in args.dataset_name:
            # ------------ [Augmented-KIT] ------------
            from datasets.kit_m2t_dataset import dataset_class
            path_txt = project_path + "\datasets\sentences_corrections.csv"
            path_motion = aug_path + "\kit_with_splits_2023.npz"
            n_joint = 21
            # -----------  H3D IMPORTS   ---------------------
        elif args.dataset_name == "h3D":
            from datasets.h3d_m2t_dataset_ import dataset_class
            path_txt = aug_path + "\sentences_corrections_h3d.csv"
            path_motion = aug_path + "\\all_humanML3D.npz"
            n_joint = 22

        "BUILD DATA"
        batch_size = config.batch_size
        train_data_loader, val_data_loader, test_data_loader = build_data(dataset_class=dataset_class, path=path_motion, min_freq=3,
                                                                          train_batch_size=batch_size,
                                                                          test_batch_size=batch_size,
                                                                          return_lengths=True, path_txt=path_txt,
                                                                          return_trg_len=True, joint_angles=False)


        INPUT_DIM = n_joint * 3
        OUTPUT_DIM = train_data_loader.dataset.lang.vocab_size_unk


        ENC_LAYERS = 1
        DEC_LAYERS = 1 

        model = create_model(OUTPUT_DIM,DEC_LAYERS,config.h_dec,PF_DIM,config.dropout,
                              INPUT_DIM,config.hid_dim,ENC_LAYERS,config.h_enc,PF_DIM,config.dropout, # ENC_PF_DIM = 2*config.hid_dim
                              device, max_length=max_length,D=config.D,r=config.r,margin=config.margin,
                              n_joint=n_joint,spatial=config.spatial,concat=config.concat)
        
        model.apply(initialize_weights)

        criterion = nn.CrossEntropyLoss(ignore_index = 0) #TRG_PAD_ID
        N_EPOCHS = config.epochs
        CLIP = 1
        best_valid_bleu = 0 #float('inf')
        optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
        start = 0

        if config.resume == 1:
            logging.info(" ------- Resume training --------- ")
            model_dict = torch.load(config.path)
            model.load_state_dict(model_dict["model"])
            optimizer.load_state_dict(model_dict["optimizer"])
            start = model_dict["epoch"] + 1
            print("RESUME AT EPOCH ", start)

        # CREATE A DIRECTORY PER PROJECT
        os.makedirs(abs_path + PROJECT_NAME, exist_ok=True)

        # DIR TO SAVE MODEL WITH UNIQUE ID GENERATED PER RUN FOR THE SPECIFIED PROJECT
        unique_path = abs_path + PROJECT_NAME + f'/model_{wandb.run.id}'

        # # INITIALIZE SAVING THE PYTORCH MODEL TO DIRECTORY
        # torch.save({'model': model.state_dict()},
        #            unique_path)
        lr = config.lr # intial learning rate
        for epoch in range(start,N_EPOCHS):
            if config.scheduling==1:
                st_sch = start  # start for resume #[trg 15]
                if epoch >=st_sch:
                   lr = step_LR(lr, epoch-st_sch, step_size=5, gamma=0.1,cyclic=True,lr_init=10*config.lr)
                   #lr = exp_LR(lr, epoch-st_sch, step_size=10, gamma=0.5,cyclic=True,lr_init=config.lr)
                for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

            train_loss,train_bleu = train(model, train_data_loader, optimizer, criterion, CLIP,config=config,epoch=epoch)
            val_loss,val_bleu=0,0

            if epoch>=0 :#30:
                val_loss,val_bleu,cross_attention,pred,inp= evaluate(model, val_data_loader, criterion,
                                                            mode="train",multiple_reference=False,name_file='')
                
            wandb.log({'train_loss': train_loss, 'train_bleu': train_bleu,
                       'val_loss': val_loss, 'val_bleu': val_bleu},
                        step=epoch)
            # # Use ThreadPoolExecutor to log asynchronously
            # with ThreadPoolExecutor(max_workers=1) as executor:
            #     pass

            if val_bleu >= best_valid_bleu:
                best_valid_bleu = val_bleu

                # SAVE THE PYTORCH MODEL TO DIRECTORY
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'epoch': epoch, 'val_bleu': val_bleu, 'val_loss': val_loss,'train_bleu':train_bleu,
                            'metadata':dict(config)},

                           unique_path+"_Ep_"+str(epoch))

                # -------------ARTIFACTS------------
                # model_artifact = wandb.Artifact(f"transformer_{wandb.run.id}", type="model",
                #                                 description="Transformer with controlled attention",
                #                                 metadata=dict(config))
                # # USE ADD_FILE IF YOU HAVE MULTIPLE FILES WITH DIFFERENT CONFIGS
                # model_artifact.add_file(unique_path)
                #
                # # # SAVE MODEL FROM THE GIVEN PATH TO WANDB CLOUD STORAGE
                # wandb.save(unique_path)
                #
                # # LOG ARTIFACTS
                # run.log_artifact(model_artifact)
                #
                # model_artifact.finalize()

            #     torch.save(model.state_dict(),#{'model':model.state_dict(),'best_epoch':epoch},
            #                abs_path+
            #                f'_D{config.D}_HE{config.h_enc}_HD{config.h_dec}_PF{PF_DIM}'
            #                f'_dm_{config.hid_dim}_r{config.r}_m{config.margin}_lr{config.lr}_Lf{config.L2}_L1{config.L1}.pt')


            if epoch >= N_EPOCHS-5 : # because torch.save may slow down training at each step
                # save last checkpoint :
                torch.save({'model_state': model.state_dict(),'epoch': epoch,'optimizer_state': optimizer.state_dict(),'train_bleu':train_bleu},
                           abs_path +
                           f'Last_D{config.D}_HE{config.h_enc}_HD{config.h_dec}_PF{PF_DIM}'
                           f'_dm_{config.hid_dim}_r{config.r}_m{config.margin}_lr{config.lr}_Lf{config.L2}_L0{config.L0}_L1{config.L1}.pt')

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Argument Parser for D and r")
    parser.add_argument("--dataset_name",type=str,default="h3D",choices=["h3D","kit"])
    parser.add_argument("-D", type=int, default=None, help="Half cross-window length")
    parser.add_argument("-r", type=int, default=None, help="Self-context length")
    parser.add_argument("--resume", type=int,default=0, help="Resume training")
    parser.add_argument("-c","--config", default="./configs/Transformer.yaml", help="config file for sweeping")

    args = parser.parse_args()

    # Set the config file defining the search space
    yaml_path = r".\configs\Transformer.yaml"
    abs_path = r"ABS PATH" # Define the absolute path

    os.makedirs(abs_path,exist_ok=True)
    with open(yaml_path,"r") as f:
        sweep_config = yaml.safe_load(f)

    # -----------------------------------------------------------------------

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # args.dataset_name = "h3D"

    # if args.dataset_name=="kit":
    #     # -------------KIT IMPORTS------------------
    #     from datasets.kit_m2t_dataset import kitm2l
    #     path_txt = "/home/karim/PycharmProjects/HumanML3D/sentences_corrections.csv"
    #
    # elif args.dataset_name=="h3D":
    #     # -----------H3D IMPORTS---------------------
    #     from datasets.h3d_m2t_dataset_ import kitm2l
    #     path_txt = "/home/karim/PycharmProjects/m2t/sentences_corrections_h3d.csv"

    # -------------------- Sweep hyperparameters with wandb -----------------#

    if args.D is not None and args.r is not None:
        # Update the 'D' and 'r' values for parallel runs only
        sweep_config['parameters']['D']['value'] = args.D
        sweep_config['parameters']['r']['value'] = args.r

    # Get the sweep id
    PROJECT_NAME = "MC_Synch-Transformer"
    sweep_id = wandb.sweep(sweep_config,project=PROJECT_NAME)

    #-------------------- RUN THE AGENT ----------------------------------------#

    wandb.agent(sweep_id,function=train_tune,count=10)