import torch
import argparse
from datasets.visualization import decode_predictions_and_compute_bleu_score
from datasets.loader import build_data
import torch.nn as nn
from architectures.Transformer import create_model
from torch.nn.utils.rnn import pad_sequence

def load_transformer_model(model_dict,args,vocab_size_unk=None,load_data=True):
    # convert config dict to an object namedtuple allowing accessing via attribute
    from collections import namedtuple
    config = model_dict["metadata"]
    _Config = namedtuple('Config', config.keys())
    config = _Config(**config)
    print(config)
    batch_size = args.batch_size
    device = torch.device(args.device)

    project_path = r"C:\Users\karim\PycharmProjects\SemMotion"
    aug_path = r"C:\Users\karim\PycharmProjects\HumanML3D"
    if "kit" in args.dataset_name:
        # ------------ [Augmented-KIT] ------------
        from datasets.kit_m2t_dataset import dataset_class
        path_txt = project_path+"\datasets\sentences_corrections.csv"
        path_motion = aug_path+"\kit_with_splits_2023.npz"
        n_joint = 21
        # -----------  [HumanML3D]   ---------------------
    elif args.dataset_name=="h3D":
        from datasets.h3d_m2t_dataset_ import dataset_class
        path_txt = aug_path+"\sentences_corrections_h3d.csv"
        path_motion = aug_path+"\\all_humanML3D.npz"
        n_joint = 22

    #path_txt = r"/home/karim/PycharmProjects/m2t/sentences_corrections_h3d.csv"
    if load_data:
        train_data_loader, val_data_loader, test_data_loader = build_data(dataset_class=dataset_class, path=path_motion,min_freq=3,
                                                                          train_batch_size=batch_size,test_batch_size=batch_size,
                                                                          return_lengths=True, path_txt=path_txt,
                                                                          return_trg_len=True, joint_angles=False,
                                                                          multiple_references=True)
        vocab_size_unk = train_data_loader.dataset.lang.vocab_size_unk
    else:
        train_data_loader, val_data_loader, test_data_loader= None,None,None

    # -----------------------------CREATE THE MODEL ---------------------------------------
    INPUT_DIM = n_joint * 3
    OUTPUT_DIM = vocab_size_unk
    HID_DIM = config.hid_dim

    ENC_LAYERS = DEC_LAYERS = 1
    # ENC_LAYERS = DEC_LAYERS = 6
    # ENC_LAYERS = 3; DEC_LAYERS = 1

    ENC_HEADS = config.h_enc
    DEC_HEADS = config.h_dec
    F = 2
    ENC_PF_DIM = F * config.hid_dim
    DEC_PF_DIM = F * config.hid_dim
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    MAX_TRG_LEN = config.max_length
    loaded_model = create_model(OUTPUT_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT,
                                INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT,
                                device,n_joint=n_joint, max_length=MAX_TRG_LEN, D=config.D, r=config.r,
                                margin=config.margin, spatial=config.spatial, concat=config.concat)

    # NEW LOADING
    loaded_model.load_state_dict(model_dict["model"])

    return loaded_model, train_data_loader, test_data_loader,val_data_loader

def evaluate(model, iterator, criterion, num_grams=4,mode='val', multiple_reference=False,name_file='',
             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),dataset_name="h3D" ):
    model.eval()
    BLEU_scores = []
    epoch_loss = 0
    global pred,inp
    if len(name_file)!=0:
        name_file = "./Predictions/Ablations_"+dataset_name+"/"+name_file
        with open(name_file + ".csv", mode="w") as _:
            pass # create or/and clean the file

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].to(device).type(torch.float)
            trg = batch[1].to(device) if not multiple_reference else \
                pad_sequence([torch.as_tensor(refs[0]) for refs in batch[1]],batch_first=True, padding_value=0).to(device)
            src_len = torch.tensor(batch[2]).unsqueeze(1)

            # RUN MODEL EVAL ------------------------------------------------------------------------

            output, cross_attention = model(src, trg[:, :-1], src_len,mode=mode)

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]
            output_dim = output.shape[-1]
            output_logits = output

            trg_out =  batch[1] if multiple_reference else trg
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            vocab_obj = iterator.dataset.lang
            bleu_score, pred , inp = decode_predictions_and_compute_bleu_score(output_logits, trg_out, vocab_obj, num_grams=num_grams,
                                                                          batch_first=True,multiple_references=multiple_reference)
            if name_file:
                with open(name_file+".csv",mode="a", encoding="utf-8") as f:
                 for p,t in zip(pred,inp):
                     f.writelines(("%s"+",%s"*len(t)+"\n")% ((" ".join(p).replace("\n",""),)+
                                    tuple(" ".join(k).replace("\n","") for k in t)))
            BLEU_scores += [bleu_score]
            print(f"Loss/Val_{mode}_batch %d --> %.3f  BLEU score_batch %.3f\r" % (i, loss, bleu_score), end='')
            torch.cuda.empty_cache()

    bleu_epoch = (sum(BLEU_scores) / len(BLEU_scores))
    print(f"\n \033[1;32m BLEU Val_{mode} ---- > %.3f" % bleu_epoch)
    if len(name_file)!=0: print(f"Predictions saved at {name_file}"+".csv")
    return epoch_loss / len(iterator),bleu_epoch,cross_attention,pred,inp


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Argument Parser Eval")
    parser.add_argument("--dataset_name",type=str,default="kit",choices=["h3D","kit"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--run_id",type=str)

    args = parser.parse_args()

    # Fix manually ----
    args.dataset_name = "h3D"
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args.device)

    abs_path = "define abs project path"

    best_epoch = 497
    run_id = "Your wandb run id"

    PROJECT_NAME = "Your wandb Project Name"


    model_path = abs_path + PROJECT_NAME + "model_" + run_id
    model_dict = torch.load(model_path)
    print(model_dict.keys(), model_dict['epoch'], model_dict['val_bleu'], model_dict['val_loss'])

    #------------------------------- Meta data for File Name ----------------------------------
    L0,L1,L2 = [str(model_dict["metadata"][l])+'_' for l in  ['L0','L1','L2']]
    hid_dim = '_dm_'+str(model_dict["metadata"]['hid_dim'])
    D = '_D_'+str(model_dict["metadata"]['D'])
    r = '_r_'+str(model_dict["metadata"]['r'])
    Henc = 'H_'+str(model_dict["metadata"]['h_enc'])
    Hdec = 'H_'+str(model_dict["metadata"]['h_dec'])
    assert Henc==Hdec
    complete_name_file = PROJECT_NAME[:-1]+'_'+run_id+'_L_'+L0+L1+L2+Henc+hid_dim+D+r
    #------------------------------------------------------------------------------------------

    loaded_model, train_data_loader, test_data_loader,val_data_loader = load_transformer_model(model_dict,args)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # TRG_PAD_IDX
    vocab_obj = test_data_loader.dataset.lang


    _, _, cross_attention, pred, inp = evaluate(loaded_model, test_data_loader, criterion, mode="val",
                                                multiple_reference=True, name_file="_Val_"+complete_name_file,
                                                device=torch.device("cuda"),dataset_name=args.dataset_name)


    print("parameters number:",sum(p.numel() for p in loaded_model.parameters() if p.requires_grad))

# def write_predictions_sentences():
#     with open("pred_trg.csv",mode="a") as f:
#         for p,t in zip(pred,inp):
#             f.writelines("%s,%s\n"%(" ".join(p)," ".join(t[0])))

