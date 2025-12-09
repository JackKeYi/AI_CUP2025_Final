import sys
import os
import hashlib
import json
from functools import partial

# Ensure package path is correct
sys.path.append("/nfs/Workspace/CardiacSegV2")

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import ray
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.search.optuna import OptunaSearch

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    Orientationd,
    ToNumpyd,
)
from monailabel.transform.post import Restored

from networks.network import network
from expers.args import get_parser, map_args_transform, map_args_optim, map_args_lrschedule, map_args_network
from data_utils.dataset import DataLoader, get_label_names, get_infer_data
from data_utils.data_loader_utils import load_data_dict_json
from data_utils.utils import get_pids_by_loader, get_pids_by_data_dicts
from runners.tuner import run_training
from runners.inferer import run_infering
from optimizers.optimizer import Optimizer, LR_Scheduler


def main(config, args=None):
    if args.tune_mode == 'transform':
        args = map_args_transform(config, args)
        
    elif args.tune_mode == 'optim':
        args.max_epochs = args.max_epoch
        args = map_args_optim(config['optim'], args)
        if 'lrschedule' in config:
            args = map_args_lrschedule(config['lrschedule'], args)
        
        lr_val = config['optim']['lr']
        wd_val = config['optim']['weight_decay']
        tag_str = f"lr_{lr_val}_wd_{wd_val}_wu_{args.warmup_epochs}"

        args.model_dir = os.path.join(args.model_dir, tag_str)
        args.log_dir   = os.path.join(args.log_dir,   tag_str)
        args.eval_dir  = os.path.join(args.eval_dir,  tag_str)

        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.log_dir,   exist_ok=True)
        os.makedirs(args.eval_dir,  exist_ok=True)
        
        args.checkpoint = os.path.join(args.model_dir, 'final_model.pth')

    elif args.tune_mode in ['lrschedule', 'lrschedule_epoch']:
        args = map_args_transform(config['transform'], args)
        args = map_args_optim(config['optim'], args)
        args = map_args_lrschedule(config['lrschedule'], args)

    elif args.tune_mode == 'network':
        args = map_args_network(config, args)

    elif args.tune_mode == 'hpo_tpe':
        args.lr = config['lr']
        args.weight_decay = config['weight_decay']
        args.drop_rate = config['drop_rate']
        args.feature_size = config['feature_size']
        args.warmup_epochs = config['warmup_epochs']
        args.lambda_dice = config['lambda_dice']
        
        args.lambda_focal = 1.0 - args.lambda_dice
        args.max_epochs = args.max_epoch

        param_dict = {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'drop_rate': args.drop_rate,
            'feature_size': args.feature_size,
            'lambda_dice': args.lambda_dice,
            'lambda_focal': args.lambda_focal,
        }
        
        json_str = json.dumps(param_dict, sort_keys=True)
        trial_code = hashlib.md5(json_str.encode()).hexdigest()[:8]

        args.model_dir = os.path.join(args.model_dir, f'trial_{trial_code}')
        args.log_dir   = os.path.join(args.log_dir,   f'trial_{trial_code}')
        args.eval_dir  = os.path.join(args.eval_dir,  f'trial_{trial_code}')

        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.log_dir,   exist_ok=True)
        os.makedirs(args.eval_dir,  exist_ok=True)

        print('=== HPO TPE Optimization ===')
        print(f'Trial Code: trial_{trial_code}')
        print(f'Dir: {args.model_dir}')
        print(f'Params: lr={args.lr}, wd={args.weight_decay}, drop={args.drop_rate}')

    else:
        args.max_epochs = args.max_epoch
        print('a_max', args.a_max)
        print('a_min', args.a_min)
        print('space_x', args.space_x)
        print('roi_x', args.roi_x)
        print('lr', args.lr)
        print('weight_decay', args.weight_decay)
        print('warmup_epochs', args.warmup_epochs)
        print('max_epochs', args.max_epochs)
    
    if args.tune_mode == 'optim':
        args.test_mode = False
        main_worker(args)

        args.test_mode = True
        args.checkpoint = os.path.join(args.model_dir, 'best_model.pth')
        args.ssl_checkpoint = None
        main_worker(args)

    else:
        # train
        args.test_mode = False
        args.checkpoint = os.path.join(args.model_dir, 'final_model.pth')
        main_worker(args)
        
        # test
        args.test_mode = True
        args.checkpoint = os.path.join(args.model_dir, 'best_model.pth')
        args.ssl_checkpoint = None
        main_worker(args)


def main_worker(args):
    from networks.network import network
    
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if torch.cuda.is_available():
        print("cuda is available")
        args.device = torch.device("cuda")
    else:
        print("cuda is not available")
        args.device = torch.device("cpu")

    model = network(args.model_name, args)
    
    if args.loss == 'dice_focal_loss':
        print('loss: dice focal loss')
        dice_loss = DiceFocalLoss(
            to_onehot_y=True, 
            softmax=True,
            gamma=2.0,
            lambda_dice=args.lambda_dice,
            lambda_focal=args.lambda_focal
        )
    else:
        print('loss: dice ce loss')
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)
    
    print(f'optimzer: {args.optim}')
    optimizer = Optimizer(args.optim, model.parameters(), args)

    if args.lrschedule is not None:
        print(f'lrschedule: {args.lrschedule}')
        scheduler = LR_Scheduler(args.lrschedule, optimizer, args)
    else:
        scheduler = None

    start_epoch = args.start_epoch
    early_stop_count = args.early_stop_count
    best_acc = 0
    
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if args.lrschedule is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        if "early_stop_count" in checkpoint:
            early_stop_count = checkpoint["early_stop_count"]
            
        print(
          "=> loaded checkpoint '{}' (epoch {}) (bestacc {}) (early stop count {})"\
          .format(args.checkpoint, start_epoch, best_acc, early_stop_count)
        )
    else:
        # SSL Pretrain Logic
        if args.model_name =='swinunetr' and args.ssl_checkpoint and os.path.exists(args.ssl_checkpoint):
            pre_train_path = os.path.join(args.ssl_checkpoint)
            weight = torch.load(pre_train_path)
            
            if "net" in list(weight["state_dict"].keys())[0]:
                print("Tag 'net' found in state dict - fixing!")
                for key in list(weight["state_dict"].keys()):
                    if 'swinViT' in key:
                        new_key = key.replace("net.swinViT", "module")
                        weight["state_dict"][new_key] = weight["state_dict"].pop(key) 
                    else:
                        new_key = key.replace("net", "module")
                        weight["state_dict"][new_key] = weight["state_dict"].pop(key)

                    if 'linear' in  new_key:
                        weight["state_dict"][new_key.replace("linear", "fc")] = weight["state_dict"].pop(new_key)
            
            model.load_from(weights=weight)
            print("Using pretrained self-supervied Swin UNETR backbone weights !")
            print("=> loaded pretrain checkpoint '{}'".format(args.ssl_checkpoint))
            
        elif args.ssl_checkpoint and os.path.exists(args.ssl_checkpoint):
            model_dict = torch.load(args.ssl_checkpoint)
            state_dict = model_dict["state_dict"]
            if "module." in list(state_dict.keys())[0]:
                print("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            if "swin_vit" in list(state_dict.keys())[0]:
                print("Tag 'swin_vit' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
            if "net" in list(state_dict.keys())[0]:
                print("Tag 'net' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("net.", "")] = state_dict.pop(key)
            
            model.load_state_dict(state_dict, strict=False)
            print(f"Using pretrained self-supervied {args.model_name} backbone weights !")
            print("=> loaded pretrain checkpoint '{}'".format(args.ssl_checkpoint))
            
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    writer = SummaryWriter(log_dir=args.log_dir)

    if not args.test_mode:
        loader = DataLoader(args.data_name, args)()
        tr_loader, val_loader = loader
        
        run_training(
            start_epoch=start_epoch,
            best_acc=best_acc,
            early_stop_count=early_stop_count,
            model=model,
            train_loader=tr_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_func=dice_loss,
            acc_func=dice_acc,
            model_inferer=model_inferer,
            post_label=post_label,
            post_pred=post_pred,
            writer=writer,
            args=args,
        )
    
    else:
        os.makedirs(args.eval_dir, exist_ok=True)
        label_names = get_label_names(args.data_name)
        
        _, _, test_dicts = load_data_dict_json(args.data_dir, args.data_dicts_json)
        
        keys = ['pred']
        if args.data_name == 'mmwhs' or args.data_name == 'mmwhs2':
            axcodes='LAS'
        else:
            axcodes='LPS'
            
        post_transform = Compose([
            Orientationd(keys=keys, axcodes=axcodes),
            ToNumpyd(keys=keys),
            Restored(keys=keys, ref_image="image")
        ])
       
        pids = get_pids_by_data_dicts(test_dicts)
        inf_dc_vals = []
        inf_iou_vals = []
        inf_sensitivity_vals = []
        inf_specificity_vals = []
        tt_dc_vals = []
        tt_iou_vals = []
        inf_times = []
        
        for data_dict in test_dicts:
            print('infer data:', data_dict)
            data = get_infer_data(data_dict, args)
            
            ret_dict = run_infering(
                model,
                data,
                model_inferer,
                post_transform,
                args
            )
            tt_dc_vals.append(ret_dict['tta_dc'])
            tt_iou_vals.append(ret_dict['tta_iou'])
            inf_dc_vals.append(ret_dict['ori_dc'])
            inf_iou_vals.append(ret_dict['ori_iou'])
            inf_sensitivity_vals.append(ret_dict['ori_sensitivity'])
            inf_specificity_vals.append(ret_dict['ori_specificity'])
            inf_times.append(ret_dict['inf_time'])
            
        eval_tt_dice_val_df = pd.DataFrame(tt_dc_vals, columns=[f'tt_dice{n}' for n in label_names])
        eval_tt_iou_val_df = pd.DataFrame(tt_iou_vals, columns=[f'tt_iou{n}' for n in label_names])
        eval_inf_dice_val_df = pd.DataFrame(inf_dc_vals, columns=[f'inf_dice{n}' for n in label_names])
        eval_inf_iou_val_df = pd.DataFrame(inf_iou_vals, columns=[f'inf_iou{n}' for n in label_names])
        eval_inf_sensitivity_val_df = pd.DataFrame(inf_sensitivity_vals, columns=[f'inf_sensitivity{n}' for n in label_names])
        eval_inf_specificity_val_df = pd.DataFrame(inf_specificity_vals, columns=[f'inf_specificity{n}' for n in label_names])
        eval_inf_time_df = pd.DataFrame(inf_times, columns=[f'inf_time'])
        pid_df = pd.DataFrame({'patientId': pids})
        
        avg_tt_dice = eval_tt_dice_val_df.T.mean().mean()
        avg_tt_iou =  eval_tt_iou_val_df.T.mean().mean()
        avg_inf_dice = eval_inf_dice_val_df.T.mean().mean()
        avg_inf_iou =  eval_inf_iou_val_df.T.mean().mean()
        avg_inf_sensitivity =  eval_inf_sensitivity_val_df.T.mean().mean()
        avg_inf_specificity =  eval_inf_specificity_val_df.T.mean().mean()
        avg_inf_time = eval_inf_time_df.T.mean().mean()

        eval_df = pd.concat([
            pid_df, eval_tt_dice_val_df, eval_tt_iou_val_df,
            eval_inf_dice_val_df, eval_inf_iou_val_df,
            eval_inf_sensitivity_val_df, eval_inf_specificity_val_df,eval_inf_time_df
        ], axis=1, join='inner').reset_index(drop=True)
        
        if args.save_eval_csv:
            eval_df.to_csv(os.path.join(args.eval_dir, f'best_model.csv'), index=False)
        
        print("\neval result:")
        print('avg tt dice:', avg_tt_dice)
        print('avg tt iou:', avg_tt_iou)
        print('avg inf dice:', avg_inf_dice)
        print('avg inf iou:', avg_inf_iou)
        print('avg inf sensitivity:', avg_inf_sensitivity)
        print('avg inf specificity:', avg_inf_specificity)
        print('avg inf time:', avg_inf_time)
        
        print(eval_df.to_string())
        
        tune.report(
            tt_dice=avg_tt_dice,
            tt_iou=avg_tt_iou,
            inf_dice=avg_inf_dice,
            inf_iou=avg_inf_iou,
            val_bst_acc=best_acc,
            inf_time=avg_inf_time
        )


if __name__ == "__main__":
    args = get_parser(sys.argv[1:])
    
    param_space_cfg = {}

    if args.tune_mode == 'test':
        print('test mode')
        
    elif args.tune_mode == 'train':
        param_space_cfg = {
            "exp": tune.grid_search([{'exp': args.exp_name}])
        }
        
    elif args.tune_mode == 'network':
        param_space_cfg = {'depths': tune.grid_search([[4, 4, 12, 4]])}
        
    elif args.tune_mode == 'transform':
        param_space_cfg = {
            'intensity': tune.grid_search([[-42, 423]]),
            'space': tune.grid_search([
                [0.4,0.4,0.5], [0.8,0.8,0.8], [0.8,0.8,1.0], [1.0,1.0,1.0]
            ]),
            'roi': tune.grid_search([[128,128,128]]),
        }
        
    elif args.tune_mode == 'optim':
        param_space_cfg = {
            "optim": {
                "lr": tune.loguniform(5e-5, 3e-4),
                "weight_decay": tune.loguniform(3e-5, 3e-4),
            },
            "lrschedule": {
                "warmup_epochs": tune.choice([10, 15, 20]),
                "max_epoch": args.max_epoch,
            }
        }
        
    elif args.tune_mode == 'lrschedule':
        param_space_cfg = {
            'transform': tune.grid_search([{
                'intensity': [-42,423], 'space': [0.7,0.7,1.0], 'roi':[128,128,128]
            }]),
            'optim': tune.grid_search([
                {'lr':5e-5, 'weight_decay': 5e-4},
                {'lr':5e-4, 'weight_decay': 5e-5},
                {'lr':1e-4, 'weight_decay': 1e-3},
            ]),
            'lrschedule': tune.grid_search([{'warmup_epochs':60,'max_epoch':1200}])
        }
        
    elif args.tune_mode == 'lrschedule_epoch':
        param_space_cfg = {
            'transform': tune.grid_search([{
                'intensity': [-42,423], 'space': [1.0,1.0,1.0], 'roi':[128,128,128]
            }]),
            'optim': tune.grid_search([
                {'lr':1e-2, 'weight_decay': 3e-5},
                {'lr':5e-3, 'weight_decay': 5e-4},
                {'lr':5e-4, 'weight_decay': 5e-5},
            ]),
            'lrschedule': tune.grid_search([
                {'warmup_epochs':40,'max_epoch':700},
                {'warmup_epochs':60,'max_epoch':700},
            ])
        }

    elif args.tune_mode == 'hpo_tpe':
        param_space_cfg = {
            'lr': tune.loguniform(1e-5, 1e-3),
            'weight_decay': tune.loguniform(1e-5, 1e-3),
            'drop_rate': tune.uniform(0.0, 0.3),
            'feature_size': tune.choice([24, 48, 96]),
            'warmup_epochs': tune.choice([10, 15, 20, 25, 30]),
            'lambda_dice': tune.uniform(0.1, 0.5),
        }
        
    else:
        raise ValueError(f"Invalid args tune mode:{args.tune_mode}")

    trial_executor = tune.with_resources(partial(main, args=args), {"cpu": 1, "gpu": 1})
    
    reporter = CLIReporter(metric_columns=[
        'tt_dice', 'tt_iou', 'inf_dice', 'inf_iou', 'val_bst_acc', 'esc', 'inf_time'
    ])

    run_path = os.path.join(args.root_exp_dir, args.exp_name)
    should_resume = os.path.exists(run_path) and len(os.listdir(run_path)) > 0
    
    if should_resume and args.tune_mode != 'test':
        print(f'resume tuner from {args.root_exp_dir}')
        ray_tuner = tune.Tuner.restore(
            path=os.path.join(args.root_exp_dir, args.exp_name),
            trainable=trial_executor,
            param_space=param_space_cfg
        )
        
        if args.tune_mode == 'test':
            # Manual test block
            print('run test mode ...')
            res_grid = ray_tuner.get_results()
            bst_res = res_grid.get_best_result(metric="inf_dice", mode="max")
            mod_pth = os.path.join(bst_res.log_dir, 'models', 'best_model.pth')
            
            args.max_epochs = args.max_epoch
            args.test_mode = True
            args.checkpoint = os.path.join(mod_pth)
            args.eval_dir = os.path.join(bst_res.log_dir, 'evals')
            
            main_worker(args)
        else:
            final_res = ray_tuner.fit()
            
    else:
        tpe_config = tune.TuneConfig()

        if args.tune_mode == "hpo_tpe":
            tpe_strategy = OptunaSearch(metric="val_bst_acc", mode="max")
            
            tpe_config = tune.TuneConfig(
                search_alg=tpe_strategy,
                num_samples=10, 
            )
        
        if args.tune_mode != 'test':
            print('=== Starting New Tuner ===')
            
            ray_tuner = tune.Tuner(
                trial_executor,
                param_space=param_space_cfg,
                tune_config=tpe_config,
                run_config=air.RunConfig(
                    name=args.exp_name,
                    local_dir=args.root_exp_dir,
                    progress_reporter=reporter,
                    failure_config=air.FailureConfig(max_failures=1)
                ),
            )
            
        final_results = None
        if args.tune_mode == 'test':
            print('run test mode ...')
            main_worker(args)
        else:
            final_results = ray_tuner.fit()
            
        if args.tune_mode == "hpo_tpe" and final_results is not None:
            best_res = final_results.get_best_result(metric="val_bst_acc", mode="max")
            
            best_dice_w = best_res.config['lambda_dice']
            best_focal_w = 1.0 - best_dice_w
            
            print("\n" + "="*70)
            print("Best hyperparameters found (hpo_tpe):")
            print("="*70)
            print(f"  - lr:           {best_res.config['lr']:.6e}")
            print(f"  - weight_decay: {best_res.config['weight_decay']:.6e}")
            print(f"  - drop_rate:    {best_res.config['drop_rate']:.4f}")
            print(f"  - feature_size: {best_res.config['feature_size']}")
            print(f"  - warmup_epochs:{best_res.config['warmup_epochs']}")
            print(f"  - lambda_dice:  {best_dice_w:.4f}")
            print(f"  - lambda_focal: {best_focal_w:.4f}")
            print(f"\n  - val_bst_acc:  {best_res.metrics['val_bst_acc']:.4f}")
            print("="*70)

            best_cfg_path = os.path.join(args.root_exp_dir, args.exp_name, 'optimal_config.json')
            
            cfg_save = best_res.config.copy()
            cfg_save['lambda_focal'] = best_focal_w
            
            with open(best_cfg_path, 'w') as f:
                json.dump({
                    'config': cfg_save,
                    'metrics': best_res.metrics,
                    'log_dir': best_res.log_dir
                }, f, indent=4)
            print(f"\nOptimal configuration saved to: {best_cfg_path}")