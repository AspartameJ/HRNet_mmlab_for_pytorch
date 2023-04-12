# Copyright (c) OpenMMLab. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import copy
import os
import os.path as osp
import time
import json
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    # 20230407
    start_time_all = time.time()
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0]+'-'+str(time.time()))

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    val_dataset = copy.deepcopy(cfg.data.val)
    val_dataset.pipeline = cfg.data.train.pipeline
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False, num_gpus=1),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
        **cfg.data.get('val_dataloader', {})
    }
    print(dataset)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    # 20230407
    sample_num = len(data_loader)
    
    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    
    # 20230407
    start_time_infer = time.time()
    
    outputs = single_gpu_test(model, data_loader)
    
    # 20230407
    end_time_infer = time.time()
    
    eval_config = cfg.get('evaluation', {})
    eval_config = merge_configs(eval_config, dict(metric=args.eval))

    results = dataset.evaluate(outputs, cfg.work_dir, **eval_config)
    '''
    for k, v in sorted(results.items()):
        print(f'{k}: {v}')
    '''
    
    # 20230407
    end_time_all = time.time()
    #fps
    all_time = end_time_all - start_time_all
    fps = sample_num / all_time
    #lantency
    latency = 1 / fps
    AP = results['AP']
    AR = results['AR']
    log_dict = {}
    end_dict = {}
    log_dict['start_time_all'] = '{:.0f}'.format(start_time_all)
    log_dict['start_time_infer'] = '{:.0f}'.format(start_time_infer)
    log_dict['end_time_infer'] = '{:.0f}'.format(end_time_infer)
    log_dict['end_time_all'] = '{:.0f}'.format(end_time_all)
    log_dict['sample_num'] = '{:.0f}'.format(sample_num)
    log_dict['infer_AP'] = '{:.4f}'.format(AP)
    log_dict['infer_AR'] = '{:.4f}'.format(AR)
    log_dict['samples/sec'] = '{:.4f}'.format(fps)
    log_dict['latency'] = '{:.4f}'.format(latency)
    end_dict['event'] = 'INFER_END'
    end_dict['value'] = log_dict

    infer_result = om_infer_result(os.path.join(cfg.work_dir,'result_keypoints.json'), cfg.data.test.ann_file)
    
    infer_result.append(json.dumps(end_dict))
    with open('log/infer_result.log','a+') as f:
        f.writelines(infer_result)
    

def om_infer_result(result_keypoints, person_keypoints_val2017):
    log_dict = {}
    result_dict = {}
    with open(result_keypoints,'r',encoding='utf-8') as f:
        json_list = json.load(f)

        for result in json_list:
            points_len = len(result['keypoints'])
            format_kepoints = []

            for i in range(0, points_len, 3):
                format_kepoints.append(int('{:.0f}'.format(result['keypoints'][i])))
                format_kepoints.append(int('{:.0f}'.format(result['keypoints'][i+1])))

            if result['image_id'] in log_dict:
                log_dict[result['image_id']]['keypoints'].append(format_kepoints)
                log_dict[result['image_id']]['score'].append(result['score'])
            else:
                log_dict[result['image_id']] = {'id': result['image_id']}
                log_dict[result['image_id']]['keypoints'] = [format_kepoints]
                log_dict[result['image_id']]['score'] = [result['score']]
                log_dict[result['image_id']]['target'] = []

    with open(person_keypoints_val2017,'r',encoding='utf-8') as ff:
        json_list = json.load(ff)['annotations']

        for val in json_list:
            if (val['image_id'] in log_dict) and (val['num_keypoints'] > 0):
                points_len = len(val['keypoints'])
                format_kepoints = []

                for i in range(0, points_len, 3):
                    format_kepoints.append(val['keypoints'][i])
                    format_kepoints.append(val['keypoints'][i+1]) 

                log_dict[val['image_id']]['target'].append(format_kepoints)

    result_list = []
    for v in log_dict.values():
        result_dict['event'] = 'INFER_RESULT'
        result_dict['value'] = v
        result_list.append(json.dumps(result_dict)+'\n')

    log_dict.clear()
    return result_list

if __name__ == '__main__':
    main()

