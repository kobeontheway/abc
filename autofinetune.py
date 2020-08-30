from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
import paddlehub as hub
import pandas as pd
import numpy as np
import sys, csv, os
import argparse
import shutil
import ast
from paddlehub.common.logger import logger

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=2, help="epochs.")
parser.add_argument("--batch_size", type=int, default=64, help="batch_size.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning_rate.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="warmup_prop.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay.")
parser.add_argument("--max_seq_len", type=int, default=128, help="max_seq_len, Number of words of the longest seqence.")
parser.add_argument("--checkpoint_dir", type=str, default="autofinetune/checkpoint_dir", help="Directory to model checkpoint")
parser.add_argument("--eval_interval", type=int, default=400, help="xx interval to evaluate by dev dataset")
parser.add_argument("--use_cuda", type=bool, default=True, help="is user gpu?")
parser.add_argument("--saved_params_dir", type=str, default="autofinetune/saved_params_dir", help="Directory for saving model during ")
parser.add_argument("--model_path", type=str, default="autofinetune/model_path", help="load model path")
args = parser.parse_args()

def is_path_valid(path):
    if path == "":
        return False
    path = os.path.abspath(path)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return True

train_data = pd.read_csv('./train_set.csv', sep='\t') #20万行
train_data['text_a'] = train_data['text']
# print(train_data['label'].value_counts()) #查看每个类别有多少样本
# {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}
csv.field_size_limit(sys.maxsize)
np.random.seed(seed=2)

#抽取10%的数据
typicalNDict={
    0:0.2,
    1:0.2,
    2:0.2,
    3:0.2,
    4:0.2,
    5:0.2,
    6:0.2,
    7:0.2,
    8:0.2,
    9:0.2,
    10:0.2,
    11:0.2,
    12:0.2,
    13:0.2
}

def typicalsamling(group, typicalNDict):
    name = group.name
    frac = typicalNDict[name]
    return group.sample(frac=frac)


# new_train_data = train_data.groupby('label').apply(typicalsamling, typicalNDict)
train_data = train_data.sample(frac=1.0)
train_data[['text_a', 'label']].iloc[:].to_csv('./data/train.csv', index=None, header=True, sep='\t')

test_data = train_data.groupby('label').apply(typicalsamling, typicalNDict)
test_data = test_data.sample(frac=1.0)
test_data['text_a'] = test_data['text']
test_data[['text_a', 'label']].iloc[:].to_csv('./data/test.csv', index=None, header=True, sep='\t')
print(test_data.head())

np.random.seed(seed=3)
dev_data = train_data.groupby('label').apply(typicalsamling, typicalNDict)
dev_data = dev_data.sample(frac=1.0)
dev_data['text_a'] = dev_data['text']
dev_data[['text_a', 'label']].iloc[:].to_csv('./data/dev.csv', index=None, header=True, sep='\t')
print(dev_data.head())
label_list=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']

predict_data = pd.read_csv('./test_a.csv', sep='\t')
predict_data['text_a'] = predict_data['text']
predict_data[['text_a']].to_csv('./data/predict.csv', index=None, header=True, sep='\t')

class ThuNews(BaseNLPDataset):
    def __init__(self):
        # 数据集存放位置
        self.dataset_dir = "./data"
        super(ThuNews, self).__init__(
            base_path=self.dataset_dir,
            train_file="train.csv",
            dev_file="dev.csv",
            test_file="test.csv",
            predict_file="predict.csv",
            train_file_with_header=True,
            dev_file_with_header=True,
            test_file_with_header=True,
            predict_file_with_header=True,
            # 数据集类别集合
            label_list=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'])

dataset = ThuNews()

module = hub.Module(name="ernie_tiny")
reader = hub.reader.ClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),
    sp_model_path=module.get_spm_path(),
    word_dict_path=module.get_word_dict_path(),
    max_seq_len=args.max_seq_len)

strategy = hub.AdamWeightDecayStrategy(
    weight_decay=args.weight_decay,
    warmup_proportion=args.warmup_proportion,
    # learning_rate=5e-5,
    lr_scheduler="linear_decay",
    learning_rate=args.learning_rate)

config = hub.RunConfig(
    use_cuda=args.use_cuda,
    use_data_parallel=True,
    num_epoch=args.num_epoch,
    checkpoint_dir=args.checkpoint_dir,
    batch_size=args.batch_size,
    eval_interval=args.eval_interval,
    strategy=strategy
)

inputs, outputs, program = module.context(
    trainable=True, max_seq_len=args.max_seq_len)

# Use "pooled_output" for classification tasks on an entire sentence.
pooled_output = outputs["pooled_output"]

feed_list = [
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name,
]

cls_task = hub.TextClassifierTask(
    data_reader=reader,
    feature=pooled_output,
    feed_list=feed_list,
    num_classes=dataset.num_labels,
    config=config,
    metrics_choices=["acc"])

if args.model_path != "":
    with cls_task.phase_guard(phase="train"):
        cls_task.init_if_necessary()
        cls_task.load_parameters(args.model_path)
        logger.info("PaddleHub has loaded model from %s" % args.model_path)

run_states = cls_task.finetune_and_eval()
eval_avg_score, eval_avg_loss, eval_run_speed = cls_task._calculate_metrics(run_states)

best_model_dir = os.path.join(config.checkpoint_dir, "best_model")
if is_path_valid(args.saved_params_dir) and os.path.exists(best_model_dir):
    shutil.copytree(best_model_dir, args.saved_params_dir)
    shutil.rmtree(config.checkpoint_dir)

hub.report_final_result(eval_avg_score["acc"])

# label_map = {val: key for key, val in reader.label_map.items()}
# predict_data_a = [[d.text_a] for d in dataset.get_predict_examples()]
# run_states = cls_task.predict(data=predict_data_a,load_best_model=True, return_result=True)

# result = pd.DataFrame()
# result['label'] = [x for x in run_states]
# result.to_csv('./submit.csv', index=None)
