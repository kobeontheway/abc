from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
import paddlehub as hub
import pandas as pd
import numpy as np
import sys, csv

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
    max_seq_len=128)

strategy = hub.AdamWeightDecayStrategy(
    weight_decay=0.01,
    warmup_proportion=0.1,
    # learning_rate=5e-5,
    lr_scheduler="linear_decay",
    learning_rate=5e-5)

config = hub.RunConfig(
    use_cuda=True,
    use_data_parallel=True,
    num_epoch=1,
    checkpoint_dir="module",
    batch_size=64,
    eval_interval=400,
    strategy=strategy
)

inputs, outputs, program = module.context(
    trainable=True, max_seq_len=128)

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

run_states = cls_task.finetune_and_eval()

# label_map = {val: key for key, val in reader.label_map.items()}
# predict_data_a = [[d.text_a] for d in dataset.get_predict_examples()]
# run_states = cls_task.predict(data=predict_data_a,load_best_model=True, return_result=True)

# result = pd.DataFrame()
# result['label'] = [x for x in run_states]
# result.to_csv('./submit.csv', index=None)
