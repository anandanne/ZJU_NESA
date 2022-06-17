from datasets import load_dataset
import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


'''fun-tine process'''
# get data to fine tune
train_dataset = load_dataset("csv",data_files="data/yelp.csv", split="train[:500]")
dev_dataset = load_dataset("csv",data_files="data/yelp.csv", split="train[500:960]")
test_dataset = load_dataset("csv",data_files="data/yelp.csv", split="train[960:]")
# print(test_dataset['text'])
# print(train_dataset)
# print(train_dataset.features)
# # dataset类型，类似于一张表，列标签有label，text

train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
dev_dataset = dev_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
            num_labels=2)
tokenizer =AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


MAX_LENGTH = 256
train_dataset = train_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
dev_dataset = dev_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
test_dataset = test_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
# # map:对dataset中每一条数据应用指定的函数，产生一个新的数据集，每一条数据都有不同的字段

train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
dev_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
print(train_dataset.features)
print(train_dataset[0])


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    learning_rate=3e-4,
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=12,  # batch size per device during training
    per_device_eval_batch_size=12,   # batch size for evaluation
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    do_train=True,
    do_eval=True,
    no_cuda=False,
    load_best_model_at_end=True,
    # eval_steps=100,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,                         # the instantiated ? Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset,            # evaluation dataset
    compute_metrics=compute_metrics
)

train_out = trainer.train()


# torch.save(model, mymodel.pth)#保存整个model的状态
# ckpt = torch.load("/home/wrc/attack_on_prompt_learning/twitter_sentiment/TextFooler-master/results/checkpoint-14/pytorch_model.bin", map_location='cpu')  # Load your best model
# model.load_state_dict(ckpt)
# model.eval()

model = model.cpu()
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
d={'LABEL_0':0,'LABEL_1':1}
correct=0
total=0
for idx in range(len(test_dataset['text'])):
   
    text=test_dataset['text'][idx]
    label=test_dataset['label'][idx]
    try:
        correct+=d[classifier(text)[0]['label']]==label
        total+=1
    except:
        pass

# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10)
# use_cuda = torch.cuda.is_available()

# total_acc_test = 0
# with torch.no_grad():
#     for i in test_dataset:
#             mask = test_input['attention_mask']
#             input_id = test_input['input_ids'].squeeze(1)
#             output = model(input_id, mask)
#             acc = (output.argmax(dim=1) == test_label).sum().item()
#             total_acc_test += acc   
# print(f'Test Accuracy: {total_acc_test / 40: .3f}')

# result = classifier(test_dataset['text'])
# if result==test_dataset['label'][0]:
#     correct+=1
# print("test acc",correct/len(test_dataset['label']))
print(correct/total)