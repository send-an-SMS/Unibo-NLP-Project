random_seed = 42
transformers.set_seed(random_seed)
torch.manual_seed(random_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
arguments_train_path = 'arguments-training.tsv'
arguments_val_path = 'arguments-validation.tsv'
arguments_test_path = 'arguments-validation-zhihu.tsv'
labels_train_path = 'labels-training.tsv'
labels_val_path = 'labels-validation.tsv'
labels_test_path = 'labels-validation-zhihu.tsv'
df_arguments_train = pd.read_csv(arguments_train_path, sep='\t')
df_labels_train = pd.read_csv(labels_train_path, sep='\t')
df_train = pd.merge(df_arguments_train, df_labels_train, on='Argument ID')
df_arguments_val = pd.read_csv(arguments_val_path, sep='\t')
df_labels_val = pd.read_csv(labels_val_path, sep='\t')
df_val = pd.merge(df_arguments_val, df_labels_val, on='Argument ID')
df_arguments_test = pd.read_csv(arguments_test_path, sep='\t')
df_labels_test = pd.read_csv(labels_test_path, sep='\t')
df_test = pd.merge(df_arguments_test, df_labels_test, on='Argument ID')
columns = df_train.columns.tolist()
replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
good_symbols_re = re.compile('[^0-9a-z #+_]')
replace_multiple_spaces_re = re.compile(' +')
good_stopwords = ['favor','against']
try:
    stopwords = set(stopwords.words('english'))
    print(stopwords)
    stopwords = stopwords - set(good_stopwords)
    print(stopwords)
except LookupError:
    nltk.download('stopwords')
    stopwords = set(stopwords.words('english'))
    stopwords = stopwords - set(good_stopwords)
def lower(text: str) -> str:
    return text.lower()
def replace_special_characters(text: str) -> str:
    return replace_by_space_re.sub(' ', text)
def replace_br(text: str) -> str:
    return text.replace('br', '')
def filter_out_uncommon_symbols(text: str) -> str:
    return good_symbols_re.sub('', text)
def remove_stopwords(text: str) -> str:
    return ' '.join([x for x in text.split() if x and x not in stopwords])
def strip_text(text: str) -> str:
    return text.strip()
def replace_double_spaces(text: str) -> str:
    return replace_multiple_spaces_re.sub(' ', text)
preprocessing_pipeline = [
                          lower,
                          replace_special_characters,
                          replace_br,
                          filter_out_uncommon_symbols,
                          remove_stopwords,
                          strip_text,
                          replace_double_spaces
                          ]
def text_prepare(text: str, filter_methods: List[Callable[[str], str]] = None) -> str:
    filter_methods = filter_methods if filter_methods is not None else preprocessing_pipeline
    return reduce(lambda txt, f: f(txt), filter_methods, text)
df_train['Conclusion'] = df_train['Conclusion'].apply(lambda txt: text_prepare(txt))
df_train['Stance'] = df_train['Stance'].apply(lambda txt: text_prepare(txt))
df_train['Premise'] = df_train['Premise'].apply(lambda txt: text_prepare(txt))
df_val['Conclusion'] = df_val['Conclusion'].apply(lambda txt: text_prepare(txt))
df_val['Stance'] = df_val['Stance'].apply(lambda txt: text_prepare(txt))
df_val['Premise'] = df_val['Premise'].apply(lambda txt: text_prepare(txt))
df_test['Conclusion'] = df_test['Conclusion'].apply(lambda txt: text_prepare(txt))
df_test['Stance'] = df_test['Stance'].apply(lambda txt: text_prepare(txt))
df_test['Premise'] = df_test['Premise'].apply(lambda txt: text_prepare(txt))
max_length = 94
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True, max_length=max_length)
def tokenize_and_encode(samples):   
    input_list = [samples.get(key) for key in ['Conclusion', 'Stance', 'Premise']]
    input = ' '.join(input_list)
    return tokenizer(input) 
def convert_to_dataset(train_dataframe, val_dataframe, test_dataframe, columns):
    train_dataset = Dataset.from_dict((train_dataframe[columns]).to_dict('list'))
    val_dataset = Dataset.from_dict((val_dataframe[columns]).to_dict('list'))
    test_dataset = Dataset.from_dict((test_dataframe[columns]).to_dict('list')) 
    ds = DatasetDict()
    ds['train'] = train_dataset
    ds['eval'] = val_dataset
    ds['test'] = test_dataset
    ds = ds.map(lambda x: {"labels": [float(x[c]) for c in columns[4:]]})
    ds_enc = ds.map(tokenize_and_encode, remove_columns=columns)
    return ds_enc, columns[4:]
ds_enc, labels = convert_to_dataset(df_train, df_val, df_test, columns)
num_labels = len(labels)
model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=num_labels
                                                           )
model = model.to(device)
batch_size_train = 128 
batch_size_eval = 128
id2label = {
 0:'Self-direction: thought',
 1:'Self-direction: action',
 2:'Stimulation',
 3:'Hedonism',
 4:'Achievement',
 5:'Power: dominance',
 6:'Power: resources',
 7:'Face',
 8:'Security: personal',
 9:'Security: societal',
 10:'Tradition',
 11:'Conformity: rules',
 12:'Conformity: interpersonal',
 13:'Humility',
 14:'Benevolence: caring',
 15:'Benevolence: dependability',
 16:'Universalism: concern',
 17:'Universalism: nature',
 18:'Universalism: tolerance',
 19:'Universalism: objectivity'}
def classification_report_per_label(y_pred, y_true, value_classes, thresh=0.5, sigmoid=True):
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_true = y_true.bool().numpy()
    y_pred = (y_pred > thresh).numpy()
    report = {}
    for i, v in enumerate(value_classes):
        report[v] = classification_report(y_true=y_true[:, i], y_pred=y_pred[:, i], zero_division=0) 
    return report
def compute_metrics(eval_pred, value_classes):
    """Custom metric calculation function for MultiLabelTrainer"""
    predictions, labels = eval_pred
    reports = classification_report_per_label(predictions, labels, value_classes)
    return reports  
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience = 3, 
    early_stopping_threshold = 0.001 
)
args = TrainingArguments(
                         model_name,
                         evaluation_strategy = "epoch",
                         save_strategy = "epoch", 
                         logging_strategy='epoch',
                         learning_rate=2e-5, 
                         per_device_train_batch_size=batch_size_train,
                         per_device_eval_batch_size=batch_size_eval,
                         num_train_epochs=20,
                         weight_decay=0.1, 
                         load_best_model_at_end=True, 
                         seed=42,
                         )
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = torch.nn.BCEWithLogitsLoss()(logits,labels)
        return (loss, outputs) if return_outputs else loss
trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=ds_enc["train"],
    eval_dataset=ds_enc["eval"],
    tokenizer=tokenizer,
    compute_metrics=lambda x: compute_metrics(x, labels),
    callbacks=[early_stopping_callback]  
)
trainer.train()