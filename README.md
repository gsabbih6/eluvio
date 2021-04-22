# Sentence Classification with TF.DATA API and Bert.

This is a sentence classifier which can be modidfed for any need. It is based of the TF.DATA API for building the pipeline to hand large dataset and BERT for
feature extraction.

# Usage
1. Download the jupyter notebook file "eluvio.ipynb" in the directory and update the following section


``` python
directory_url = 'https://drive.google.com/file/d/15X00ZWBjla7qGOIW33j8865QdF89IyAk/view'
file_name = '/content/Eluvio_DS_Challenge.csv'
BERT_TFHUB_URL = "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/4"
max_length = 128
batch_size=4
label_data = ['worldnews']  # fill with the list of categories or labels for each sentence
```

```python
def preprocess(sample, lable):
    print(sample['title']). # change 'title' to the header name of the sentence in your file
    tokenized_inputs = preprocessor(
        sample['title'])  # preprocessor.bert_pack_inputs([tokenized_inputs,tokenized_inputs],tf.constant(max_length))
    
    print(lable)
    label = one_hot(lable)
    print(tokenized_inputs)
    print('labe:',label)
    return tokenized_inputs, tf.reshape(label,(batch_size,1))

dataset = tf.data.experimental.make_csv_dataset(
    file_name,
    # select_columns=['title'],
    select_columns=['title', 'category'], # change 'title' and 'category' to the header names of the sentence and its category in your file
    batch_size=batch_size, shuffle_seed=123,
    label_name="category"
)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.map(preprocess)  # preprocess each text of the dataset
train_size = int(0.9 * DATASET_SIZE)  # train batch
val_size = int(0.1 * DATASET_SIZE)  # test/val batch
```

3. RUN the script and wait for your results.
4. Adjust the model nyperparameters as needed
