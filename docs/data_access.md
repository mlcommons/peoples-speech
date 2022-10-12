# People's Speech Data Description and Download

## Data Download

Please get the dataset from an official mirror: https://huggingface.co/datasets/MLCommons/peoples_speech


Then run the following commands:

```
mkdir -p ${YOUR_LOCAL_DESTINATION_DIR}
gsutil cp gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b/credits.csv ${YOUR_LOCAL_DESTINATION_DIR}/credits.csv
gsutil cp gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b/dataset_manifest_mp3_956_all.json ${YOUR_LOCAL_DESTINATION_DIR}/dataset_manifest_mp3_956_all.json
gsutil rsync -r gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b/training_set ${YOUR_LOCAL_DESTINATION_DIR}/training_set
```

You may also use `gsutil cp` instead of `gsutil rsync`. However, we
recommend gsutil `rsync` because it allows you to restart the download
without recopying data. The full dataset is 1.12TiB, so a broken
connection during download should be anticipated.

## Data Description

credits.csv lists the name of each CC-BY or CC-BY-SA work, the author
of the work, the URL to the exact license they used, and any
statements the author made about the rights granted for the work. This
file is here to comply with CC-BY and CC-BY-SA sources' requirements
that we attribute the creators of the works.

dataset_manifest_mp3_956_all.json is a [JSON
Lines](https://jsonlines.org/) formatted file that associates
transcripts with the paths to the corresponding audio snippets. We use json lines
over CSV format to avoid problems with the comma character appearing
in our dataset.

The schema of "dataset_manifest_mp3_956_all.json" is as follows:

```
root
 |-- audio_document_id: string (nullable = true)
 |-- identifier: string (nullable = true)
 |-- text_document_id: string (nullable = true)
 |-- training_data: struct (nullable = true)
 |    |-- duration_ms: array (nullable = true)
 |    |    |-- element: long (containsNull = true)
 |    |-- label: array (nullable = true)
 |    |    |-- element: string (containsNull = true)
 |    |-- output_paths: array (nullable = true)
 |    |    |-- element: string (containsNull = true)
```

identifier, audio_document_id, and text_document_id all can be used to
uniquely identify the original audio file and transcript from which
the training samples appearing in the "training_data" struct appear.

"training_data" is a "struct of arrays", where each array has the same
length. Effectively, the ith element of each of the three arrays
together represent the ith training sample created from a particular
(identifier, audio_document_id, text_document_id) triple.

"duration_ms[i]" is the duration in milliseconds of the audio
file. This could actually be derived from exmaining the audio file,
but this metadata is useful for to know beforehand in training
pipelines (e.g., to filter out segments that are too long, or to sort
data into buckets of similar length), so we provide it to users.

"label[i]" is the groundtruth transcript of the the audio file.

"output_paths[i]" is the path to the audio file underneath
"${YOUR_LOCAL_DESTINATION_DIR}/training_set". Effectively, the full
path to an audio file can be retrieved by conceptually doing
`os.path.join("${YOUR_LOCAL_DESTINATION_DIR}/training_set",
output_paths[i])`

## Example Data Prepration Script

We show an example script to convert the dataset into a format usable
by NVIDIA NeMo here:
[process_peoples_speech_data.py](/scripts/peoples_speech/process_peoples_speech_data.py). NeMo's
speech recognition input format is described
[here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/datasets.html#preparing-custom-asr-data).
