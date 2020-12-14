# Setting up your machine for featurization / training / eval

Set up conda environment and build lingvo using bazel on VM / local machine
```sh
# install ubuntu dependencies first
sudo apt install wget git screen neovim build-essential htop parallel sox libsox-fmt-mp3

# Download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

# Create conda environment for lingvo, includes development tools (like yapf, pylint) as well as cuda
conda env create -n lingvo -f lingvo-copy/environment.yml
echo "conda activate lingvo" >> ~/.bashrc
bazel build -c opt //lingvo:trainer 
```
If you are running featurization you will need to create an extra volume when provisioning the VM. Then you will need to mount it. I have added the commands to `setup.sh` just in case, you can also find it in google cloud documentation.

# Submit a training job on TPU
Make sure you are inside a google cloud cpu VM, and using the `lingvo` conda environment.
```sh 
./submit.sh $device_name executor $class_model_name $fldr_name 

## Example usage: 
./submit.sh ag-tpu1-1019 executor Grphm_DO_SpecAug_ConvStk_6x512Bidi_40batchsize 1127
``` 

`device_name`: The `device_name` contains `-tpu`, script assumes you are training on a tpu. If a TPU by that name exists, the TPU is reused, if not, a new TPU is created with that name and the job is launched. `device_name` can also be `gpu` or `cpu`, and the job is launched locally in either of these cases.

`executor`: Lingvo has a bunch of `runner` classes for managing the training loop. We use the `executor_tpu` for training on the tpu. Use `executor` when training on tpu, or `trainer` when running locally on cpu/gpu. Training on local machines is not fully tested, you may have to change batch size, data dir etc when training locally. 
The tpu_executor runs a few iterations of training, then a few iterations of evaluation on the train and dev set, so you'll see 3 loss curves on tensorboard. 
- train-train: training loss on the training dataset with augmentation (dropout, specuagment)
- eval-train: eval loss on the training dataset, without augmentation (should be better than training loss)
- eval-dev: eval loss on the dev dataset. I use this curve for picking which model to choose for evaluation.

`class_model_name`: One of the fully specified model clases available in `lingvo/tasks/asr/params/peoplesspeech_ctc.py`. Use this class to specify all hyper-parameters. The class name in example above is our sota hyper-parameter spec for librispeech. You can also find the location of the training / dev / test data specified in this file.

`fldr_name`: Google cloud storage suffix where to save the checkpoints and tensorboard events. *IMPORTANT*: Change the GS_BASE path in `tpu_submit` to your directory (right now it's pointing to `gs://the-peoples-speech-west-europe/PeoplesSpeech/ag_training`). If there are checkpoints of `$class_model_name` in `$fldr_name` directory, you will be restarting that job, and the folder will be reused for new checkpoints.

# Eval a model on GPU/CPU:
We could not get the decoder / evaluation to run on TPU. Lingvo supports marking some functions to be only run on the tpu_host", but even this was not sufficient to make the decoding work, while the forward prop is running on the tpu. Therefore, I run the eval on my local gpu machine. You could also run on a cpu machine, but that is super slow. On a gtx1070 evaluation finishes in ~15 minutes. Make sure you set up the conda environment, just like above, and setup google cloud cli from https://cloud.google.com/sdk/docs/install

Download the latest checkpoint, and tensorboard events of all the models in the experiment group to `$HOME_BASE` which is hardcoded in `download_models.sh`.
```sh
./download_models.sh ${fldr_name}

## Example usage:
./download_models.sh 1127
```
Downloads from `gs://the-peoples-speech-west-europe/PeoplesSpeech/ag_training/1127` to `/home/anjali/data/PeoplesSpeech/models/1127`. Now you can evaluate the model on the dev set and see sample predictions, cer, wer using
```sh
./submit.sh gpu decode Grphm_DO_SpecAug_ConvStk_6x512Bidi_40batchsize 1127
```
Since I run locally on my home machine, I sometimes run into google cloud storage errors when evaling. So I have a copy of the devtest set locally - you may need this hack. You could also just get a small gpu vm for 1 hour and avoid all the model and dataset downloading.

After one pass, the process will keep monitoring the checkpoint directory for new checkpoints. The decoder job dumps a lot of the decoded output to tensorboard, so you should not let the decoder continue past an epoch.

# Featurizing a new dataset:
Make sure you have allocated a large disk on a large VM (>16 cores). See setup instructions above.
```sh
# Copy all the tarballs and csv files locally, takes a few hours
mkdir -p /mnt/disks/dataset/raw/v0.7/raw /mnt/disks/dataset/raw/v0.7/feats
gsutil -m rsync the-peoples-speech-west-europe/peoples-speech-v0.7 /mnt/disks/dataset/raw/v0.7/raw

# featurize the dataset, usually takes ~24 hours
# ./featurize.sh $input_dir $output_dir
./featurize.sh /mnt/disks/dataset/raw/v0.7/raw /mnt/disks/dataset/raw/v0.7/feats

# Upload the tfrecords back to the cloud
gsutil -m rsync /mnt/disks/dataset/raw/v0.7/feats gs://the-peoples-speech-west-europe/PeoplesSpeech/v0.7.1
```
`featurize.sh` calls `create_peoples_speech_asr_features.py` which first loads the csv file to read the audio filenames and transcripts (other metadata is thrown away) into a dictionary, keyed by the audio filepath. Then it makes a pass through the tarball, reading one audio file at a time, standardizing it (to 16Khz, single channel, 16 bit wav audio), calculates mel spectrograms (80 bins, 25ms window, 10ms step) and then saves it to tfrecord.

The featurization outputs 1 tfrecord file each for dev and test set. Training dataset is split into 512 tfrecords, and we use 16 processes in parallel to do the featurization. Worker 0 outputs files 0..31, worker 1 outputs 32..63 and so on. I picked 512 files so that each tfrecord file is around 1GB, same as librispeech. Therefore, there are 512 output_shards, and 16 worker shards. 
