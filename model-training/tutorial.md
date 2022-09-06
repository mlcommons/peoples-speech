# Training Speech to Text Models on The People's Speech Dataset

[The People's Speech](https://mlcommons.org/en/peoples-speech/)[1] is among the world's largest English speech recognition datasets that is licensed for academic and commercial use. It includes 30,000+ hours of transcribed speech in English with a diverse set of speakers and acoustic environments. This open dataset by [MLCommons.org](https://mlcommons.org/en/) is large enough to train strong speech-to-text systems and is available with a permissive license.

In this tutorial, we’ll show you how to train your own models on this dataset. We’ll be using GPU-equipped VM instances on [Google Cloud Platform](https://cloud.google.com/) and training code from NVIDIA’s conversational toolkit, [NeMo](https://developer.nvidia.com/nvidia-nemo).

## Rough Overview
- Set up training infrastructure on GCP
- Download the dataset
- Download our packaged training software
- Train a 27M-parameter [Conformer](https://arxiv.org/abs/2005.08100)
- Use the trained model to transcribe audio.

## 1. Set up the Google Cloud SDK

You'll need a Google CLoud account and a project for this. If you already have one, you can skip steps 1 and 2.

We’ll be making extensive use of the Cloud SDK’s `gcloud`, which will allow us to create and manage most cloud resources from the terminal. I find that this minimizes errors since it will enable me to provide the exact commands that you need to execute (save for one bit).

1. [Create an account](https://console.cloud.google.com/freetrial/signup/tos) on Google Cloud if you don't already have one.
2. [Create a project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#console) in that account
3. Follow the instructions [here](https://cloud.google.com/sdk/docs/install) to install the GCP SDK
4. Follow the instructions [here](https://cloud.google.com/sdk/docs/initializing) to set it up. In In particular, you’ll be running
`gcloud init` , through which you’ll link your account and project to the local SDK in the terminal. You won’t be charged for any of this. In fact, you’ll likely claim 300 USD in credits for creating an account.

## 2. Create a GCP virtual machine

We’ll start by creating the [GCP Virtual Machine](https://cloud.google.com/compute/docs/instances) in which we’ll be doing all the training. Because we have to download the data first (which can take over a day), we’ll start by creating one without any GPUs attached, and less CPU and memory than needed for training. We’ll use this weaker machine to download The People’s Speech, and beef it up later when we’re ready to train.

To begin, we'll create an instance.The command we're running will create:

- A [preemptible](https://cloud.google.com/compute/docs/instances/preemptible) VM with 4vCPU and 15GB RAM. It comes with NVIDIA drives installed so you can use a GPU(thanks to the--image-family and --image-project parameters). Installing these drivers yourself is surprisingly difficult. 
- A 1000GB magnetic disk attached to the VM

In order to create the instance run the following command:

    gcloud compute instances create peoples-speech-training --project the-peoples-speech --zone us-central1-a --machine-type n1-standard-4  --no-restart-on-failure  --maintenance-policy TERMINATE --scopes https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --image-family pytorch-latest-gpu --image-project deeplearning-platform-release --boot-disk-size 200GB --boot-disk-type projects/the-peoples-speech/zones/europe-west4-b/diskTypes/pd-standard --boot-disk-device-name peoples-speech-training-disk  --metadata "install-nvidia-driver=True,proxy-mode=project_editors" --preemptible

Don't forget to replace both the project name and the [GCP zone](https://cloud.google.com/compute/docs/regions-zones) where you have GPUs available. This will create and start your VM.

## 3. Clone the repository with the training pipeline

We prepared a Docker image as well as some scripts to easily download the dataset. To use them you'll need to clone the repository for the tutorial into your VM. This can be done using:

    cd ~
    git clone https://github.com/mlcommons/peoples-speech.git

## 4. Download the data 

A few subsets of the People's Speech dataset are available at Huggingface, characterized by the type of license (CC-BY-4.0 or CC-BY-SA) and quality of the data (there are both a clean  and dirty subset, with higher quality labels in the former than in the latter). We'll be using the clean subset, CC-BY-4.0 licensed subset in this tutorial. If you want to use a different subset, visit the dataset's [homepage](https://mlcommons.org/en/peoples-speech/), browse for the right download links there, and replace them in the `wget` commands below.

To download the data, start by logging into your new VM:

    gcloud compute ssh peoples-speech-training

You'll need to download the manifest (which contains the text transcriptions) and the audio separately. First the manifest: 

    mkdir ~/data
    mkdir ~/data/the-peoples-speech/
    mkdir ~/data/librispeech
    mkdir ~/data/the-peoples-speech/cc-by-clean 
    cd ~/data/the-peoples-speech/cc-by-clean
    wget -O nemo_manifest.json https://huggingface.co/datasets/MLCommons/peoples-speech-v2/resolve/main/train/clean.json

Then the audio. We suggest you open a screen/tmux to perform this download, to protect it from connection failures between your local machine and the VM. We'll be using tmux, but if you're more familiar with screen you can use it.

    tmux
    mkdir ~/data/the-peoples-speech/cc-by-clean/audios
    python3 ~/peoples-speech/model-training/download_hf.py

Now that the download script is running, you can unattach from the tmux window using `ctrl+b` and then `d`on your keyboard. The download script will keep on running behind the scenes, and if your connection to the VM drops it'll keep on going.

Note that we are using preemptible instances. They're a lot cheaper, but they shut off after 24 hours. Given that we're using a subset for this tutorial, downloading the files should take no more than a couple minutes. Training with the whole dataset will imply downloading data for more than 30 hours!

## 4. Pull our Docker image to your VM

We prepared a Docker image with the software you'll need to train NeMo models. It's on Github. To download it run the following commands:

    cd peoples-speech/model-training
    sudo systemctl start docker
    sudo chmod 666 /var/run/docker.sock
    sudo docker build -t peoplespeech .

If it was pulled correctly, the output of `docker image list`
should include a line containing the following tag: 
peoplespeech

## 5. Verify GPU availability

We’re almost ready to start training speech-to-text models on a GPU! We’ll attach one to our VM in the next step, but first we need to verify that our GCP quota allows it. Google Cloud uses [quotas](https://cloud.google.com/compute/quotas#gpu_quota) to restrict users’ access to resources, so that they won’t accidentaly incur in unforeseen charges. You can check for your quotas on any region by issuing the following command. Note that quotas work
region -wise. We specified a zone when creating our VM; regions are larger than zones. My VM’s zone is `europe-west4-b`, and its region is `europe-west4`.

Running the following command should return a list of quotas for individual resources. Find the entry (remember Cmd+f exists) concerning `PREEMPTIBLE_NVIDIA_V100_GPUS` and verify that `limit` is > 0. Output on the command line should be as follows:

    - limit: 16.0
    metric: PREEMPTIBLE_NVIDIA_V100_GPUS
    usage: 0.0

You’re welcome to use another GPU if it better fits your needs, V100 is just what we used. However, We strongly suggest that you employ preemptible GPUs and VMs instead of non-preemptible. Preemptible resources last only 24 hours before they shut off automatically, but they’re WAY cheaper.

### What if the limit is zero?

Don’t worry, increasing quotas in GCP rarely takes over a day in my personal experience. We would give you instructions on this, but Google does it better, so go check out their [guide](https://cloud.google.com/docs/quota#requesting_higher_quota). Just remember you’re trying to increase the limit for
`PREEMPTIBLE_NVIDIA_V100_GPUS` in your VM’s region. Requesting quota increases is completely free, so request away!

## 6. Verify download

At this point you should check back on your download script. To do so, you have to attach back to the tmux window where we left the script running. Run `tmux attach -t 0` and, if the download is not done, wait (it should take at most 15 minutes). After downloading the files, run the `exit` command. 

## 7. Attach a GPU to your VM

Unfortunately, there’s currently no way to attach a GPU to an existing VM using
gcloud. You’ll have to use the Google Cloud Console (GCP’s browser UI). Follow
GCP’s [guide](https://cloud.google.com/compute/docs/gpus/add-remove-gpus#add-gpu-to-instance) and add between 1 and 4 NVIDIA V100 GPU to your VM. Don’t worry about preemptibility; the VM is preemptible, so you’ll be charged preemptible rates for the GPU.

Bear in mind that this will increase the price of keeping your instance on to 1.03 USD/hour for one GPU and up to 5 USD/hour if you attach 4 of them. Please don’t get the feeling that this is cheap, it adds up quickly!

## 8. Improve your VM's vCPU count and memory 

In order to train on a GPU, you need to increase the vCPU count and memory of our VM. These components are in charge of reading samples from disk, preprocessing them, and passing them to the GPU. Multiple vCPU can preprocess data in parallel (hence the count increase), and the VM needs enough memory to hold one batch of preprocessed data (hence the memory increase).

First, we need to stop the VM to make changes to its hardware:

    gcloud compute instances stop peoples-speech-training

We then change the machine type to one with 8 vCPU and 52GB RAM:
    
    gcloud compute instances set-machine-type peoples-speech-training --machine-type n1-highmem-8

## 9. Run the container and verify the GPU

Start your VM again, log into it, and run the image we pulled in Step 3.

    tmux 
    sudo chmod 666 /var/run/docker.sock
    docker run --gpus all -it --rm -v ~/data:/data -v ~/experiments:/experiments \
        --shm-size=52g --ulimit memlock=-1 --ulimit stack=67108864 \
        -p 6006:6006  peoplespeech:latest

Notice that we specified two [volumes](https://docs.docker.com/storage/volumes/) (via `-v`), a mechanism to exchange data between a container and its host. The first makes sure that the container has access to the data we downloaded, and the second makes sure that our experiment checkpoints and metrics persist in the host VM even if the container is killed.

Your terminal should now be connected to the running container. Verify that the GPU drivers are in place by running `nvidia-smi` . The output should list the GPU you attached in Step 6:

![smi](./smi.png)

If it doesn’t you either didn’t attach it correctly or its drivers aren’t in place. In the first case, do Step 6 more carefully. The second can be trickier. You could try stepping out of the container (Ctrl+a+d) and reinstalling the drivers (assuming you correctly set --image-family and --image-project) via: 

    sudo /opt/deeplearning/install-driver.sh

## 10. Obtain a validation set

We'll be using a subset of [Librispeech](https://ieeexplore.ieee.org/document/7178964) [2] as a validation set. NeMo includes scripts to download and format this dataset:

    python /workspace/NeMo/scripts/dataset_processing/get_librispeech_data.py --data_root=/data/librispeech --data_set=dev_clean

## 11. Train the Model

We’re ready to train! We’ll use the `train_ctc_model.py` for that. This script uses
NVIDIA NeMo to specify a 27M-parameter Conformermodel. NeMo uses [Hydra](https://hydra.cc/) to configure models. This allows for easier configuration versioning, which is often overlooked in Machine Learning code. The model configuration we’ll use is inside
`conformer_ctc_char.yaml`. You’ll notice some fields are yet to be configured because they have a value of ???:

- n_gpus: We'll set this to 1 because we attached a single GPU before. If you attached more than 1, please specify the correct number.
- model.train_ds.manifest_filepath : The path to the NeMo manifest generated in Step 4.
- model.validation_ds.manifest_filepath : The path to the librispeech manifest generated in Step 10.
- exp_manager.exp_dir: The directory where logs, metrics, and checkpoints will be stored. We’ll set this at the `/experiments` volume so these artifacts persist if the container or VM shut down.

    python train_ctc_model.py --config-name=conformer_ctc_char \
        model.train_ds.manifest_filepath="/data/the-peoples-speech/cc-by-clean/nemo_manifest.json" \
        model.validation_ds.manifest_filepath="/data/librispeech/dev_clean.json"\
        exp_manager.exp_dir=/experiments &>/data/clean_data_test.log 

NeMo will store logs, checkpoints, and metrics (Tensorboard) in the
`~/experiments/tutorial-training` directory.

## 12. Keep training or fine tuning
After 18 hours of training, our model reached 12.75% CER on Librispeech's dev-clean
subset. To keep training, rerun the container, find the artifacts from the previous training at `/experiments/tutorial-training` and issue a command like:

    python train_ctc_model.py --config-name=conformer_ctc_char name=tutorial-training n_gpus=1 model.train_ds.manifest_filepath=/data/the-peoples-speech/cc-by-clean/nemo_manifest.jsonl model.train_ds.tarred_audio_filepaths=/data/the-peoples-speech/cc-by-clean/audios.tar model.validation_ds.manifest_filepath=/data/librispeech/dev_clean.json exp_manager.exp_dir=/experiments trainer.resume_from_checkpoint=/experiments/tutorial-training/????-??-??_??-??-??/checkpoints/tutorial-training--val_wer\\=?-epoch\\=?-last.ckpt 


We trained a model using this method that reached 5.02% CER, 10.47% WER on the same subset, and 10.76% WER on `test-clean`, after about 48h of training on 4 parallel GPUs. We would expect marginal gains from training beyond this point.

Note that `trainer.resume_from_checkpoint` also allows you to finetune a pre-trained model, [like the ones published by NVIDIA](https://catalog.ngc.nvidia.com/models).

## 13. What's next? 

The model we trained is an [acoustic model](https://en.wikipedia.org/wiki/Acoustic_model)— one that maps auditive units to textual units. You could now:
- Use the model to transcribe audio. Hint: use our script `peoples-speech/model-training/transcribe.py`.
- Improve said transcriptions by involving a language model in the transcription process. An acoustic model is mostly unaware of which letter combinations make up actual words; a LM can provide this kind of information. Hint : Use our script `peoples-speech/model-training/transcribe.py` . Specify the `--kenlm_model_path` parameter ([see the docs for KenLM](https://github.com/kpu/kenlm)).
- Finetune on other data, or mix The People’s Speech in with other datasets.

## 14. References 

[1] Galvez, Daniel, Greg Diamos, Juan Ciro, Juan Felipe Cerón, Keith Achorn, Anjali Gopi, David Kanter, Maximilian Lam, Mark Mazumder, and Vijay Janapa Reddi. “The People’s Speech: A Large-Scale Diverse English Speech Recognition Dataset for Commercial Usage.” arXiv preprint arXiv:2111.09344 (2021).

[2] Panayotov, Vassil, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. “Librispeech: an asr corpus based on public domain audio books.” In
2015 IEEE international conference on acoustics, speech and signal processing (ICASSP) , pp. 5206–5210. IEEE, 2015.



