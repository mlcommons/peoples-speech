# Running lingvo models on Librispeech dataset

* Set up conda environment and build lingvo using bazel on VM:

  * conda env create -f environment.yml
  * bazel build -c opt //lingvo:trainer 


* Submit a training job on TPU:

  * ./tpu_submit.sh tpu_name executor class_model_name fldr_name. It the TPU exists, the TPU is reused, if not, new TPU is created and the job is launched. The TPU name should contain -tpu .
  * Eg: ./tpu_submit.sh ag-tpu1-1019 executor Grphm_DO_SpecAug_ConvStk_6x512Bidi 1020
  * The class_model_names for different models are available in librispeech_ctc.py.

  * IMPORTANT: Change the GS base path to your directory (right now it's pointing to the root GS dir). This is in line 9 of the tpu_submit.sh script (GS_BASE).

* Submit an eval job on GPU/CPU:

  * Decoder does not run on TPU. Use CPU or GPU.
  * IMPORTANT: Specify your GS Cloud path in -logdir. Specify the name of the model that yoi want to run in -model.

  * CPU:
  
  bazel-bin/lingvo/trainer --logdir=gs://the-peoples-speech-west-europe/YOUR_GS_PATH --mode=sync --model=asr.librispeech_ctc.YOUR_MODEL_NAME --logtostderr --run_locally=cpu --job=decoder_dev
  
  Eg:
  bazel-bin/lingvo/trainer --logdir=gs://the-peoples-speech-west-europe/ag/ctc_librispeech/training_logs/1020/Grphm_DO_SpecAug_ConvStk_6x512Bidi --mode=sync --model=asr.librispeech_ctc.Grphm_DO_SpecAug_ConvStk_6x512Bidi --logtostderr --run_locally=cpu --job=decoder_dev

  * GPU:

  bazel-bin/lingvo/trainer --logdir=gs://the-peoples-speech-west-europe/YOUR_GS_PATH --mode=sync --model=asr.librispeech_ctc.YOUR_MODEL_NAME --logtostderr --run_locally=gpu --job=decoder_dev
  
  Eg:
  bazel-bin/lingvo/trainer --logdir=gs://the-peoples-speech-west-europe/ag/ctc_librispeech/training_logs/1020/Grphm_DO_SpecAug_ConvStk_6x512Bidi --mode=sync --model=asr.librispeech_ctc.Librispeech_Grphm_DO_SpecAug_InptStack_6x1024 --logtostderr --run_locally=gpu --job=decoder_dev