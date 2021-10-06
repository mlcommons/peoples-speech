gpu_count=$1
manifest_path=$2
tarred_audio_paths=$3
val_manifest_path=$4
echo "GPU count: $gpu_count"
echo "Manifest filepath: $manifest_path"
echo "Tarred audio paths: $tarred_audio_path"
echo "LibriSpeech dev clean manifest: $val_manifest_path"

echo "---------------"
echo "Starting training"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 1 $gpu_count)
echo $CUDA_VISIBLE_DEVICES
cd "${0%/*}"
cd NeMo
python -m examples.asr.speech_to_text \
    --config-path=conf/quartznet/ \
    --config-name=quartznet_15x5 \
    model.train_ds.manifest_filepath=$2 \
    model.train_ds.tarred_audio_filepaths=$3 \
    model.train_ds.is_tarred=true \
    model.train_ds.batch_size=$((32 / $gpu_count)) \
    +model.train_ds.num_workers=8 \
    model.validation_ds.manifest_filepath=$4 \
    +model.validation_ds.num_workers=4 \
    model.validation_ds.batch_size=2 \
    +model.use_cer=true \
    trainer.max_epochs=5 \
    trainer.val_check_interval=0.02 \
    trainer.gpus=-1