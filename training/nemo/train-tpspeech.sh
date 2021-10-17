gpu_count=$1
manifest_path=$2
tarred_audio_paths=$3
val_manifest_path=$4
exp_name=$5
init_ckpt_path=${6-null}
init_ckpt_path=${init_ckpt_path//=/\\=}
seed=${7-41}

echo "GPU count: $gpu_count"
echo "Manifest filepath: $manifest_path"
echo "Tarred audio paths: $tarred_audio_paths"
echo "LibriSpeech dev clean manifest: $val_manifest_path"
echo "Experiment name: $exp_name"
echo "Load weights and schedule from : $init_ckpt_path"
echo "Random seed for everything : $seed"

echo "---------------"
echo "Starting training"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 1 $gpu_count)
cd "${0%/*}"
python train_ctc_model.py \
    --config-path=NeMo/examples/asr/conf/quartznet/ \
    --config-name=quartznet_15x5 \
    +trainer.resume_from_checkpoint=${init_ckpt_path} \
    exp_manager.exp_dir=trained-models \
    exp_manager.name=${exp_name} \
    model.train_ds.manifest_filepath=${manifest_path} \
    model.train_ds.tarred_audio_filepaths=${tarred_audio_paths} \
    model.train_ds.is_tarred=true \
    model.train_ds.batch_size=$((32 / $gpu_count)) \
    +model.train_ds.num_workers=8 \
    model.validation_ds.manifest_filepath=${val_manifest_path} \
    +model.validation_ds.num_workers=4 \
    model.validation_ds.batch_size=2 \
    +model.use_cer=true \
    trainer.max_epochs=5 \
    trainer.val_check_interval=0.02 \
    trainer.gpus=-1 \
    +seed=${seed}