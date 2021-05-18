gcloud container clusters get-credentials peoples-speech-platform
kubectl create deployment data-export --image=gcr.io/peoples-speech/data-export:latest
