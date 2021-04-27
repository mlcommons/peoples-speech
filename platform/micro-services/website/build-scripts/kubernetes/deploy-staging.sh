gcloud container clusters get-credentials peoples-speech-platform
kubectl create deployment website --image=gcr.io/peoples-speech/platform:latest
