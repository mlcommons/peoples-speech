from gcloud import storage
client = storage.Client()
bucket = client.get_bucket('the-peoples-speech-west-europe')
print(bucket)
# Then do other things...
blob = bucket.get_blob('ag/ctc_librispeech/test')
print(blob.download_as_string())

blob2 = bucket.blob('ag/storage.txt')
blob2.upload_from_filename(filename='test')
import pdb
pdb.set_trace()
import tensorflow as tf
tf.io.gfile.makedirs("gs://the-peoples-speech-west-europe/ag/ctc_librispeech/test_tf_storage_access/")

"""
blob.upload_from_string('New contents!')
blob2 = bucket.blob('remote/path/storage.txt')
blob2.upload_from_filename(filename='/local/path.txt')
"""
