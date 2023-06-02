import boto3
import os
import io

from scipy.io.wavfile import write

#TODO class화
def upload_file(wave, user_id, team_id, project_id, block_id):
    s3r = boto3.resource('s3', aws_access_key_id=os.getenv('S3_ACCESS_KEY_ID'), aws_secret_access_key=os.getenv('S3_SECRET_ACCESS_KEY'), region_name='ap-northeast-2')
    bucket = s3r.Bucket(os.getenv('S3_BUCKET_NAME'))

    bytes_wav = bytes()
    wav_object = io.BytesIO(bytes_wav)
    write(wav_object, 24000, wave)

    bucket.upload_fileobj(wav_object, str(team_id) + '/' + str(project_id) + '/' + str(block_id) + '.wav')