from config import AWS_ACCESS_KEY, AWS_SECRET_KEY
from botocore.client import Config
import boto3


def s3_connection():
    s3 = boto3.resource('s3',
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY,
                        config=Config(signature_version='s3v4'))
    return s3
