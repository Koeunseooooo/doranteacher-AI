from connection import s3_connection
from config import BUCKET_NAME, LOCATION
from botocore.client import Config
from botocore.client import Config
import boto3

# ap-northeast-2


def get_image_url(data):

    # s3 = s3_connection()
    image_url = f'https://{BUCKET_NAME}.s3.{LOCATION}.amazonaws.com/'+data

    print(image_url)
    return image_url


# get_image_url('test0609')
