from connection import s3_connection
from config import BUCKET_NAME
from botocore.client import Config


def send_image_to_s3(imgName):
    s3 = s3_connection()
    data = open('img/'+imgName+'.jpg', 'rb')
    s3.Bucket(BUCKET_NAME).put_object(
        Body=data,
        Key=imgName,
        ContentType='image/jpg')
    print("finish")


# send_image_to_s3()
