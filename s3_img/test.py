from upload_image import send_image_to_s3
from get_image_url import get_image_url


def test():
    imgName = "recommend"
    send_image_to_s3(imgName)
    res = get_image_url(imgName)
    return res
