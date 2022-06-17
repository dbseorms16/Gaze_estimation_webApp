
import hashlib
import os

import requests
from tqdm import tqdm

def md5(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def request_if_not_exist(file_name, url, md5sum=None, chunksize=1024):
    if not os.path.isfile(file_name):
        print("Download model: {}".format(os.path.basename(file_name)))
        request = requests.get(url, timeout=10, stream=True)
        with open(file_name, 'wb') as fh:
            # Walk through the request response in chunks of 1MiB
            for chunk in tqdm(request.iter_content(chunksize), desc=os.path.basename(file_name),
                              total=int(int(request.headers['Content-length']) / chunksize),
                              unit="KiB"):
                fh.write(chunk)
        if md5sum is not None:
            print("Checking md5 for {}".format(os.path.basename(file_name)))
            assert md5sum == md5(
                file_name), "MD5Sums do not match for {}. Please) delete the same file name to re-download".format(
                file_name)
                
def download_gaze_pytorch_models():
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     './gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model'),
        "https://imperialcollegelondon.box.com/shared/static/6rvctw7wmkpl7a9bw9hm1b9b7dwntfut.model",
        "ae435739673411940eed18c98c29bfb1")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     './gaze_model_pytorch_vgg16_prl_mpii_allsubjects2.model'),
        "https://imperialcollegelondon.box.com/shared/static/xuhs5qg7eju4kw3e4to7db945qk2c123.model",
        "4afd7ccf5619552ed4a9f14606b7f4dd")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     './gaze_model_pytorch_vgg16_prl_mpii_allsubjects3.model'),
        "https://imperialcollegelondon.box.com/shared/static/h75tro719fcyvgdkzr8tarpco32ve21u.model",
        "743902e643322c40bd78ca36aacc5b4d")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     './gaze_model_pytorch_vgg16_prl_mpii_allsubjects4.model'),
        "https://imperialcollegelondon.box.com/shared/static/1xywt1so20vw09iij4t3tp9lu6f6yb0g.model",
        "06a10f43088651053a65f9b0cd5ac4aa")
    
download_gaze_pytorch_models()