import os
from obs import ObsClient
import yaml
from uuid import uuid4
import time


class ObsUtil:
    def __init__(self, logger=None):
        config = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)
        ak = config['obs']['ak']
        sk = config['obs']['sk']
        server = config['obs']['server']
        self.bucket = config['obs']['bucket']
        self.base_dir = config['obs']['base_dir']

        if ak is None or sk is None or server is None:
            raise ValueError('OBS config error')

        date_str = time.strftime("%Y-%m-%d", time.localtime())
        self.base_dir = os.path.join(self.base_dir, date_str)

        self.client = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)
        self.logger = logger

    # 上传文件
    # file_path: 本地文件路径
    # return: OBS文件路径
    def upload(self, file_path):
        logger = self.logger
        if not os.path.exists(file_path):
            msg = f'file not exists: {file_path}'
            logger.error(msg)
            raise ValueError(msg)
        file_name = str(uuid4()) + "." + file_path.split('.')[-1]
        target_path = os.path.join(self.base_dir, file_name)
        self.client.putFile(self.bucket, target_path, file_path)
        return target_path

    # 下载文件
    # file_path: OBS文件路径
    # return: 本地文件路径
    def download(self, file_path):
        target_path = os.path.join("./tmp", str(uuid4()) + "." + file_path.split('.')[-1])
        self.client.getObject(self.bucket, file_path, downloadPath=target_path)
        return target_path

    def close(self):
        self.client.close()

