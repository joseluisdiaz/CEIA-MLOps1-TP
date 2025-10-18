import json
import pandas as pd
# import awswrangler as wr

import boto3
import botocore
import botocore.exceptions

import utils.constants as consts


def get_metadata_info(client: boto3.client) -> dict:
    """Obtiene la metadata del dataset desde S3"""
    try:
        client.head_object(Bucket=consts.BUCKET, Key=consts.DATA_INFO_PATH)
        result = client.get_object(
            Bucket=consts.BUCKET,
            Key=consts.DATA_INFO_PATH)
        text = result["Body"].read().decode()
        data_dict = json.loads(text)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchKey" and e.response["Error"]["Code"] != "404":
            raise e  # Something else has gone wrong.
        data_dict = {}
    return data_dict


def save_metadata_info(client: boto3.client, data_dict: dict):
    """Guarda la metadata del dataset en S3"""
    data_string = json.dumps(data_dict, indent=2)
    client.put_object(
        Bucket=consts.BUCKET,
        Key=consts.DATA_INFO_PATH,
        Body=data_string
    )
