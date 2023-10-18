from boto3.session import Session
import os
import zipfile
from dotenv import load_dotenv
import argparse
import os
import pathlib

HOME = str(pathlib.Path(__file__).parent.parent)
load_dotenv()

session = Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
)

import logging

logging.basicConfig(level=logging.INFO)


def download_from_s3(
    bucket_name,
    bucket_path,
    local_dir,
):
    # create local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    # download the zip file
    logging.info("Downloading the zip file from S3...")
    s3_file = bucket_path.split("/")[-1]
    s3 = session.resource("s3")
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(bucket_path, os.path.join(local_dir, s3_file))

    # unzip the file
    logging.info("Unzipping the zip file ...")
    with zipfile.ZipFile(os.path.join(local_dir, s3_file), "r") as zip_ref:
        zip_ref.extractall(local_dir)
    # delete the zip file
    os.remove(os.path.join(local_dir, s3_file))


# upload .zip files to S3
def upload_zip_to_s3(local_dir, bucket_name):
    s3 = session.resource("s3")
    bucket = s3.Bucket(bucket_name)
    # subir solo los archivos .zip
    for file in os.listdir(local_dir):
        if file.endswith(".zip"):
            # subirlo al bucket
            logging.info(f"Uploading {file} to S3...")
            bucket.upload_file(
                os.path.join(local_dir, file), os.path.join("videos", file)
            )
    logging.info("Uploading completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline obtener los archivos de video de AWS"
    )

    parser.add_argument(
        "--bucket_name",
        default=AWS_BUCKET,
        type=str,
        help="Nombre del bucket",
    )

    parser.add_argument(
        "--bucket_path",
        default=AWS_FILE_PATH,
        type=str,
        help="Path al archivo .zip dentro del bucket",
    )

    parser.add_argument(
        "--videos_dir",
        default=os.path.join(HOME, "videos"),
        type=str,
        help="Ruta al archivo de video",
    )

    parser.add_argument(
        "--dataset_dir",
        default=os.path.join(HOME, "dataset"),
        type=str,
        help="Ruta al archivo de video",
    )

    parser.add_argument(
        "--download",
        default=True,
        action="store_false",
        help="Si se desea descargar de S3",
    )

    args = parser.parse_args()

    if args.download:
        download_from_s3(
            bucket_name=args.bucket_name,
            bucket_path=args.bucket_path,
            local_dir=args.videos_dir,
        )
    else:
        upload_zip_to_s3(args.dataset_dir, args.bucket_name)


if __name__ == "__main__":
    main()
