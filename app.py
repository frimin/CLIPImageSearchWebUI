import os

#os.environ["SSLKEYLOGFILE"] = "./nss_ssl_sfagent.log"
#os.environ["HF_HOME"] = "./.cache/huggingface"

import argparse
from module.webui import start_app

def args_builder():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--share', action="store_true", help='共享此应用', default=False
    )

    parser.add_argument(
        '--server_name', type=str, help='服务器地址', default=None
    )

    parser.add_argument(
        '--server_port', type=str, help='服务器地址', default=None
    )

    parser.add_argument(
        '--root_path', type=str, default=None
    )

    parser.add_argument(
        '--userdata_dir', type=str, default="userdata"
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = args_builder()
    start_app(args)