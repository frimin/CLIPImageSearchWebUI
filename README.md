# CLIPImageSearchWebUI

[CLIP](https://github.com/mlfoundations/open_clip) 模型通过训练巨量的图片和文本对，学习将两者的嵌入向量 (vector embeddings) 的相似程度。

本工具提供了对本地的图片库生成嵌入向量，并通过图片或文本来进行对本地图库进行搜索的功能。

![](./doc/img/search_page.jpg)

# 安装部署

## Windows

Windows 下提供了 PowerShell 脚本编写的快速安装脚本。

保存 [此脚本文件](https://raw.githubusercontent.com/frimin/CLIPImageSearchWebUI/main/scripts/create_embeded_env.ps1) 到空目录中执行即可进行快速环境部署。

## 手动安装

建议使用 python 3.10.9，其它版本效果未知。目前仅在 Windows 下测试开发，WSL2 和 Linux 效果未知。

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip3 install -r requirements.txt
    pip3 install -e .

# 启动

    python app.py

# 使用说明

  * [创建本地图片库的嵌入向量](./doc/zh-cn/create_vectors.md)