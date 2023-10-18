# CLIP 图片搜索工具

# 安装部署

建议使用 python 3.10.9 及以上，其它不保证正常工作。目前仅正常工作在 Windows 下。

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip3 install -r requirements.txt


# 启动

    python app.py

# 使用

    图片向量库构建 -> 建立图片元信息
    图片向量库构建 -> 构建向量库索引
    图片向量搜索 -> 加载库
    图片向量搜索 -> 输入参考图片和提示词(英文)进行搜索