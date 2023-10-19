# 创建本地图片库的嵌入向量

想对本地图库进行搜索功能之前，需要对图集目录中的每个图片文件生成一个向量文件。

在对本地图库批处理之前请尽量确保图片文件的完整性，建议通过 `[扩展功能]` -> `[文件检查器]` 功能，先过滤并删除小于 1KB 的文件。

## 生成 embedding 向量文件

在 `[扩展功能]` -> `[构建向量库]` 中的 **生成 embedding 向量文件** 部分输入本地图片路径，可以一次性输入多个目录进行处理。

正确的例子:

    /path/to/your/image_dataset/game
    /path/to/your/image_dataset/person
    /path/to/your/image_dataset/animal

但是不要重复输入已经被包含在父级目录中的子目录，错误的例子：

    /path/to/your/image_dataset/game
    /path/to/your/image_dataset/game/subset0  <--此目录已经被父级包含
    /path/to/your/image_dataset/game/subset1 
    /path/to/your/image_dataset/person
    /path/to/your/image_dataset/animal

`批大小` 建议使用默认设置，对于 `工作进程` 数量建议使用 CPU 的物理核心数 ()。

生成速度除了取决于 CPU 和 GPU 资源以外，图片文件大小影响也很大。

## 生成向量库磁盘文件

此功能是可选的，如果图片数量不多 (小于一万张) 可以跳过此选项，图片较多的图库建议都生成向量库磁盘文件。

如果图库在慢速设备上也建议预先生成 (SMB 挂载的远程磁盘, 机械硬盘) 向量库磁盘文件。

在 `[扩展功能]` -> `[构建向量库]` 中的 **生成向量库磁盘文件** 部分输入需要生成的路径列表，规则与生成向量文件规则一致。