import gradio as gr

NO_QUERY_RESULT_ERROR = gr.Error("没有查询结果, 请先执行查询")
INVALID_QUERT_RECORD_ERROR = gr.Error("无效的查询记录")
GR_ERR_NO_VECTOR_DATABASE_SELECTED = gr.Error("未选择任何向量库")
GR_ERR_NO_MULTIPLE_VECTOR_DATABASE_SUPPORTED = gr.Error("不支持处理多个向量库")
DISABLE_DELETE_FILE_ERROR = gr.Error("文件删除功能已被全局禁用, [设置]->[安全] 中可关闭")

IMAGE_EXTENSIONS = ('.jpg', '.png', '.jpeg','.webp')