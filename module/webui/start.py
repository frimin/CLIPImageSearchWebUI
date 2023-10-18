import gradio as gr
import module.webui.search_vector as search_vector
from module.utils import path_util
from module.webui.extensions.video_screenshot import create_video_screenshot
from module.webui import guides
from module.foundation.webui import (
    TopElements
)
from module.webui.components import (
    create_top,
    create_webui_configs,
    create_embedding_page,
    create_file_checker,
)
from module.data import init_data

def start_app(args):
    init_data(args)
    path_util.clear_cache_dir()

    with gr.Blocks(theme=gr.themes.Default(), title="CLIPImageSearchWebUI") as demo:
        with gr.Row(): 
            top_elems: TopElements = create_top()

        with gr.Tabs() as tabs:
            with gr.TabItem("向量搜索") as t:
                search_vector.page(demo, args, top_elems)

            with gr.TabItem("扩展功能"):
                with gr.Tab("构建向量库"):
                    create_embedding_page(top_elems)
                with gr.Tab("视频截图"):
                    create_video_screenshot(top_elems)
                with gr.Tab("文件检查器"):
                    create_file_checker(top_elems)

            #with gr.Tab("使用指南"):
            #    guides.on_gui(args)

            with gr.TabItem("设置"):
                create_webui_configs(top_elems)

    demo.launch(share=args.share, 
                inbrowser=True, 
                enable_queue=True,
                server_name=args.server_name,
                server_port=args.server_port,
                show_api=False,
                root_path=args.root_path)