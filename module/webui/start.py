import gradio as gr
import module.webui.search_vector as search_vector
from module.webui.extensions.video_screenshot import create_video_screenshot
from module.foundation.webui import (
    TopElements,
    TopTab, 
    TopTabContext,
)
from module.webui.components import (
    create_top,
    create_webui_configs,
    create_embedding_page,
    create_file_checker,
    create_image_augmentation,
)
from module.data import init_data

def start_app(args):
    init_data(args)
    context = TopTabContext()

    with gr.Blocks(theme=gr.themes.Default(), title="CLIPImageSearchWebUI") as demo:
        with gr.Row(): 
            top_elems: TopElements = create_top(context)
         
        with TopTab.Tabs(context) :
            with TopTab.TabItem(context, "search", "向量搜索"):
                search_vector.page(demo, args, top_elems)

            with TopTab.TabItem(context, "extensions", "扩展功能"):
                with TopTab.Tabs(context):
                    with TopTab.TabItem(context, "build_vectors", "构建向量库"):
                        create_embedding_page(top_elems)
                    with TopTab.TabItem(context, "video_screenshot", "视频截图"):
                        create_video_screenshot(top_elems)
                    with TopTab.TabItem(context, "file_checker", "文件检查器"):
                        create_file_checker(top_elems)
                    with TopTab.TabItem(context, "image_augmentation", "图像增广"):
                        create_image_augmentation(top_elems)

            with TopTab.TabItem(context, "config", "设置"):
                create_webui_configs(top_elems)

        top_elems.do_delay()
        #context.bind_all_requests()

    demo.launch(share=args.share, 
                inbrowser=True, 
                enable_queue=True,
                server_name=args.server_name,
                server_port=args.server_port,
                show_api=False,
                root_path=args.root_path)