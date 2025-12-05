# ComfyUI_OmniSVG
[OmniSVG](https://github.com/OmniSVG/OmniSVG): A Unified Scalable Vector Graphics Generation Model,you  can try it in ComfyUI.


# Update
* 支持官方4B和8B模型，支持单体Qwen2.5-VL模型，支持高版本的transformer（4.57.0），可选accelerate加载（主要是12G跑7B 使用accelerate会遇到meta数据不匹配，虽然做了适配，好像效果很一般，不如开启动态内存支持），12G跑4B模型测试下就好，7B太慢了，新模型感觉还是命中率不够，要抽卡，也不知道是不是我的环境的问题；
* Support official 4B and 8B models, support monolithic Qwen2.5-VL models, support higher versions of transformers, optional acceleration loading (mainly encountered when using acceleration on 12G running 7B, meta data mismatch may occur, although adaptation has been done, the effect seems to be mediocre, it is better to enable dynamic memory support)

---

# 1 . Installation /安装

In the ./ComfyUI /custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_OmniSVG.git
```


# 2 . Requirements  
* if windows ,cairo is difficult to install . use [links](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases/download/2022-01-04/gtk3-runtime-3.24.31-2022-01-04-ts-win64.exe)   install / 如果cairo安装不上，用链接的exe安装
cairo window install guide 安装指南 
```
pip install -r requirements.txt
```

# 3. Model
* [OmniSVG/OmniSVG1.1_8B](https://huggingface.co/OmniSVG/OmniSVG1.1_8B) download / 下载pytorch_model.bin后改名，注意要有8B字样
* [OmniSVG/OmniSVG1.1_4B](https://huggingface.co/OmniSVG/OmniSVG1.1_4B/tree/main) download / 下载pytorch_model.bin后改名，注意要有8B字样
* [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main) all files /所有文件 适配4B模型
* [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main) all files /所有文件 适配8B模型
* [qwen_2.5_vl_7b.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders) /不下repo也可以用单体模型  
  ```
  |   ├──you comfyui/models/diffusion_models/
  |       ├──OmniSVG1.1_8B.bin or OmniSVG1.1_4B.bin
  |   ├──any path/Qwen/Qwen2.5-VL-3B-Instruct or  /Qwen2.5-VL-7B-Instruct
  |       ├──all files # 所有文件
  |   ├──you comfyui/models/clip/
  |       ├──qwen_2.5_vl_7b.safetensors
  
  ```

# 4. Tips
* 文生矢量和图生矢量，连图则图，无图则文。

# 5 .Example
* ![](https://github.com/smthemex/ComfyUI_OmniSVG/blob/main/example_workflows/example.png)
* ![](https://github.com/smthemex/ComfyUI_OmniSVG/blob/main/example_workflows/example_.png)


# 6. Citation
```
@article{yang2025omnisvg,
  title={OmniSVG: A Unified Scalable Vector Graphics Generation Model}, 
  author={Yiying Yang and Wei Cheng and Sijin Chen and Xianfang Zeng and Jiaxu Zhang and Liao Wang and Gang Yu and Xinjun Ma and Yu-Gang Jiang},
  journal={arXiv preprint arxiv:2504.06263},
  year={2025}
}

```
