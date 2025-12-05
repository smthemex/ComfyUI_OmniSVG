# Cairo中文安装指南（包含vcpkg安装指南）--方便没有代码基础的新手

 * 1、安装vcpkg (因为vcpkg包很大，网络不好的，也可以直接下载zip，解压后，去掉目录名-master)

```
git clone https://github.com/microsoft/vcpkg.git

```
* 2、 cd vcpkg  （CMD或powershell）就是进入vcpkg目录

* 3、运行bootstrap-vcpkg.bat （会下载对应win版本的vcpkg.exe，GitHub网络不好可能要多等等）

* 4、vcpkg安装好之后，在vcpkg目录下打开CMD，运行

```
vcpkg install cairo

```
安装tips：vcpkg会自动下配套的包文件到vcpkg的download 目录下，但是国内网络不好的情况下（未翻墙），可能会下载很久，主要可能会遇到诸如PowerShell-7.2.24-win-x64.zip包更新，cairo-1.18.4.tar.gz包这两个文件较大。如果你能提前下载，可以放到download 目录下。注意cairo-1.18.4.tar.gz在windows环境下是看不到后缀.gz的，如果是预下载，你需要将cairo-1.18.4.tar重命名为cairo-cairo-1.18.4.tar。
任何下载不了的文件 都会有-->连接和指向名称，主动下载后 可以改名然后放置在download里。
（所有因为网络不好，下载不了的包都可以按此方法）

* 5、如果能正常安装cairo，为确保环境运行，你可能还需要安装pycairo;

```
pip install pycairo
```
* 6、 打开vcpkg的安装路径，将‘你的路径\vcpkg\installed\x64-windows\bin’添加到系统环境变量里；

* 7、如果使用的是python_embeded（comfyUI便携或者秋叶包），需要执行以下命令：
```
.\vcpkg export cairo --output=portable-python-libs --zip
```
输出的zip包里，找到x64-windows目录，将其内容按以下结构复制到你的python_embeded（便携python包）
```
📁 your-portable-python/
├── 📁 DLLs/             # 复制所有 .dll 到这里
├── 📁 include/          # 复制 include 文件夹
├── 📁 libs/             # 复制 lib/*.lib 到这里
├── 📁 Scripts/
├── 📄 python.exe
└── 📄 ...其他文件
```
* 8、安装 cairo报错处理：
  * 8.1  编译可能报错处理方式：
修改`你的路径/vcpkg/ports/cairo/portfile.cmake`，在`vcpkg_configure_meson`的`OPTIONS`中添加：
 ```cmake
   vcpkg_configure_meson(
     ...
     OPTIONS
        ... # 保留其他选项
        -Doptimization=1   # 使用O1优化，避免O2下出现的问题
   )
 ```
  * 8.2 如果8.1的还是解决不了，在后面按下所示，加上Doptimization=0，也就是不优化，这个主要是svg编译的问题
```
vcpkg_configure_meson(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${OPTIONS}
        -Dtests=disabled
        -Dzlib=enabled
        -Dpng=enabled
        -Dspectre=auto
        -Dgtk2-utils=disabled
        -Dsymbol-lookup=disabled
        -Doptimization=1
)
vcpkg_install_meson(
     OPTIONS
        ${OPTIONS}
        -Doptimization=0
)
```
  * 8.3 如果还是报错，尝试使用Clang-cl编译:
```
.\vcpkg install llvm --triplet x64-windows
# 设置使用 Clang
$env:CC = "clang"
$env:CXX = "clang++"
.\vcpkg install cairo --triplet x64-windows
  * 8.4 还是报错，使用debug模式安装，查看log解决问题
```
 vcpkg install cairo --triplet x64-windows --debug
```

* 9 安装完成后测试：
创建 test_cairo.py
```
import cairo

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 200, 100)
ctx = cairo.Context(surface)

ctx.set_source_rgb(0.8, 0.2, 0.2)
ctx.rectangle(10, 10, 180, 80)
ctx.fill()

surface.write_to_png("output.png")
print("Cairo 测试成功！")

```
打开CMD 运行 python test_cairo.py

