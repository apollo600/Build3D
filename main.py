import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QFileSystemModel, QGraphicsScene, QMessageBox, QGraphicsView
from PyQt5.QtCore import QTimer, QProcess, QUrl
from PyQt5.QtGui import QPixmap, QMovie
import subprocess
import time
import psutil
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "back_end"))
from front_end.main_ui import Ui_Form
from back_end.render_dataset import visulize_dataset


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # 变量定义
        self.root_path = None
        self.dataset_length = None
        self.ckpt_path = None
        self.embedding_idx = None
        self.render_output_path = None

        # 槽函数绑定
        self.ui.selectFileButton.clicked.connect(self.show_file_dialog)
        self.ui.startTrainButton.clicked.connect(self.start_train)
        self.ui.stopTrainButton.clicked.connect(self.stop_train)
        self.ui.chooseModelButton.clicked.connect(self.choose_model)
        self.ui.spinBox.valueChanged.connect(self.change_embedding)
        self.ui.startRenderButton.clicked.connect(self.start_render)

        # 定时器
        self.update_train_log_timer = QTimer(self)
        self.update_train_log_timer.timeout.connect(self.update_train_log)
        self.update_render_log_timer = QTimer(self)
        self.update_render_log_timer.timeout.connect(self.update_render_log)

        # 输出中转文件
        self.output_file_name = "output.txt"
        self.output_file = open(self.output_file_name, "w")

    def reportStatus(self, message:str):
        self.ui.statusLog.setText(message)

    def show_file_dialog(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getExistingDirectory(self, "选择数据所在文件夹")
        if file_path:
            print("选中的文件夹：", file_path)
            # 更新显示的文件夹名称
            self.ui.dataPathLabel.setText(file_path)
            # 创建QFileSystemModel对象
            model = QFileSystemModel()
            # 设置根目录路径
            self.root_path = file_path
            model.setRootPath(file_path)
            # 将QFileSystemModel设置为QTreeView的模型
            self.ui.treeView.setModel(model)
            # 设置QTreeView的根索引为根路径
            root_index = model.index(self.root_path)
            self.ui.treeView.setRootIndex(root_index)
            # 隐藏显示Last Modified列
            self.ui.treeView.setColumnWidth(0, 300)
            self.ui.treeView.setColumnHidden(3, True)
            # 渲染点云
            # self.render_visulize_dataset()

    def check_data_exists(self):
        if not self.root_path:
            # 创建 QMessageBox 对象
            msg_box = QMessageBox()

            # 设置提示信息内容和按钮样式
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先选择用于训练的数据集！")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setDefaultButton(QMessageBox.Ok)
            # 显示消息框
            msg_box.exec()
            return False
        else:
            return True
    
    def check_ckpt_exists(self):
        if not self.ckpt_path:
            # 创建 QMessageBox 对象
            msg_box = QMessageBox()

            # 设置提示信息内容和按钮样式
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先选择用于渲染的模型！")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setDefaultButton(QMessageBox.Ok)
            # 显示消息框
            msg_box.exec()
            return False
        else:
            return True
        
    def check_embedding_exists(self):
        if not self.embedding_idx:
            # 创建 QMessageBox 对象
            msg_box = QMessageBox()

            # 设置提示信息内容和按钮样式
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先选择用于渲染的背景！")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setDefaultButton(QMessageBox.Ok)
            # 显示消息框
            msg_box.exec()
            return False
        else:
            return True

    def start_train(self):
        self.reportStatus("开始训练")
        self.update_train_log_timer.start(7000)
        if self.check_data_exists():
            order = f"python train.py \
                    --root_dir {self.root_path} --dataset_name phototourism \
                    --img_downscale 2 --use_cache --N_importance 64 --N_samples 64 \
                    --encode_a --encode_t --beta_min 0.03 --N_vocab 1500 \
                    --num_epochs 40 --batch_size 3600 \
                    --optimizer adam --lr 5e-4 --lr_scheduler cosine \
                    --exp_name brandenburg_gate_2_v1 \
                    --num_gpus 1 "
            self.train_process = subprocess.Popen(f"cd back_end && {order}", stdout=self.output_file, shell=True)

    def setScene(self, graphicsView:QGraphicsView, image_path:str):
        # 进行缩放
        image = Image.open(image_path)
        # 原始图片的宽度和高度
        original_width, original_height = image.size
        # 目标框的宽度和高度
        target_width, target_height = graphicsView.width(), graphicsView.height()
        # 计算宽度和高度的缩放比例
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        # 选择较小的缩放比例作为最终的缩放比例
        scale_ratio = min(width_ratio, height_ratio)
        # 计算缩放后的宽度和高度
        scaled_width = int(original_width * scale_ratio)
        scaled_height = int(original_height * scale_ratio)
        # 缩放图片并保持宽高比
        scaled_image = image.resize((scaled_width, scaled_height), Image.ANTIALIAS)
        # 保存图片到本地
        scaled_image.save("loss.png")
        # 读取图片并显示
        pixmap = QPixmap("loss.png")
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        graphicsView.setScene(scene)

    def update_train_log(self):
        self.reportStatus(f"执行更新 {time.time()}")
        #*** 更新左侧训练日志部分 ***
        f = open(self.output_file_name, "r")
        output = f.read()
        f.close()
        # 更新内容
        self.ui.trainLog.setPlainText(output)
        # 滚动到最下方
        scroll_bar = self.ui.trainLog.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        #*** 更新右侧损失函数曲线 ***
        try:
            self.setScene(self.ui.graphicsView, "loss.png")
        except Exception as e:
            print("尚无图片", e)

    def stop_train(self):
        self.reportStatus(f"停止训练")
        self.update_train_log_timer.stop()
        process = psutil.Process(self.train_process.pid)
        for child in process.children(recursive=True):
            child.terminate()
        process.terminate()
        if os.path.exists(self.output_file_name):
            os.remove(self.output_file_name)
        if os.path.exists("loss.png"):
            os.remove("loss.png")

    def render_visulize_dataset(self):
        self.reportStatus(f"正在渲染数据集点云")
        try:
            visulize_dataset(self.root_path)
            self.reportStatus(f"数据集点云渲染完成")
            # 加载 HTML 文件到 QWebEngineView
            self.ui.webEngineView.load(QUrl.fromLocalFile("plot.html"))
        except Exception as e:
            self.reportStatus(f"渲染出错")
            print("渲染出错", e)
    
    def choose_model(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择模型文件", filter="Checkpoint files (*.ckpt)")
        if file_path:
            self.reportStatus("模型选择完毕")
            self.ui.modelName.setPlainText(file_path)
            self.ckpt_path = file_path

    def change_embedding(self, value):
        if not self.dataset_length:
            if self.check_data_exists():
                self.dataset_length = len(os.listdir(os.path.join(
                    self.root_path, "dense", "images")))
                self.ui.spinBox.setRange(0, self.dataset_length)
                self.ui.chooseModelTip.setText(
                    self.ui.chooseModelTip.text() + f"[0-{self.dataset_length}]"
                )
        else:
            self.reportStatus(f"更新Embedding为{value}")
            data_dir = os.listdir(os.path.join(
                            self.root_path, "dense", "images"))
            chosen_image_path = os.path.join(
                            self.root_path, "dense", "images", data_dir[value])
            self.setScene(self.ui.graphicsView_2, chosen_image_path)
            self.reportStatus(f"图片渲染完成")
            self.embedding_idx = value

    def start_render(self):
        if self.check_ckpt_exists() and self.check_data_exists() and self.check_embedding_exists():
            self.render_output_path = os.path.join(
                self.root_path, "..", "..", "results", "phototourism", self.root_path.split('/')[-1]
            )
            self.reportStatus("开始渲染")
            order = f"python render_model.py \
                    --ckpt_path {self.ckpt_path}\
                    --data_path {self.root_path}\
                    --output_path {self.render_output_path}\
                    --appearence_idx {self.embedding_idx}"
            print(order)
            self.render_process = subprocess.Popen(f"cd back_end && {order}", shell=True)
            self.update_render_log_timer.start(5000)
        else:
            pass
    
    def update_render_log(self):
        if self.render_output_path:
            #*** 输出文字提示 ***
            total = 120
            num = len(os.listdir(self.render_output_path))
            if num > 0:
                num -= 1
            self.ui.renderLog.setPlainText(f"已经完成渲染{num}/{total}")
            self.reportStatus(f"执行更新 {time.time()}")

            #*** 渲染动画 ***
            movie = QMovie(os.path.join(self.render_output_path, "render_result.gif"))
            self.ui.gifLabel.setMovie(movie)
            movie.start()
        else:
            print("没有输出路径")
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
