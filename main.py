#!/usr/bin/env python

import sys
import os
import numpy as np
from vtk.util import numpy_support
import csv
import SimpleITK as sitk
import nibabel as nib
from skimage import measure
import pyvista as pv

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QWidget, QLabel, QMessageBox, QFileDialog,
                           QProgressDialog, QGroupBox, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from vmtk import vmtkscripts
import vtk

# 设置日志系统
import logging
from datetime import datetime

def setup_logging():
    """设置日志系统，将信息写入results目录下的log.txt文件"""
    # 确保results目录存在
    log_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "log.txt")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # 同时输出到控制台
        ]
    )
    
    # 重写print函数，将标准输出也记录到日志
    class LoggerWriter:
        def __init__(self, level):
            self.level = level
            
        def write(self, message):
            if message.strip():
                self.level(message.strip())
                
        def flush(self):
            pass
    
    # 重定向标准输出和标准错误
    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)
    
    logging.info("=" * 50)
    logging.info("主动脉中心线分析系统启动")
    logging.info(f"日志文件: {log_file}")
    logging.info("=" * 50)

# 在程序开始时设置日志
setup_logging()

class ConversionThread(QThread):
    """用于后台执行转换任务的线程"""
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, dcm_folder, output_dir):
        super().__init__()
        self.dcm_folder = dcm_folder
        self.output_dir = output_dir
        
    def run(self):
        try:
            self.progress_signal.emit(0, "开始DICOM到NIfTI转换...")
            logging.info("开始DICOM到NIfTI转换...")
            
            # 步骤1: DICOM 转 NIfTI
            nii_path = self.dcm_to_nii()
            if not nii_path:
                error_msg = "DICOM转NIfTI失败"
                logging.error(error_msg)
                self.error_signal.emit(error_msg)
                return
                
            self.progress_signal.emit(33, "DICOM转NIfTI完成，开始深度学习分割...")
            logging.info("DICOM转NIfTI完成，开始深度学习分割...")
            
            # 步骤2: 深度学习分割
            mask_path = self.run_segmentation(nii_path)
            if not mask_path:
                error_msg = "深度学习分割失败"
                logging.error(error_msg)
                self.error_signal.emit(error_msg)
                return
                
            self.progress_signal.emit(66, "分割完成，开始生成3D模型...")
            logging.info("分割完成，开始生成3D模型...")
            
            # 步骤3: NIfTI 转 STL
            stl_path = self.nii_to_stl(mask_path)
            if not stl_path:
                error_msg = "NIfTI转STL失败"
                logging.error(error_msg)
                self.error_signal.emit(error_msg)
                return
                
            self.progress_signal.emit(100, "所有转换步骤完成！")
            logging.info("所有转换步骤完成！")
            self.finished_signal.emit(stl_path)
            
        except Exception as e:
            error_msg = f"转换过程中出错: {str(e)}"
            logging.error(error_msg, exc_info=True)
            self.error_signal.emit(error_msg)
    
    def dcm_to_nii(self):
        """将DICOM文件夹转换为NIfTI格式"""
        try:
            output_file = os.path.join(self.output_dir, "output_volume.nii.gz")
            logging.info(f"开始DICOM转NIfTI，输出文件: {output_file}")
            
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(self.dcm_folder)
            
            if not dicom_names:
                error_msg = "在指定文件夹中未找到DICOM文件"
                logging.error(error_msg)
                self.error_signal.emit(error_msg)
                return None
                
            logging.info(f"找到 {len(dicom_names)} 个DICOM文件")
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            
            self.progress_signal.emit(10, "正在读取DICOM数据...")
            logging.info("正在读取DICOM数据...")
            image = reader.Execute()
            
            self.progress_signal.emit(20, "正在保存为NIfTI格式...")
            logging.info("正在保存为NIfTI格式...")
            sitk.WriteImage(image, output_file, useCompression=True)
            
            logging.info(f"DICOM转NIfTI成功: {output_file}")
            return output_file
            
        except Exception as e:
            error_msg = f"DICOM转NIfTI错误: {str(e)}"
            logging.error(error_msg, exc_info=True)
            self.error_signal.emit(error_msg)
            return None
    
    def run_segmentation(self, nii_path):
        """运行深度学习分割"""
        try:
            # 设置路径
            checkpoint_path = "./model/Iteration_3500"
            output_mask_path = os.path.join(self.output_dir, "pred_mask.nii.gz")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
            
            self.progress_signal.emit(40, "正在加载分割模型...")
            logging.info("正在加载分割模型...")
            
            # 导入必要的模块
            try:
                import sys
                # 添加当前项目根目录到系统路径
                project_path = os.path.dirname(os.path.abspath(__file__))
                if project_path not in sys.path:
                    sys.path.insert(0, project_path)

                from inference.inference_sega import (
                    default_single_step_inference_params,
                    single_step_inference,
                    run_inference
                )
                logging.info("成功导入分割模块")
            except ImportError as e:
                error_msg = f"导入分割模块失败: {str(e)}"
                logging.error(error_msg, exc_info=True)
                self.error_signal.emit(error_msg)
                return None
            
            self.progress_signal.emit(45, "正在配置推理参数...")
            logging.info("正在配置推理参数...")
            
            # 构建推理参数
            params = default_single_step_inference_params(checkpoint_path)
            
            # 设置推理参数
            params['normalization_window'] = (-100, 400)  # 主动脉CT的窗宽
            params['echo'] = False                        # 在后台运行时关闭详细输出
            params['postprocess'] = True                  # 去除小连通域
            params['threshold'] = 0.5                     # 分割阈值
            
            self.progress_signal.emit(50, "正在进行主动脉分割...")
            logging.info("正在进行主动脉分割...")
            
            # 执行推理
            logging.info(f"开始推理: {nii_path}")
            try:
                output, output_mesh = run_inference(
                    input_path=nii_path,
                    ground_truth_path=None,
                    output_path=output_mask_path,
                    inference_method=single_step_inference,
                    inference_method_params=params
                )
                logging.info("推理执行成功")
            except Exception as inference_error:
                error_msg = f"推理执行错误: {str(inference_error)}"
                logging.error(error_msg, exc_info=True)
                self.error_signal.emit(error_msg)
                return None
            
            self.progress_signal.emit(60, "分割完成，正在保存结果...")
            logging.info("分割完成，正在保存结果...")
            
            # 检查输出文件是否存在
            if not os.path.exists(output_mask_path):
                error_msg = "分割结果文件未生成"
                logging.error(error_msg)
                self.error_signal.emit(error_msg)
                return None
            
            # 打印统计信息
            try:
                import nibabel as nib
                mask_img = nib.load(output_mask_path)
                mask_data = mask_img.get_fdata()
                voxel_count = mask_data.sum()
                logging.info(f"分割体积: {voxel_count} 个体素")
            except Exception as e:
                logging.warning(f"无法读取分割结果统计信息: {str(e)}")
            
            logging.info(f"推理完成！结果已保存至: {output_mask_path}")
            return output_mask_path
            
        except Exception as e:
            error_msg = f"分割过程错误: {str(e)}"
            logging.error(error_msg, exc_info=True)
            self.error_signal.emit(error_msg)
            return None
    
    def nii_to_stl(self, mask_path):
        """将分割结果转换为STL格式"""
        try:
            output_stl_path = os.path.join(self.output_dir, "pred_mask_3d_model.stl")
            logging.info(f"开始NIfTI转STL，输出文件: {output_stl_path}")
            
            self.progress_signal.emit(70, "正在读取分割结果...")
            logging.info("正在读取分割结果...")
            img = nib.load(mask_path)
            data = img.get_fdata()
            
            self.progress_signal.emit(80, "正在提取3D表面...")
            logging.info("正在提取3D表面...")
            # 选择标签值（根据您的分割结果调整）
            label_value = 1
            binary_data = (data == label_value)
            
            if not np.any(binary_data):
                error_msg = "分割结果中没有找到主动脉数据"
                logging.error(error_msg)
                self.error_signal.emit(error_msg)
                return None
            
            # 使用marching cubes提取3D表面
            verts, faces, normals, values = measure.marching_cubes(binary_data, level=0.5)
            
            # 转换为PyVista格式
            faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(int)
            
            self.progress_signal.emit(90, "正在生成STL文件...")
            logging.info("正在生成STL文件...")
            mesh = pv.PolyData(verts, faces_pv)
            mesh.save(output_stl_path)
            
            logging.info(f"STL转换成功: {output_stl_path}")
            return output_stl_path
            
        except Exception as e:
            error_msg = f"STL转换错误: {str(e)}"
            logging.error(error_msg, exc_info=True)
            self.error_signal.emit(error_msg)
            return None

class CenterlinePicker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.surface = None
        self.start_point = None
        self.end_point = None
        self.points_selected = 0
        self.actor_start = None
        self.actor_end = None
        self.current_picker_position = None
        self.temp_actor = None
        self.centerline_actor = None
        self.surface_actor = None
        self.centerline_data = None
        self.centerline_points = None
        self.centerline_radii = None
        self.centerline_diameters = None
        self.diameter_label_actor = None
        self.diameter_line_actor = None
        self.diameter_highlight_actor = None
        self.display_mode = "both"
        self.current_stl_path = None
        self.conversion_thread = None
        
        # 设置输出目录为当前目录下的results
        self.output_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logging.info(f"输出目录设置为: {self.output_dir}")
        
        self.initUI()

    def closeEvent(self, event):
        """重写关闭事件，在窗口关闭时弹出确认对话框"""
        reply = QMessageBox.question(
            self, 
            '确认退出',
            '您确定要退出主动脉中心线分析系统吗？\n\n未保存的数据可能会丢失。',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 安全清理资源
            self.cleanupBeforeExit()
            event.accept()
        else:
            event.ignore()

    def confirmQuit(self):
        """退出按钮的确认退出方法"""
        reply = QMessageBox.question(
            self,
            '确认退出',
            '您确定要退出主动脉中心线分析系统吗？\n\n未保存的数据可能会丢失。',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.cleanupBeforeExit()
            self.close()

    def cleanupBeforeExit(self):
        """退出前的资源清理"""
        logging.info("正在清理资源并退出...")
        
        # 停止转换线程
        if hasattr(self, 'conversion_thread') and self.conversion_thread and self.conversion_thread.isRunning():
            logging.info("停止转换线程...")
            self.conversion_thread.terminate()
            self.conversion_thread.wait(2000)  # 等待2秒
        
        # 清理VTK资源
        if hasattr(self, 'vtk_widget'):
            logging.info("清理VTK资源...")
            self.vtk_widget.Finalize()
        
        logging.info("资源清理完成")

    def initUI(self):
        self.setWindowTitle('主动脉中心线分析系统 - Aortic Centerline Analysis System')
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置专业风格的样式表
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
                font-family: "Segoe UI", Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2c3e50;
            }
            QPushButton {
                background-color: #3498db;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 12px;
                margin: 2px;
                border-radius: 4px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
                padding: 9px 17px 7px 15px;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
            QLabel {
                font-size: 11px;
                color: #2c3e50;
                padding: 2px;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                background-color: #ecf0f1;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                width: 20px;
            }
        """)
        
        # 创建中央widget和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # === 左侧控制面板 ===
        left_panel = QFrame()
        # 移除固定宽度，使用大小策略实现自适应
        left_panel.setMaximumWidth(400)  # 设置最大宽度
        left_panel.setMinimumWidth(250)  # 设置最小宽度
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #bdc3c7;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(8, 8, 8, 8)
        
        # 标题
        title_label = QLabel("主动脉中心线分析系统")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                text-align: center;
            }
        """)
        title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        left_layout.addWidget(title_label)
        
        # === 数据加载组 ===
        data_group = QGroupBox("数据加载")
        data_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        data_layout = QVBoxLayout(data_group)
        
        # 输出目录信息
        output_info = QLabel(f"输出目录: {self.output_dir}")
        output_info.setStyleSheet("font-size: 10px; color: #7f8c8d; background-color: #ecf0f1; padding: 5px; border-radius: 3px;")
        output_info.setWordWrap(True)
        output_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        data_layout.addWidget(output_info)
        
        self.load_dcm_button = QPushButton("📁 加载DICOM数据")
        self.load_dcm_button.clicked.connect(self.loadAndProcessDICOM)
        self.load_dcm_button.setToolTip("选择包含DICOM文件的文件夹进行自动处理")
        self.load_dcm_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        data_layout.addWidget(self.load_dcm_button)
        
        self.load_stl_button = QPushButton("🔍 加载STL模型")
        self.load_stl_button.clicked.connect(self.loadSTLFile)
        self.load_stl_button.setToolTip("直接加载已有的STL模型文件")
        self.load_stl_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        data_layout.addWidget(self.load_stl_button)
        
        left_layout.addWidget(data_group)
        
        # === 状态信息组 ===
        status_group = QGroupBox("系统状态")
        status_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("🔵 等待加载数据")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                padding: 8px;
                border-radius: 4px;
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
            }
        """)
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        status_layout.addWidget(self.status_label)
        
        self.diameter_info_label = QLabel("📏 直径信息: 计算中心线后查看")
        self.diameter_info_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                padding: 8px;
                border-radius: 4px;
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
            }
        """)
        self.diameter_info_label.setWordWrap(True)
        self.diameter_info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        status_layout.addWidget(self.diameter_info_label)
        
        self.display_label = QLabel("👁️ 显示模式: 未加载数据")
        self.display_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                padding: 8px;
                border-radius: 4px;
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                color: #0c5460;
            }
        """)
        self.display_label.setWordWrap(True)
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        status_layout.addWidget(self.display_label)
        
        left_layout.addWidget(status_group)
        
        # === 中心线分析组 ===
        analysis_group = QGroupBox("中心线分析")
        analysis_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        analysis_layout = QVBoxLayout(analysis_group)
        
        # 操作说明
        instruction_label = QLabel(
            "操作说明:\n"
            "1. 加载数据后，在3D视图中选择点\n"
            "2. 按空格键选择起点和终点\n"
            "3. 点击计算中心线进行分析\n"
            "4. 鼠标悬停查看直径信息"
        )
        instruction_label.setStyleSheet("font-size: 10px; color: #7f8c8d; background-color: #f8f9fa; padding: 8px; border-radius: 4px;")
        instruction_label.setWordWrap(True)
        instruction_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        analysis_layout.addWidget(instruction_label)
        
        # 分析按钮
        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("🔄 重新选择")
        self.reset_button.clicked.connect(self.resetSelection)
        self.reset_button.setEnabled(False)
        self.reset_button.setStyleSheet("background-color: #95a5a6; color: white;")
        self.reset_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button_layout.addWidget(self.reset_button)
        
        self.compute_button = QPushButton("📐 计算中心线")
        self.compute_button.clicked.connect(self.computeCenterlines)
        self.compute_button.setEnabled(False)
        self.compute_button.setStyleSheet("background-color: #f39c12; color: white;")
        self.compute_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button_layout.addWidget(self.compute_button)
        
        analysis_layout.addLayout(button_layout)
        
        left_layout.addWidget(analysis_group)
        
        # === 显示控制组 ===
        display_group = QGroupBox("显示控制")
        display_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        display_layout = QVBoxLayout(display_group)
        
        display_buttons_layout = QHBoxLayout()
        self.show_both_button = QPushButton("👁️ 全部显示")
        self.show_both_button.clicked.connect(lambda: self.setDisplayMode("both"))
        self.show_both_button.setEnabled(False)
        self.show_both_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white;
                text-align: left;
                padding-left: 10px;
            }
            QPushButton:disabled {
                background-color: #4CAF50;
                color: white;
            }
        """)
        self.show_both_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        display_buttons_layout.addWidget(self.show_both_button)
        
        self.show_centerline_button = QPushButton("📈 仅中心线")
        self.show_centerline_button.clicked.connect(lambda: self.setDisplayMode("centerline_only"))
        self.show_centerline_button.setEnabled(False)
        self.show_centerline_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; 
                color: white;
                text-align: left;
                padding-left: 10px;
            }
            QPushButton:disabled {
                background-color: #2196F3;
                color: white;
            }
        """)        
        self.show_centerline_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        display_buttons_layout.addWidget(self.show_centerline_button)
        
        self.show_surface_button = QPushButton("🔍 仅主动脉")
        self.show_surface_button.clicked.connect(lambda: self.setDisplayMode("surface_only"))
        self.show_surface_button.setEnabled(False)
        self.show_surface_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800; 
                color: white;
                text-align: left;
                padding-left: 10px;
            }
            QPushButton:disabled {
                background-color: #FF9800;
                color: white;
            }
        """)
        self.show_surface_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        display_buttons_layout.addWidget(self.show_surface_button)
        
        display_layout.addLayout(display_buttons_layout)
        
        # 快捷键提示
        shortcut_label = QLabel(
            "快捷键:\n"
            "空格键 - 选择点\n"
            "R - 重新选择\n"
            "C - 计算中心线\n"
            "1/2/3 - 切换显示模式"
        )
        shortcut_label.setStyleSheet("font-size: 9px; color: #666; background-color: #f8f9fa; padding: 6px; border-radius: 3px;")
        shortcut_label.setWordWrap(True)
        shortcut_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        display_layout.addWidget(shortcut_label)
        
        left_layout.addWidget(display_group)
        
        # 退出按钮
        self.quit_button = QPushButton("🚪 退出系统")
        self.quit_button.clicked.connect(self.close)
        self.quit_button.setStyleSheet("background-color: #e74c3c;")
        self.quit_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        left_layout.addWidget(self.quit_button)
        
        # 添加伸缩空间
        left_layout.addStretch()
        
        # === 右侧3D视图区域 ===
        right_panel = QFrame()
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border: 1px solid #34495e;
                border-radius: 8px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # 3D视图标题
        view_title = QLabel("3D 可视化视图")
        view_title.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: white;
                padding: 8px;
                background-color: #34495e;
                border-radius: 5px;
                text-align: center;
            }
        """)
        view_title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        right_layout.addWidget(view_title)
        
        # VTK渲染窗口 - 使用Expanding大小策略
        self.vtk_widget = QVTKRenderWindowInteractor()
        self.vtk_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vtk_widget.setMinimumSize(400, 400)  # 设置最小尺寸
        right_layout.addWidget(self.vtk_widget)
        
        # 将左右面板添加到主布局
        # 设置左侧面板的伸缩因子为0，右侧为1，这样右侧会占据更多空间
        main_layout.addWidget(left_panel, 0)  # 0表示左侧面板不会过度拉伸
        main_layout.addWidget(right_panel, 1)  # 1表示右侧面板会占据更多空间
        
        # 初始化VTK渲染器
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        # 设置交互器样式
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.001)
        
        # 添加事件监听
        self.interactor.AddObserver("MouseMoveEvent", self.onMouseMove)
        self.interactor.AddObserver("KeyPressEvent", self.onKeyPress)
        
        # 设置初始背景
        self.renderer.SetBackground(0.1, 0.1, 0.2)  # 深蓝色背景
        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground2(0.3, 0.3, 0.4)
        
        logging.info("用户界面初始化完成")

    def resizeEvent(self, event):
        """处理窗口大小变化事件"""
        super().resizeEvent(event)
        # 窗口大小变化时强制重新渲染VTK窗口
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.GetRenderWindow().Render()

    def loadAndProcessDICOM(self):
        """选择DICOM文件夹并进行完整处理"""
        dcm_folder = QFileDialog.getExistingDirectory(self, "选择DICOM文件夹")
        if not dcm_folder:
            return
            
        logging.info(f"选择的DICOM文件夹: {dcm_folder}")
        
        # 使用固定的results目录，不再询问输出目录
        output_dir = self.output_dir
        
        # 创建进度对话框
        self.progress_dialog = QProgressDialog("正在处理DICOM数据...", "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle("处理进度")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setStyleSheet("""
            QProgressDialog {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
            }
        """)
        self.progress_dialog.show()
        
        # 启动转换线程
        self.conversion_thread = ConversionThread(dcm_folder, output_dir)
        self.conversion_thread.progress_signal.connect(self.updateProgress)
        self.conversion_thread.finished_signal.connect(self.onConversionFinished)
        self.conversion_thread.error_signal.connect(self.onConversionError)
        self.conversion_thread.start()
        
    def updateProgress(self, value, message):
        """更新进度条"""
        self.progress_dialog.setValue(value)
        self.progress_dialog.setLabelText(message)
        
    def onConversionFinished(self, stl_path):
        """转换完成回调"""
        self.progress_dialog.close()
        self.current_stl_path = stl_path
        self.loadSurface(stl_path)
        self.status_label.setText("✅ 所有处理步骤已完成！\nSTL文件已保存")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                padding: 8px;
                border-radius: 4px;
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
        """)
        logging.info(f"所有处理步骤已完成，STL文件: {stl_path}")
        QMessageBox.information(self, "处理完成", f"所有处理步骤已完成！\n\n输出文件保存在:\n{self.output_dir}")
        
    def onConversionError(self, error_message):
        """转换错误回调"""
        self.progress_dialog.close()
        self.status_label.setText("❌ 处理失败\n" + error_message)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                padding: 8px;
                border-radius: 4px;
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
        """)
        logging.error(f"转换过程错误: {error_message}")
        QMessageBox.critical(self, "处理错误", error_message)
        
    def loadSTLFile(self):
        """直接加载STL文件"""
        stl_path, _ = QFileDialog.getOpenFileName(self, "选择STL文件", "", "STL Files (*.stl)")
        if stl_path:
            logging.info(f"直接加载STL文件: {stl_path}")
            self.current_stl_path = stl_path
            self.loadSurface(stl_path)
            
    def loadSurface(self, stl_path=None):
        """加载STL表面并进行平滑预处理"""
        if stl_path is None:
            if self.current_stl_path:
                stl_path = self.current_stl_path
            else:
                QMessageBox.warning(self, "警告", "没有可用的STL文件路径")
                return
                
        logging.info(f"Step 1: Reading STL surface from {stl_path}...")
        
        try:
            # 使用VTK的STL读取器
            reader = vtk.vtkSTLReader()
            reader.SetFileName(stl_path)
            reader.Update()
            
            original_surface = reader.GetOutput()
            
            if original_surface.GetNumberOfPoints() == 0:
                raise Exception("STL文件没有数据点")
            
            # === 新增：表面平滑预处理 ===
            logging.info("正在进行表面平滑预处理...")
            
            # 方法1: 使用vtkSmoothPolyDataFilter进行拉普拉斯平滑
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputData(original_surface)
            smoother.SetNumberOfIterations(50)  # 增加迭代次数
            smoother.SetRelaxationFactor(0.1)   # 较小的松弛因子避免过度平滑
            smoother.FeatureEdgeSmoothingOn()   # 开启特征边平滑
            smoother.SetFeatureAngle(60.0)      # 设置特征角度
            smoother.SetEdgeAngle(15)           # 设置边角度
            smoother.BoundarySmoothingOn()      # 开启边界平滑
            smoother.Update()
            smoothed_surface = smoother.GetOutput()
            
            """
            # 方法2: 可选 - 使用WindowedSincPolyDataFilter进行更高质量的平滑
            # 如果方法1效果不好，可以取消注释下面的代码
            sinc_smoother = vtk.vtkWindowedSincPolyDataFilter()
            sinc_smoother.SetInputData(original_surface)
            sinc_smoother.SetNumberOfIterations(30)
            sinc_smoother.SetPassBand(0.01)  # 较低的pass band保持形状
            sinc_smoother.NormalizeCoordinatesOn()
            sinc_smoother.Update()
            smoothed_surface = sinc_smoother.GetOutput()
            """

            # 验证平滑后的表面质量
            if smoothed_surface.GetNumberOfPoints() == 0:
                logging.warning("平滑后表面为空，使用原始表面")
                self.surface = original_surface
            else:
                self.surface = smoothed_surface
                logging.info(f"表面平滑完成: {original_surface.GetNumberOfPoints()} -> {smoothed_surface.GetNumberOfPoints()} 个点")
            
            """
            # === 可选：添加表面清理 ===
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(self.surface)
            cleaner.SetTolerance(0.01)  # 合并相近的点
            cleaner.Update()
            self.surface = cleaner.GetOutput()
            """

            # 创建表面mapper和actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.surface)
            mapper.ScalarVisibilityOff()
            
            self.surface_actor = vtk.vtkActor()
            self.surface_actor.SetMapper(mapper)
            self.surface_actor.GetProperty().SetColor(0.7, 0.7, 0.7)  # 灰色
            self.surface_actor.GetProperty().SetOpacity(0.6)
            
            self.renderer.AddActor(self.surface_actor)
            self.renderer.ResetCamera()
            
            self.vtk_widget.GetRenderWindow().Render()
            self.interactor.Initialize()
            
            # 更新界面状态
            self.status_label.setText("🔵 表面平滑处理完成\n等待选择起点...")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 11px;
                    padding: 8px;
                    border-radius: 4px;
                    background-color: #d1ecf1;
                    border: 1px solid #bee5eb;
                    color: #0c5460;
                }
            """)
            self.display_label.setText("👁️ 显示模式: 平滑后的主动脉表面")
            
            logging.info("STL文件加载和平滑处理成功")
            
        except Exception as e:
            error_msg = f"加载和处理STL文件失败: {str(e)}"
            logging.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "错误", error_msg)

    def onMouseMove(self, obj, event):
        """处理鼠标移动事件，实时显示当前位置"""
        mouse_pos = self.interactor.GetEventPosition()
        
        self.picker.Pick(mouse_pos[0], mouse_pos[1], 0, self.renderer)
        picked_position = self.picker.GetPickPosition()
        
        if self.picker.GetCellId() != -1:  # 确保在表面上
            self.current_picker_position = picked_position
            
            if self.centerline_data is not None:
                self.checkCenterlineHover(picked_position)
            
            if self.points_selected < 2:
                self.updateTempMarker()
        else:
            if self.temp_actor and self.points_selected < 2:
                self.renderer.RemoveActor(self.temp_actor)
                self.temp_actor = None
                self.vtk_widget.GetRenderWindow().Render()
            
            self.removeDiameterDisplay()

    def checkCenterlineHover(self, position):
        """检查鼠标是否在中心线上，如果是则显示直径"""
        if self.centerline_points is None or self.centerline_diameters is None:
            return
        
        min_distance = float('inf')
        closest_index = -1
        closest_point = None
        
        for i, point in enumerate(self.centerline_points):
            distance = np.linalg.norm(np.array(point) - np.array(position))
            if distance < min_distance:
                min_distance = distance
                closest_index = i
                closest_point = point
        
        if min_distance < 5.0:
            diameter = self.centerline_diameters[closest_index]
            radius = self.centerline_radii[closest_index]
            self.showDiameterAtPosition(closest_point, diameter, radius, closest_index)
        else:
            self.removeDiameterDisplay()

    def showDiameterAtPosition(self, position, diameter, radius, point_index):
        """在指定位置显示直径信息"""
        self.removeDiameterDisplay()
        
        text = f"直径: {diameter:.2f} mm\n半径: {radius:.2f} mm\n点索引: {point_index}"
        
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(text)
        text_actor.GetTextProperty().SetFontSize(18)
        text_actor.GetTextProperty().SetColor(1, 1, 0)
        text_actor.GetTextProperty().SetBold(True)
        text_actor.GetTextProperty().SetBackgroundColor(0, 0, 0)
        text_actor.GetTextProperty().SetBackgroundOpacity(0.7)
        text_actor.SetPosition(20, 50)
        
        self.diameter_label_actor = text_actor
        self.renderer.AddActor2D(text_actor)
        
        line_source = vtk.vtkLineSource()
        line_start = [0.1, 0.1, 0]
        line_end = [0.3, 0.3, 0]
        line_source.SetPoint1(line_start)
        line_source.SetPoint2(line_end)
        
        line_mapper = vtk.vtkPolyDataMapper2D()
        line_mapper.SetInputConnection(line_source.GetOutputPort())
        
        line_actor = vtk.vtkActor2D()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(1, 1, 0)
        line_actor.GetProperty().SetLineWidth(2)
        
        self.diameter_line_actor = line_actor
        self.renderer.AddActor2D(line_actor)
        
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(position)
        sphere.SetRadius(1.0)
        sphere.SetPhiResolution(12)
        sphere.SetThetaResolution(12)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        
        highlight_actor = vtk.vtkActor()
        highlight_actor.SetMapper(mapper)
        highlight_actor.GetProperty().SetColor(1, 0, 0)
        highlight_actor.GetProperty().SetOpacity(1.0)
        
        self.diameter_highlight_actor = highlight_actor
        self.renderer.AddActor(highlight_actor)
        
        self.diameter_info_label.setText(f"当前位置直径: {diameter:.2f} mm, 半径: {radius:.2f} mm, 点索引: {point_index}")
        self.diameter_info_label.setStyleSheet("font-size: 12px; color: yellow; background-color: black; padding: 5px;")
        
        self.vtk_widget.GetRenderWindow().Render()

    def removeDiameterDisplay(self):
        """移除直径显示"""
        if self.diameter_label_actor:
            self.renderer.RemoveActor2D(self.diameter_label_actor)
            self.diameter_label_actor = None
        
        if self.diameter_line_actor:
            self.renderer.RemoveActor2D(self.diameter_line_actor)
            self.diameter_line_actor = None
        
        if hasattr(self, 'diameter_highlight_actor') and self.diameter_highlight_actor:
            self.renderer.RemoveActor(self.diameter_highlight_actor)
            self.diameter_highlight_actor = None
        
        if self.centerline_data is not None:
            self.diameter_info_label.setText("直径信息: 将鼠标放在中心线上查看实时直径")
            self.diameter_info_label.setStyleSheet("font-size: 12px; color: red; padding: 5px;")
        
        self.vtk_widget.GetRenderWindow().Render()

    def onKeyPress(self, obj, event):
        """处理键盘按键事件"""
        key = self.interactor.GetKeySym()
        
        if key == "space":
            self.onSpacePressed()
        elif key == "r":
            self.resetSelection()
        elif key == "c":
            self.computeCenterlines()
        elif key == "1":
            self.setDisplayMode("both")
        elif key == "2":
            self.setDisplayMode("centerline_only")
        elif key == "3":
            self.setDisplayMode("surface_only")

    def onSpacePressed(self):
        """处理空格键按下事件"""
        try:
            logging.debug(f"空格键按下 - 当前picker位置: {self.current_picker_position}")
            logging.debug(f"当前选择点数: {self.points_selected}")
            
            if self.current_picker_position is None:
                error_msg = "请在血管表面上选择点\n\n可能的原因：\n1. 鼠标没有在血管表面上\n2. 表面数据可能有问题\n3. 请先移动鼠标到血管表面"
                logging.warning(error_msg)
                QMessageBox.warning(self, "警告", error_msg)
                return
                
            if self.points_selected == 0:
                # 选择起点
                self.start_point = self.current_picker_position
                self.addPointMarker(self.start_point, (0, 0, 1), "start", 3.0)
                self.points_selected = 1
                self.status_label.setText("🔵 起点已选择（蓝色球体）\n移动鼠标选择终点（黄色立方体），按空格键确认")
                self.status_label.setStyleSheet("""
                    QLabel {
                        font-size: 11px;
                        padding: 8px;
                        border-radius: 4px;
                        background-color: #d1ecf1;
                        border: 1px solid #bee5eb;
                        color: #0c5460;
                    }
                """)
                logging.info(f"起点选择: {self.start_point}")
                
                if self.temp_actor:
                    self.renderer.RemoveActor(self.temp_actor)
                    self.temp_actor = None
                
            elif self.points_selected == 1:
                # 选择终点
                self.end_point = self.current_picker_position
                self.addPointMarker(self.end_point, (1, 1, 0), "end", 3.0)
                self.points_selected = 2
                self.status_label.setText("🟡 起点（蓝色球体）和终点（黄色立方体）已选择\n按C键计算中心线或R键重新选择")
                self.status_label.setStyleSheet("""
                    QLabel {
                        font-size: 11px;
                        padding: 8px;
                        border-radius: 4px;
                        background-color: #fff3cd;
                        border: 1px solid #ffeaa7;
                        color: #856404;
                    }
                """)
                self.compute_button.setEnabled(True)
                self.reset_button.setEnabled(True)
                logging.info(f"终点选择: {self.end_point}")
                
                if self.temp_actor:
                    self.renderer.RemoveActor(self.temp_actor)
                    self.temp_actor = None
            
            self.vtk_widget.GetRenderWindow().Render()
            logging.debug("空格键处理完成")
            
        except Exception as e:
            error_msg = f"空格键处理出错: {e}"
            logging.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "错误", f"处理空格键时发生错误: {str(e)}")

    def updateTempMarker(self):
        """更新临时标记点，显示当前位置"""
        if self.points_selected >= 2:
            return
            
        if self.temp_actor:
            self.renderer.RemoveActor(self.temp_actor)
            self.temp_actor = None
        
        if self.current_picker_position is not None:
            if self.points_selected == 0:
                color = (0, 0, 1)
                marker_size = 2.5
                marker_type = "sphere"
            else:
                color = (1, 1, 0)
                marker_size = 2.5
                marker_type = "cube"
            
            if marker_type == "sphere":
                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(self.current_picker_position)
                sphere.SetRadius(marker_size)
                sphere.SetPhiResolution(16)
                sphere.SetThetaResolution(16)
                
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(sphere.GetOutputPort())
                
                self.temp_actor = vtk.vtkActor()
                self.temp_actor.SetMapper(mapper)
                self.temp_actor.GetProperty().SetColor(color)
                self.temp_actor.GetProperty().SetOpacity(0.6)
            else:
                cube = vtk.vtkCubeSource()
                cube.SetCenter(self.current_picker_position)
                cube.SetXLength(marker_size * 2)
                cube.SetYLength(marker_size * 2)
                cube.SetZLength(marker_size * 2)
                
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(cube.GetOutputPort())
                
                self.temp_actor = vtk.vtkActor()
                self.temp_actor.SetMapper(mapper)
                self.temp_actor.GetProperty().SetColor(color)
                self.temp_actor.GetProperty().SetOpacity(0.6)
            
            self.renderer.AddActor(self.temp_actor)
            self.vtk_widget.GetRenderWindow().Render()

    def addPointMarker(self, position, color, marker_type, size=3.0):
        """添加正式点标记"""
        if marker_type == "start":
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(position)
            sphere.SetRadius(size)
            sphere.SetPhiResolution(20)
            sphere.SetThetaResolution(20)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetOpacity(1.0)
            
            self.addCrossLines(position, size * 1.5, (1, 1, 1))
            
            if self.actor_start:
                self.renderer.RemoveActor(self.actor_start)
            self.actor_start = actor
            logging.info("添加蓝色起点标记（球体+十字线）")
            
        else:
            cube = vtk.vtkCubeSource()
            cube.SetCenter(position)
            cube.SetXLength(size * 2)
            cube.SetYLength(size * 2)
            cube.SetZLength(size * 2)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(cube.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetOpacity(1.0)
            actor.GetProperty().SetEdgeColor(1, 0, 0)
            actor.GetProperty().SetEdgeVisibility(1)
            actor.GetProperty().SetLineWidth(2)
            
            if self.actor_end:
                self.renderer.RemoveActor(self.actor_end)
            self.actor_end = actor
            logging.info("添加黄色终点标记（立方体+红色边框）")
            
        self.renderer.AddActor(actor)

    def addCrossLines(self, position, length, color):
        """添加十字线标记"""
        line_x = vtk.vtkLineSource()
        line_x.SetPoint1(position[0] - length, position[1], position[2])
        line_x.SetPoint2(position[0] + length, position[1], position[2])
        
        mapper_x = vtk.vtkPolyDataMapper()
        mapper_x.SetInputConnection(line_x.GetOutputPort())
        
        actor_x = vtk.vtkActor()
        actor_x.SetMapper(mapper_x)
        actor_x.GetProperty().SetColor(color)
        actor_x.GetProperty().SetLineWidth(3)
        
        line_y = vtk.vtkLineSource()
        line_y.SetPoint1(position[0], position[1] - length, position[2])
        line_y.SetPoint2(position[0], position[1] + length, position[2])
        
        mapper_y = vtk.vtkPolyDataMapper()
        mapper_y.SetInputConnection(line_y.GetOutputPort())
        
        actor_y = vtk.vtkActor()
        actor_y.SetMapper(mapper_y)
        actor_y.GetProperty().SetColor(color)
        actor_y.GetProperty().SetLineWidth(3)
        
        line_z = vtk.vtkLineSource()
        line_z.SetPoint1(position[0], position[1], position[2] - length)
        line_z.SetPoint2(position[0], position[1], position[2] + length)
        
        mapper_z = vtk.vtkPolyDataMapper()
        mapper_z.SetInputConnection(line_z.GetOutputPort())
        
        actor_z = vtk.vtkActor()
        actor_z.SetMapper(mapper_z)
        actor_z.GetProperty().SetColor(color)
        actor_z.GetProperty().SetLineWidth(3)
        
        self.renderer.AddActor(actor_x)
        self.renderer.AddActor(actor_y)
        self.renderer.AddActor(actor_z)

    def setDisplayMode(self, mode):
        """设置显示模式"""
        if mode == self.display_mode:
            return
            
        self.display_mode = mode
        
        if mode == "both":
            if self.surface_actor:
                self.surface_actor.SetVisibility(1)
            if self.centerline_actor:
                self.centerline_actor.SetVisibility(1)
            self.display_label.setText("👁️ 显示模式: 主动脉 + 中心线")
            self.display_label.setStyleSheet("""
                QLabel {
                    font-size: 11px;
                    padding: 8px;
                    border-radius: 4px;
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    color: #155724;
                }
            """)
            self.show_both_button.setStyleSheet("background-color: #4CAF50; color: white;")
            self.show_centerline_button.setStyleSheet("background-color: #2196F3; color: white;")
            self.show_surface_button.setStyleSheet("background-color: #FF9800; color: white;")
            
        elif mode == "centerline_only":
            if self.surface_actor:
                self.surface_actor.SetVisibility(0)
            if self.centerline_actor:
                self.centerline_actor.SetVisibility(1)
            self.display_label.setText("📈 显示模式: 仅中心线")
            self.display_label.setStyleSheet("""
                QLabel {
                    font-size: 11px;
                    padding: 8px;
                    border-radius: 4px;
                    background-color: #d1ecf1;
                    border: 1px solid #bee5eb;
                    color: #0c5460;
                }
            """)
            self.show_both_button.setStyleSheet("background-color: #4CAF50; color: white;")
            self.show_centerline_button.setStyleSheet("background-color: #1976D2; color: white;")
            self.show_surface_button.setStyleSheet("background-color: #FF9800; color: white;")
            
        elif mode == "surface_only":
            if self.surface_actor:
                self.surface_actor.SetVisibility(1)
            if self.centerline_actor:
                self.centerline_actor.SetVisibility(0)
            self.display_label.setText("🔍 显示模式: 仅主动脉")
            self.display_label.setStyleSheet("""
                QLabel {
                    font-size: 11px;
                    padding: 8px;
                    border-radius: 4px;
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    color: #856404;
                }
            """)
            self.show_both_button.setStyleSheet("background-color: #4CAF50; color: white;")
            self.show_centerline_button.setStyleSheet("background-color: #2196F3; color: white;")
            self.show_surface_button.setStyleSheet("background-color: #F57C00; color: white;")
        
        self.vtk_widget.GetRenderWindow().Render()
        logging.info(f"切换显示模式: {mode}")

    def resetSelection(self):
        """重新选择点"""
        try:
            # 移除起点标记
            if self.actor_start:
                self.renderer.RemoveActor(self.actor_start)
                self.actor_start = None
                logging.info("移除起点标记")
            
            # 移除终点标记
            if self.actor_end:
                self.renderer.RemoveActor(self.actor_end)
                self.actor_end = None
                logging.info("移除终点标记")
            
            # 移除临时标记
            if self.temp_actor:
                self.renderer.RemoveActor(self.temp_actor)
                self.temp_actor = None
                logging.info("移除临时标记")
            
            # 移除中心线
            if self.centerline_actor:
                self.renderer.RemoveActor(self.centerline_actor)
                self.centerline_actor = None
                logging.info("移除中心线")
            
            # 移除直径显示
            self.removeDiameterDisplay()
            
            # 重置中心线数据
            self.centerline_data = None
            self.centerline_points = None
            self.centerline_radii = None
            self.centerline_diameters = None
                
            # 重置点选择状态
            self.start_point = None
            self.end_point = None
            self.points_selected = 0
            
            # 重要：不要重置 current_picker_position，让鼠标移动事件来更新它
            # self.current_picker_position = None  # 注释掉这行
            
            # 启用计算按钮
            self.compute_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            
            # 禁用显示控制按钮
            self.show_both_button.setEnabled(False)
            self.show_centerline_button.setEnabled(False)
            self.show_surface_button.setEnabled(False)
            self.show_both_button.setStyleSheet("background-color: #bdc3c7; color: #7f8c8d;")
            self.show_centerline_button.setStyleSheet("background-color: #bdc3c7; color: #7f8c8d;")
            self.show_surface_button.setStyleSheet("background-color: #bdc3c7; color: #7f8c8d;")
            
            # 重置显示模式
            self.setDisplayMode("both")
            
            # 更新状态标签
            self.status_label.setText("🔵 等待选择起点\n移动鼠标到表面，按空格键选择起点")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 11px;
                    padding: 8px;
                    border-radius: 4px;
                    background-color: #d1ecf1;
                    border: 1px solid #bee5eb;
                    color: #0c5460;
                }
            """)
            self.diameter_info_label.setText("📏 直径信息: 计算中心线后查看")
            self.diameter_info_label.setStyleSheet("""
                QLabel {
                    font-size: 11px;
                    padding: 8px;
                    border-radius: 4px;
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    color: #856404;
                }
            """)
            
            # 强制重新渲染
            self.vtk_widget.GetRenderWindow().Render()
            
            # 重新初始化交互器（重要修复）
            if hasattr(self, 'interactor') and self.interactor:
                self.interactor.ReInitialize()
            
            logging.info("重置所有选择完成")
            
        except Exception as e:
            error_msg = f"重置选择时出错: {e}"
            logging.error(error_msg, exc_info=True)

    def computeCenterlines(self):
        """计算中心线"""
        if not self.start_point or not self.end_point:
            QMessageBox.warning(self, "错误", "请先选择起点和终点")
            return
            
        logging.info("Step 2: Computing centerlines...")
        
        try:
            logging.info("使用方法1: pointlist方式")
            centerlineGenerator = vmtkscripts.vmtkCenterlines()
            centerlineGenerator.Surface = self.surface
            centerlineGenerator.SeedSelectorName = "pointlist"
            
            source_points = [
                self.start_point[0], self.start_point[1], self.start_point[2]
            ]
            target_points = [
                self.end_point[0], self.end_point[1], self.end_point[2]
            ]
            
            logging.info(f"源点: {source_points}")
            logging.info(f"目标点: {target_points}")
            
            centerlineGenerator.SourcePoints = source_points
            centerlineGenerator.TargetPoints = target_points
            centerlineGenerator.RadiusArrayName = "MaximumInscribedSphereRadius"
            centerlineGenerator.AppendEndPoints = 1
            centerlineGenerator.Execute()
            
            cl = centerlineGenerator.Centerlines
            
            if cl is None or cl.GetNumberOfPoints() == 0:
                logging.warning("方法1失败，尝试方法2: 使用表面点方式")
                centerlineGenerator = vmtkscripts.vmtkCenterlines()
                centerlineGenerator.Surface = self.surface
                centerlineGenerator.SeedSelectorName = "pointlist"
                
                start_surface_point = self.findClosestSurfacePoint(self.start_point)
                end_surface_point = self.findClosestSurfacePoint(self.end_point)
                
                source_points = [
                    start_surface_point[0], start_surface_point[1], start_surface_point[2]
                ]
                target_points = [
                    end_surface_point[0], end_surface_point[1], end_surface_point[2]
                ]
                
                logging.info(f"表面源点: {source_points}")
                logging.info(f"表面目标点: {target_points}")
                
                centerlineGenerator.SourcePoints = source_points
                centerlineGenerator.TargetPoints = target_points
                centerlineGenerator.RadiusArrayName = "MaximumInscribedSphereRadius"
                centerlineGenerator.AppendEndPoints = 1
                centerlineGenerator.Execute()
                
                cl = centerlineGenerator.Centerlines
            
            if cl is None or cl.GetNumberOfPoints() == 0:
                logging.warning("方法2失败，尝试方法3: pickpoint方式")
                QMessageBox.information(self, "提示", "将使用交互式选择模式，请在3D窗口中依次选择起点和终点")
                
                centerlineGenerator = vmtkscripts.vmtkCenterlines()
                centerlineGenerator.Surface = self.surface
                centerlineGenerator.SeedSelectorName = "pickpoint"
                centerlineGenerator.RadiusArrayName = "MaximumInscribedSphereRadius"
                centerlineGenerator.Execute()
                
                cl = centerlineGenerator.Centerlines

            if cl is None or cl.GetNumberOfPoints() == 0:
                raise Exception("无法生成中心线，请尝试重新选择点")
                
            logging.info(f"Centerline generated with {cl.GetNumberOfPoints()} points")
            
            self.extractCenterlineData(cl)
            self.displayCenterline(cl)
            
            self.centerline_data = cl
            self.storeCenterlinePoints(cl)
            
            self.show_both_button.setEnabled(True)
            self.show_centerline_button.setEnabled(True)
            self.show_surface_button.setEnabled(True)
            
            success_msg = "中心线计算完成！数据已保存到 centerline_diameters.csv\n\n可以使用按钮切换显示模式：\n- 显示主动脉+中心线\n- 只显示中心线\n- 只显示主动脉\n\n快捷键：1=全部显示，2=只显示中心线，3=只显示主动脉\n\n现在可以将鼠标放在中心线上查看实时直径信息！"
            logging.info(success_msg)
            QMessageBox.information(self, "完成", success_msg)
            
        except Exception as e:
            error_msg = f"计算中心线时出错: {str(e)}"
            logging.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "错误", error_msg)

    def storeCenterlinePoints(self, cl):
        """存储中心线点坐标和直径数据"""
        points = cl.GetPoints()
        n_points = cl.GetNumberOfPoints()
        
        self.centerline_points = []
        for i in range(n_points):
            self.centerline_points.append(points.GetPoint(i))
        
        radius_array = cl.GetPointData().GetArray("MaximumInscribedSphereRadius")
        if radius_array is None:
            radius_array_names = ["Radius", "radius", "MaximumInscribedSphereRadius", "InscribedSphereRadius"]
            for name in radius_array_names:
                radius_array = cl.GetPointData().GetArray(name)
                if radius_array:
                    break
            
            if radius_array is None:
                radius_array = vtk.vtkDoubleArray()
                radius_array.SetName("MaximumInscribedSphereRadius")
                radius_array.SetNumberOfValues(n_points)
                for i in range(n_points):
                    radius_array.SetValue(i, 1.0)
        
        self.centerline_radii = numpy_support.vtk_to_numpy(radius_array)
        self.centerline_diameters = 2 * self.centerline_radii
        
        logging.info(f"存储了 {len(self.centerline_points)} 个中心线点的数据")

    def findClosestSurfacePoint(self, point):
        """找到表面上离给定点最近的点坐标"""
        if not self.surface:
            return point
            
        point_locator = vtk.vtkPointLocator()
        point_locator.SetDataSet(self.surface)
        point_locator.BuildLocator()
        
        closest_point_id = point_locator.FindClosestPoint(point)
        closest_point = self.surface.GetPoint(closest_point_id)
        return closest_point

    def displayCenterline(self, centerline):
        """在3D视图中显示中心线"""
        if self.centerline_actor:
            self.renderer.RemoveActor(self.centerline_actor)
        
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputData(centerline)
        tube_filter.SetRadius(0.3)
        tube_filter.SetNumberOfSides(12)
        tube_filter.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube_filter.GetOutputPort())
        
        self.centerline_actor = vtk.vtkActor()
        self.centerline_actor.SetMapper(mapper)
        self.centerline_actor.GetProperty().SetColor(0, 1, 0)
        self.centerline_actor.GetProperty().SetLineWidth(4)
        
        self.renderer.AddActor(self.centerline_actor)
        self.vtk_widget.GetRenderWindow().Render()

    def extractCenterlineData(self, cl):
        """提取中心线数据并保存到CSV"""
        points = cl.GetPoints()
        n_points = cl.GetNumberOfPoints()

        if n_points == 0:
            raise Exception("中心线没有点数据")

        radius_array = cl.GetPointData().GetArray("MaximumInscribedSphereRadius")
        if radius_array is None:
            radius_array_names = ["Radius", "radius", "MaximumInscribedSphereRadius", "InscribedSphereRadius"]
            for name in radius_array_names:
                radius_array = cl.GetPointData().GetArray(name)
                if radius_array:
                    logging.info(f"找到半径数组: {name}")
                    break
            
            if radius_array is None:
                logging.warning("未找到半径数据，使用默认值1.0")
                radius_array = vtk.vtkDoubleArray()
                radius_array.SetName("MaximumInscribedSphereRadius")
                radius_array.SetNumberOfValues(n_points)
                for i in range(n_points):
                    radius_array.SetValue(i, 1.0)
                cl.GetPointData().AddArray(radius_array)

        radii = numpy_support.vtk_to_numpy(radius_array)
        diameters = 2 * radii

        coords = np.zeros((n_points, 3))
        for i in range(n_points):
            coords[i] = points.GetPoint(i)

        with open("results/centerline_diameters.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["PointIndex", "X", "Y", "Z", "Radius", "Diameter"])
            for i in range(n_points):
                writer.writerow([i, coords[i][0], coords[i][1], coords[i][2], radii[i], diameters[i]])

        logging.info(f"Centerline diameter data saved to results/centerline_diameters.csv")
        logging.info(f"Diameter range: {diameters.min():.3f} ~ {diameters.max():.3f} mm")

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    window = CenterlinePicker()
    window.show()
    
    window.vtk_widget.Start()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
