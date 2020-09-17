import sys
import types
from PyQt5.QtWidgets import QMainWindow, QApplication, QMenu, QMenuBar, QAction, QFileDialog, QLabel,QButtonGroup, QWidget, QListWidgetItem
from PaintView import *
from NavigationView3D import *
import cv2
import numpy as np
from numpy import linalg as LA
import math
import rendererpyw as rd
import torch
import torchvision
import time
from pyquaternion import Quaternion
import threading
import qimage2ndarray
import sketch as sk # for generate sketch from the shape
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint,QRect
from PyQt5.QtGui import  QIcon, QImage, QPainter, QPen, QBrush, QPixmap, QColor

import projection

#globalGrid = None

class MyThread(QThread):
    # Create a counter thread
    change_value = pyqtSignal(int)
    def run(self):
        cnt = 0
        while cnt <100:
            cnt+=5
            time.sleep(0.3)
            self.change_value.emit(cnt)


COLORS = [
    '#000000', '#82817f', '#820300', '#868417', '#007e03', 
    '#ffffff', '#c1c1c1', '#f70406', '#fffd00', '#08fb01', 
]

MODES = ['pen','eraser','fill']

CANVAS_DIMENSIONS = 256, 256
class Canvas(QLabel):
    primary_color = QColor(Qt.black)  #for drawing
    secondary_color = None
    active_color = None
    mode = 'pen'   #
    def __init__(self, filePath):
        super().__init__()
        self.background_color = QColor(Qt.white)
        self.eraser_color =  QColor(Qt.white)
        self.eraser_color.setAlpha(100)
        self.imagefilePath = filePath  #for updating maybe
        self.reset()

    def reset(self):
        # Create the pixmap for display.
        self.setPixmap(QPixmap(*CANVAS_DIMENSIONS))
        self.setMinimumSize(1,1)
        self.setMaximumSize(256,256) #currently

        # Clear the canvas.
        self.pixmap().fill(self.background_color)
       
    # Eraser events
    def eraser_mousePressEvent(self, e):
        self.generic_mousePressEvent(e)

    def eraser_mouseMoveEvent(self, e):
        #print(e.x())
        #print(e.y())

        image = self.pixmap().toImage()
        w, h = image.width(), image.height()
        x, y = e.x(), e.y()

        have_seen = set()
        queue = [(x, y)]

        def get_cardinal_points(have_seen, center_pos):
            points = []
            cx, cy = center_pos
            for x, y in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                xx, yy = cx + x, cy + y
                if (xx >= 0 and xx < w and
                    yy >= 0 and yy < h and
                    (xx, yy) not in have_seen):
                    points.append((xx, yy))
                    have_seen.add((xx, yy))
            return points

        if self.last_pos:
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.eraser_color, 30, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            print(e.pos())
            p.drawLine(self.last_pos, e.pos())
            self.last_pos = e.pos()
            self.update()

    def eraser_mouseReleaseEvent(self, e):
        self.generic_mouseReleaseEvent(e)

    def mousePressEvent(self, e):
        fn = getattr(self, "%s_mousePressEvent" % self.mode, None)
        if fn:
            return fn(e)

    def mouseMoveEvent(self, e):
        fn = getattr(self, "%s_mouseMoveEvent" % self.mode, None)
        if fn:
            return fn(e)

    def mouseReleaseEvent(self, e):
        fn = getattr(self, "%s_mouseReleaseEvent" % self.mode, None)
        if fn:
            return fn(e)

    def mouseDoubleClickEvent(self, e):
        fn = getattr(self, "%s_mouseDoubleClickEvent" % self.mode, None)
        if fn:
            return fn(e)

    # Generic events (shared by brush-like tools)
    def generic_mousePressEvent(self, e):
        self.last_pos = e.pos()
        if e.button() == Qt.LeftButton:
            self.active_color = self.primary_color
        else:
            self.active_color = self.secondary_color

    def generic_mouseReleaseEvent(self, e):
        self.last_pos = None

    # Pen events
    def pen_mousePressEvent(self, e):
        self.generic_mousePressEvent(e)

    def pen_mouseMoveEvent(self, e):
        #print(e.x())
        #print(e.y())
        if self.last_pos:
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.active_color, 1, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin))
            p.drawLine(self.last_pos, e.pos())

            self.last_pos = e.pos()
            self.update()

    def pen_mouseReleaseEvent(self, e):
        self.generic_mouseReleaseEvent(e)

    def set_mode(self, mode):      
        self.current_pos = None
        self.last_pos = None
        # Apply the mode - eraser or pen
        self.mode = mode
    
    def set_primary_color(self, hex):
        self.primary_color = QColor(hex)

     # Fill events
    def fill_mousePressEvent(self, e):
        #print(e.x())
        #print(e.y())
        if e.button() == Qt.LeftButton:
            self.active_color = self.primary_color
        else:
            self.active_color = self.secondary_color

        image = self.pixmap().toImage()
        w, h = image.width(), image.height()
        x, y = e.x(), e.y()

        # Get our target color from origin.
        target_color = image.pixel(x,y)

        have_seen = set()
        queue = [(x, y)]

        def get_cardinal_points(have_seen, center_pos):
            points = []
            cx, cy = center_pos
            for x, y in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                xx, yy = cx + x, cy + y
                if (xx >= 0 and xx < w and
                    yy >= 0 and yy < h and
                    (xx, yy) not in have_seen):

                    points.append((xx, yy))
                    have_seen.add((xx, yy))

            return points

        # Now perform the search and fill.
        p = QPainter(self.pixmap())
        p.setPen(QPen(self.active_color))

        while queue:
            x, y = queue.pop()
            if image.pixel(x, y) == target_color:
                p.drawPoint(QPoint(x, y))
                queue.extend(get_cardinal_points(have_seen, (x, y)))
        self.update()

    #for new image
    def initialize(self):
        self.background_color = QColor(Qt.white)
        self.eraser_color =  QColor(Qt.white)
        self.eraser_color.setAlpha(100)
        self.reset()
    

#class thumbnail, maybe not in needed
class ThumbLabel(QLabel):  
    def __init__(self,widget, canvas, appwindow, CameraPos):  #related to the cavans for recovering the Thumblabel
        if widget!=None:
            super().__init__(widget)   #previous implmentation for adding in centralwidget
        else:
            super().__init__()  #derictly with QLabel for QlistView
        self.Canvas = canvas
        self.background_color= QColor(Qt.white)
        self.bigpixmap = QPixmap(256, 256)
        self.bigpixmap.fill(self.background_color)   #default white
        self.CameraPos = CameraPos
        self.AppWindow = appwindow
        self.grid = None   # try later
        self.reset()

    def update_grid(self, newgrid):
        self.grid = newgrid

    def reset(self):
        # Create the pixmap for display.
        self.setScaledContents(True);
        self.setPixmap(QPixmap(36,36))
        self.setMinimumSize(1,1)
        self.setMaximumSize(36,36) #currently
        # Clear the canvas.
        self.pixmap().fill(self.background_color)     
    
    def setbigmap(self,pixmap):
        self.bigpixmap = pixmap.copy()  #copy to avoid memory problem
    def mouseDoubleClickEvent(self, event):
        pixmap = self.bigpixmap.copy()  #recover thumbnail to canvas
        self.Canvas.setPixmap(pixmap)
        self.update_grid(self.AppWindow.getgrid())
        #print(self.CameraPos)
        #global globalGrid
        
        if self.grid is not None:
            rd.update_image(self.CameraPos,self.grid ,"opened")
        else:
            print("Grid is None")
        
        if self.AppWindow is not None:
            self.AppWindow.Update_image()
        else:
            print("AppWindow is None")
    def setCanvas(self,canvas):
        self.Canvas = canvas
    def setCameraPos(self, camerapos):
        self.Camerapos = camerapos
    def setPaintWindow(self, window):
        self.AppWindow = window

class QCustomQWidget(QWidget):   # add the grid parameter
    def __init__(self, AppWindow, CameraPos, parent=None):  #
        super(QCustomQWidget, self).__init__(parent)
        self.textQVBoxLayout = QtWidgets.QVBoxLayout()
        self.textUpQLabel    = QLabel()
        self.textDownQLabel  = QLabel()
        # Up an Down Texts  -Later only Camera View: (0, 0, 0)
        self.textQVBoxLayout.addWidget(self.textUpQLabel)
        self.textQVBoxLayout.addWidget(self.textDownQLabel)

        self.AppWindow = AppWindow
        self.CameraPos = CameraPos
        self.grid = None   #currently now, update later
        
        self.allQHBoxLayout  = QtWidgets.QHBoxLayout()
        self.iconQLabel      = ThumbLabel(None, None, self.AppWindow, self.CameraPos)
        self.iconQLabel.setScaledContents(True);
        #self.update_grid(None)  #for update the grid for iconQlabel

        self.allQHBoxLayout.addWidget(self.iconQLabel, 0)
        self.allQHBoxLayout.addLayout(self.textQVBoxLayout, 1)

        self.setLayout(self.allQHBoxLayout)
        # setStyleSheet  - Blue and Red
        self.textUpQLabel.setStyleSheet('''
            color: rgb(0, 0, 255);
        ''')
        self.textDownQLabel.setStyleSheet('''
            color: rgb(255, 0, 0);
        ''')

    def update_grid(self,grid):
        self.grid = grid
        if self.iconQLabel is not None:
            self.iconQLabel.update_grid(grid)

    def setTextUp (self, text):
        self.textUpQLabel.setText(text)

    def setTextDown (self, text):
        self.textDownQLabel.setText(text)

    def setCanvas(self,canvas):
        self.iconQLabel.setCanvas(canvas)   #relating to the current canvas for recovering

    def setIconLabel(self,pixmap):
        self.iconQLabel.setPixmap(pixmap)

    def setIcon (self, filepath):  #pass the drawed map into thumbnal 
        pixmap = QPixmap()
        pixmap.load(filepath)
        self.iconQLabel.setPixmap(pixmap)
    
    def setBigmap(self,pixmap):
        self.iconQLabel.setbigmap(pixmap)  #for thumbnail recovering
 
#paint
class PaintWindow(QMainWindow):
    def listToString(self,s):  
    # initialize an empty string 
        s = list(map(int, s)) 
        str1 = ' '.join(map(str, s)) 

    # return string   
        return str1 
    def editList(self, row, pixmap, cw, ch, camerapose=''): #row is an int here, for 6 views actually
        #edit the size first
        #print("Edit the list")
        bigmap = pixmap.copy()
        pixmap = pixmap.scaledToWidth(cw)
        pixmap = pixmap.scaledToHeight(ch)
        pixmap = pixmap.copy()
        cntItem = self.ui.ThumbnailListView.item(row)  
        myQCustomQWidget = self.ui.ThumbnailListView.itemWidget(cntItem)
        myQCustomQWidget.setIconLabel(pixmap)
        if self.grid is None:   #currently may be None since grid has not generated yet.
            print("NULL Grid")
        myQCustomQWidget.update_grid(self.grid)    #update grid here
        if camerapose !='':
            name =  'View: ('+self.listToString(camerapose)+')'  #could be none
        else:
            name =  'View: ('+str(camerapose)+')'  #could be none
        myQCustomQWidget.setTextDown(name)
        # for recovering
        myQCustomQWidget.setBigmap(bigmap)
    
    def addList(self,pixmap,cw,ch,camerapose=''): #add pixmap and no, remember to pass in the camerapos now
        #print("Add to List")
        index = 'No. '+str(self.cnt)
        if camerapose !='':
            name =  'View: ('+self.listToString(camerapose)+')'  #could be none
        else:
            name =  'View: ('+str(camerapose)+')'  #could be none
        myQCustomQWidget = QCustomQWidget(self.AppWindow, camerapose)
        myQCustomQWidget.setTextUp(index)
        myQCustomQWidget.setTextDown(name)
        myQCustomQWidget.setCanvas(self.ui.Canvas)

        #if self.grid ==None:  grid is there to update
        #    print("Error");
        bigmap = pixmap.copy()
        pixmap = pixmap.scaledToWidth(cw)
        pixmap = pixmap.scaledToHeight(ch)
        pixmap = pixmap.copy()
        myQCustomQWidget.setIconLabel(pixmap)
        myQCustomQWidget.update_grid(self.grid)    # update grid here

        # for recovering
        myQCustomQWidget.setBigmap(bigmap)
        # Create QListWidgetItem
        myQListWidgetItem = QListWidgetItem(self.ui.ThumbnailListView)
        # Set size hint
        myQListWidgetItem.setSizeHint(myQCustomQWidget.sizeHint())
        # Add QListWidgetItem into QListWidget
        self.ui.ThumbnailListView.addItem(myQListWidgetItem)
        self.ui.ThumbnailListView.setItemWidget(myQListWidgetItem, myQCustomQWidget)
 
    def initThumbnailItems(self):
        #print("Init thumbnail")
        self.ui.leftThumbnail.deleteLater()  
        self.ui.rightThumbnail.deleteLater()
        self.ui.bottomThumbnail.deleteLater()
        self.ui.topThumbnail.deleteLater()
        self.ui.frontThumbnail.deleteLater()
        self.ui.backThumbnail.deleteLater()

        for index, name, icon in [('No. 1', 'Left View: (5, 0, 0)',  'icons/stamp.png'),
            ('No. 2', 'Right View: (-5, 0, 0)', 'icons/stamp.png'),  # didn't use this png actually here
            ('No. 3', 'Front View: (0, 0, 5)',  'icons/stamp.png'),
            ('No. 4', 'Back View: (0, 0, -5)',  'icons/stamp.png'),
            ('No. 5', 'Top View: (0, 5, 0)',  'icons/stamp.png'),
            ('No. 6', 'Bottom View: (0, -5, 0)',  'icons/stamp.png'),
            ]:

            self.cnt+=1  #for later counting
             # Create QCustomQWidget

            cameraVals = ((name[name.find("(")+1:name.find(")")]).split(",")) #splits by parenthesis and gets camera values from name i.e gets [5,0,0], [-5,0,0] ,etc from name 
            cameraPos = list(map(int, cameraVals))                            #converts cameraVals to list to pass into widget
            myQCustomQWidget = QCustomQWidget(self.AppWindow, cameraPos)
            myQCustomQWidget.setTextUp(index)
            myQCustomQWidget.setTextDown(name)
            myQCustomQWidget.setCanvas(self.ui.Canvas)
            #myQCustomQWidget.setIcon(icon)
            myQCustomQWidget.update_grid(self.grid)

            # Create QListWidgetItem
            myQListWidgetItem = QListWidgetItem(self.ui.ThumbnailListView)
            # Set size hint
            myQListWidgetItem.setSizeHint(myQCustomQWidget.sizeHint())
            # Add QListWidgetItem into QListWidget
            self.ui.ThumbnailListView.addItem(myQListWidgetItem)
            self.ui.ThumbnailListView.setItemWidget(myQListWidgetItem, myQCustomQWidget)


    def initThumbnailItemsTest(self):  #for testing item adding dynamically (in the initialization phase)
        for index, name, icon in [
            ('No.1', 'Meyoko',  'icons/stamp.png'),
            ('No.2', 'Nyaruko', 'icons/stamp.png'),
            ('No.3', 'Louise',  'icons/stamp.png')]:
            # Create QCustomQWidget
            myQCustomQWidget = QCustomQWidget()
            myQCustomQWidget.setTextUp(index)
            myQCustomQWidget.setTextDown(name)
            myQCustomQWidget.setIcon(icon)
            # Create QListWidgetItem
            myQListWidgetItem = QListWidgetItem(self.ui.ThumbnailListView)
            # Set size hint
            myQListWidgetItem.setSizeHint(myQCustomQWidget.sizeHint())
            # Add QListWidgetItem into QListWidget
            self.ui.ThumbnailListView.addItem(myQListWidgetItem)
            self.ui.ThumbnailListView.setItemWidget(myQListWidgetItem, myQCustomQWidget)


    def configure(self,filepath):
        #include set filepath and the canvas pixmap
        self.imagefilePath = filepath
        pixmap = QPixmap(self.imagefilePath)   #then load the image to this window
        self.ui.Canvas.setPixmap(pixmap)

    
    def update_grid(self,newgrid):
        self.grid = newgrid
    def __init__(self, filePath, appWindow):  #adding the grid here
        super().__init__()
        self.ui= Ui_PaintWindow()  #directly self = this object of that class
        self.ui.setupUi(self) 

        self.AppWindow = appWindow
        self.grid = None   #for containing the grid from the windows 1

        self.cnt = 1  #
        self.imagefilePath = filePath
        pixmap = QPixmap(self.imagefilePath)
        self.Pixmap = pixmap

        # Replace canvas placeholder from QtDesigner.
        self.ui.horizontalLayout.removeWidget(self.ui.Canvas)
        self.ui.Canvas = Canvas(None)
        self.ui.horizontalLayout.addWidget(self.ui.Canvas)
        self.initThumbnailItems()   #then initialize for listView

        # initialization 
        self.drawing = False
        self.brushSize = 2
        self.brushColor = Qt.black
        self.lastPoint = None

        #action assigned
        self.ui.actionSave.triggered.connect(self.save_file)
        self.ui.actionOpen.triggered.connect(self.open_file)
        self.ui.actionClear.triggered.connect(self.clear)
        self.ui.actionNew.triggered.connect(self.ui.Canvas.initialize)

        # two buttons assign
        # Setup the mode buttons
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)

        for mode in MODES:
            btn = getattr(self.ui, '%sButton' % mode)
            btn.pressed.connect(lambda mode=mode: self.ui.Canvas.set_mode(mode))
            mode_group.addButton(btn)

    
        # Initialize button colours.
        for n, hex in enumerate(COLORS, 1):
            btn = getattr(self.ui, 'colorButton_%d' % n)
            btn.setStyleSheet('QPushButton { background-color: %s; }' % hex)
            btn.hex = hex  # For use in the event below

            def patch_mousePressEvent(self_, e):
                if e.button() == Qt.LeftButton:
                    self.set_primary_color(self_.hex)

                elif e.button() == Qt.RightButton:
                    self.set_secondary_color(self_.hex)

            btn.mousePressEvent = types.MethodType(patch_mousePressEvent, btn)

        # Initialize for the view buttons
        viewgroup = QtWidgets.QButtonGroup()
        viewgroup.addButton(self.ui.leftButton)
        viewgroup.addButton(self.ui.rightButton)
        viewgroup.addButton(self.ui.frontButton)
        viewgroup.addButton(self.ui.backButton)
        viewgroup.addButton(self.ui.topButton)
        viewgroup.addButton(self.ui.bottomButton)

        self.ui.bottomButton.toggled.connect(self.BottomView)
        self.ui.topButton.toggled.connect(self.TopView)
        self.ui.backButton.toggled.connect(self.BackView)
        self.ui.frontButton.toggled.connect(self.FrontView)
        self.ui.leftButton.toggled.connect(self.LeftView)
        self.ui.rightButton.toggled.connect(self.RightView)

        # for add_button
        self.ui.AddButton.clicked.connect(self.Add_Function)
        self.windows2 = None 
        self.ViewList = []  #initialized firstly as null list for reconstructing 3D objects from mask images
        self.CameraList = []
        self.View =[5,0,0]  #default view
        # link initialization button
        self.ui.InitializeButton.clicked.connect(self.Initialize_Button)
        self.ui.UpdateButton.clicked.connect(self.Update_Button)

        # for thumbnail
        #self.ui.leftThumbnail.deleteLater()
        #self.ui.leftThumbnail = ThumbLabel(self.ui.centralwidget,self.ui.Canvas)
        #self.ui.leftThumbnail.setGeometry(QtCore.QRect(420, 60, 36, 36))  

    # ViewList here is used to generate 3D Objects again
    def Update_Button(self):
        # call add_function first and initialize again
        self.Add_Function()
        cw, ch = 36 , 36 
        cntbounding_box_min_x = -2
        cntbounding_box_max_x = 2
        grid_res_x = 64 
        cntwidth = 256
        cntheight = 256 
        grid_now = sk.initialization(self.ViewList, self.CameraList,cntbounding_box_min_x,cntbounding_box_max_x,grid_res_x,cntwidth, cntheight) 
        #image = rd.generate_image(-2,-2,-2,2,2,2, 4. / (grid_res_x-1), grid_res_x, grid_res_x, grid_res_x, cntwidth, cntheight, grid_now, self.CameraList[0], 1)
        #torchvision.utils.save_image(image, "./test.png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
        #global globalGrid
        #globalGrid = grid_now
        #self.AppWindow.saveFile_Mask()
        self.grid = grid_now   #update
        for index, camera in [(0, [5,0,0]),
            (1, [-5, 0, 0]),
            (2, [0, 0, 5])
            #(3, [0, 0, -5]),
            #(4, [0, 5, 0]),  
            #(5, [0, -5, 0])
            ]:

            rd.update_mask(list(camera),grid_now,"mask")   #produce the mask
            self.maskfilepath = "results/mask.png"  #set up an filepath for the mask file
            #configure the second window
            self.AppWindow.windows2.configure(self.maskfilepath)
            pixmap = self.ui.Canvas.pixmap()
            self.convert_imagetoTensor(pixmap,list(camera))
            self.editList(index,pixmap,cw,ch,self.View)
            self.windows2.update_grid(grid_now)
            self.windows2.activateWindow();  #active the right onw

    def set_primary_color(self, hex):
        self.ui.Canvas.set_primary_color(hex)

    #choose the current one and unable the other checkboxes
    def BottomView(self):
        self.ui.topButton.setChecked(False)
        self.ui.leftButton.setChecked(False)
        self.ui.rightButton.setChecked(False)
        self.ui.frontButton.setChecked(False)
        self.ui.backButton.setChecked(False)

    def TopView(self):
        self.ui.bottomButton.setChecked(False)
        self.ui.leftButton.setChecked(False)
        self.ui.rightButton.setChecked(False)
        self.ui.frontButton.setChecked(False)
        self.ui.backButton.setChecked(False)    

    def LeftView(self):
        self.ui.bottomButton.setChecked(False)
        self.ui.topButton.setChecked(False)
        self.ui.rightButton.setChecked(False)
        self.ui.frontButton.setChecked(False)
        self.ui.backButton.setChecked(False)

    def RightView(self):
        self.ui.bottomButton.setChecked(False)
        self.ui.topButton.setChecked(False)
        self.ui.leftButton.setChecked(False)
        self.ui.frontButton.setChecked(False)
        self.ui.backButton.setChecked(False) 

    def BackView(self):
        self.ui.topButton.setChecked(False)
        self.ui.leftButton.setChecked(False)
        self.ui.rightButton.setChecked(False)
        self.ui.frontButton.setChecked(False)
        self.ui.bottomButton.setChecked(False)

    def FrontView(self):
        self.ui.topButton.setChecked(False)
        self.ui.leftButton.setChecked(False)
        self.ui.rightButton.setChecked(False)
        self.ui.backButton.setChecked(False)
        self.ui.bottomButton.setChecked(False)     

   
    def clear(self):      #potentially clear viewlist and camera list as well
        self.ui.Canvas.reset()  
        #self.update()
    
    def returnProcess(self):
        #back 
        self.NavigationView = AppWindow(None)  #back to call View
        self.NavigationView.show()
        self.close()

    def open_file(self):
        """
        Open image file for editing, scaling the smaller dimension and cropping the remainder.
        :return:
        """
        path, _ = QFileDialog.getOpenFileName(self, "Open file", "", "PNG image files (*.png); JPEG image files (*jpg); All files (*.*)")

        if path:
            pixmap = QPixmap()
            pixmap.load(path)

            # We need to crop down to the size of our canvas. Get the size of the loaded image.
            iw = pixmap.width()
            ih = pixmap.height()

            # Get the size of the space we're filling.
            cw, ch = CANVAS_DIMENSIONS

            if iw/cw < ih/ch:  # The height is relatively bigger than the width.
                pixmap = pixmap.scaledToWidth(cw)
                hoff = (pixmap.height() - ch) // 2
                pixmap = pixmap.copy(
                    QRect(QPoint(0, hoff), QPoint(cw, pixmap.height()-hoff))
                )

            elif iw/cw > ih/ch:  # The height is relatively bigger than the width.
                pixmap = pixmap.scaledToHeight(ch)
                woff = (pixmap.width() - cw) // 2
                pixmap = pixmap.copy(
                    QRect(QPoint(woff, 0), QPoint(pixmap.width()-woff, ch))
                )
            self.imagefilePath = path  # for the opened sketch files
            self.ui.Canvas.setPixmap(pixmap)
        
    def save_file(self):
        """
        Save active canvas to image file.
        :return:
        """
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "PNG Image file (*.png)")
        if path:
            pixmap = self.ui.Canvas.pixmap()
            pixmap.save(path, "PNG" )
        
    #make sure they can refer each other
    def addWindow(self,newwindow):
        self.windows2=newwindow

    def convert_imagetoTensor(self,pixmap, camerapos): # and add to the sketch list
        v = qimage2ndarray.recarray_view(pixmap.toImage())['r']
        tv = torch.from_numpy(v).type(torch.cuda.FloatTensor)
        tv[tv>0] = 1   #white from 255 to 1
        # if new
        if camerapos in self.CameraList:
            cntindex = self.CameraList.index(torch.cuda.FloatTensor(camerapos))
            self.ViewList[cntindex] = tv  #update
            Dflag = True   # for add thumbnail item or edit
        else:     
        # or update ViewList only
            self.ViewList.append(tv)   # for mask
            self.CameraList.append(torch.cuda.FloatTensor(camerapos))  #for cameraList
            Dflag = False   # for add thumbnail item or edit
        
        return Dflag

    def Initialize_Button(self):   #default 500 back then 
        # call initialization function
        cntbounding_box_min_x = -2
        cntbounding_box_max_x = 2
        grid_res_x = 64 
        cntwidth = 256
        cntheight = 256

        grid_now = sk.initialization(self.ViewList, self.CameraList,cntbounding_box_min_x,cntbounding_box_max_x,grid_res_x,cntwidth, cntheight) 
        #image = rd.generate_image(-2,-2,-2,2,2,2, 4. / (grid_res_x-1), grid_res_x, grid_res_x, grid_res_x, cntwidth, cntheight, grid_now, self.CameraList[0], 1)
        #torchvision.utils.save_image(image, "./test.png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
        self.windows2.update_grid(grid_now)
        #global globalGrid
        #globalGrid = grid_now
        self.grid = grid_now 
        print("Initialized")
        projection.projectXY(self.grid, 256, 256)
        self.windows2.startProgressBar()
        self.windows2.activateWindow();  #active the right onw
    
    #pose could be a list stuff?
    def Add_ViewPose(self, pose):  # add viewpose from 3D navigation view for adding in the paint view - reset ViewList and Camera List
        self.View = pose

    # for the fundamental 6 views
    def Edit_ViewandCameraList(self):
        s=3   #ignore this
    # consider add list now - 
    def Add_Function(self):
        flag = False   # this flag is for adding other views except the 6 standard views
        # crop first, 
        cw, ch = 36, 36
        #utilize the structure of ViewChosen fun
        if self.ui.bottomButton.isChecked() == True:
            flag = True
            pixmap = self.ui.Canvas.pixmap()
            self.convert_imagetoTensor(pixmap,[0,-5,0])
            self.editList(5,pixmap,cw,ch,[0,-5,0])  
        if self.ui.topButton.isChecked() == True:
            flag = True
            pixmap = self.ui.Canvas.pixmap()
            self.convert_imagetoTensor(pixmap,[0,5,0])
            self.editList(4,pixmap,cw,ch,[0,5,0]) 
            #self.ui.topThumbnail.setPixmap(pixmap)
        if self.ui.leftButton.isChecked() ==True:
            flag = True
            pixmap = self.ui.Canvas.pixmap()
            self.convert_imagetoTensor(pixmap,[5,0,0])
            #self.ui.leftThumbnail.setbigmap(pixmap)  # for recover, little trick here, currently not use this one
            #pixmap = pixmap.copy(
            #        QRect(QPoint(420, 60), QPoint(cw, ch))
            #    )       
            self.editList(0,pixmap,cw,ch,[5,0,0])    
        if self.ui.rightButton.isChecked() ==True:
            flag = True
            pixmap = self.ui.Canvas.pixmap()
            self.convert_imagetoTensor(pixmap,[-5,0,0])
            self.editList(1,pixmap,cw,ch,[-5,0,0])  
        if self.ui.frontButton.isChecked() ==True:
            flag = True
            pixmap = self.ui.Canvas.pixmap()
            self.convert_imagetoTensor(pixmap,[0,1,5.5])
            self.editList(2,pixmap,cw,ch,[0,1,5.5])
        if self.ui.backButton.isChecked() ==True:
            flag = True
            pixmap = self.ui.Canvas.pixmap()
            self.convert_imagetoTensor(pixmap,[0,0,-5])
            self.editList(3,pixmap,cw,ch,[0,0,-5])
        if flag ==False: #adding my own view
            pixmap = self.ui.Canvas.pixmap()
            itemflag =self.convert_imagetoTensor(pixmap,self.View)
            if itemflag ==True:  #edit
                self.editList(pixmap,cw,ch,self.View)  #currently no camera pos input
            else:
                self.addList(pixmap,cw,ch,self.View)
            self.cnt+=1


class AppWindow(QMainWindow):
    def getgrid(self):
        return self.grid
    def update_grid(self, grid):
        self.grid = grid
        #global globalGrid
        #globalGrid = grid
        #currently update the image as well
        rd.update_image(list([5,0,0]),self.grid,"opened")
        self.imagefilepath = "results/opened.png"   #for finding that update image through this file path
        self.Update_image()

    def __init__(self,grid):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.grid = grid #coule be None
        #global globalGrid
        #globalGrid = grid
        self.ui.actionOpen.triggered.connect(self.OpenGridFile)
        self.ui.actionClose.triggered.connect(self.clear)
        self.ui.actionCube.triggered.connect(self.OpenCube)
        self.ui.actionCylinder.triggered.connect(self.OpenCylinder)
        self.count = 0

        #init for rotating parts, label is that image
        self.p1 = np.array([0,0,0])
        self.p2 = np.array([0,0,0])
        self.pose = np.array([0,0,5])
        self.isIn = False

        self.ui.label.mousePressEvent=self.getPosPress
        self.ui.label.mouseMoveEvent = self.getPosMove
        self.ui.label.mouseReleaseEvent=self.getPosRelease
        self.last2d = np.array([0,0])
        self.origintransformation = np.eye(4)
        self.originquaternion = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)   #default is like this (0, 0, 0, 0), could change later to fit for the sketch forming
        self.ui.confirmButton.clicked.connect(self.confirm)
        self.perturbvec = np.array([0,0,5])
        self.clear()
        self.windows2 = None # paint window

    def addWindow(self,newwindow):
        self.windows2=newwindow

    def saveFile_Mask(self):  #care about the mask now
        rd.update_mask(list(self.pose[0:3]),self.grid,"mask")   #produce the mask
        self.maskfilepath = "results/mask.png"  #set up an filepath for the mask file
        #configure the second window
        self.windows2.configure(self.maskfilepath)
        

    def saveFile(self):  #add camera pose needed
        rd.update_sketch(list(self.pose[0:3]),self.grid,"sketch")   #produce the sketch
        self.sketchfilepath = "results/sketch.png"
        # save now
        self.windows2.configure(self.sketchfilepath)
        # pixmap =  QtGui.QPixmap(self.imagefilepath)


    def OpenCube(self):
        self.grid,self.imagefilepath= rd.initial_setting_basicshape("cube")   #should be already assigned the grid
        #global globalGrid
        #globalGrid = self.grid
        self.startProgressBar()
        self.Update_image();
    
    def OpenCylinder(self):
        self.grid, self.imagefilepath= rd.initial_setting_basicshape("cylinder")
        #global globalGrid
        #globalGrid = self.grid
        self.startProgressBar()
        self.Update_image();

    def OpenGridFile(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Obtain Grid", "", "All Files(*.*);;PNG(*.png);;JPEG(*.jpg *.jpeg)")
        self.grid, self.imagefilepath = rd.initial_setting_filepath(filePath)
        #global globalGrid
        #globalGrid = self.grid
        self.startProgressBar()
        self.Update_image();

    def startProgressBar(self):
        self.thread = MyThread()
        self.thread.change_value.connect(self.setProgressVal)
        self.thread.start()

    def setProgressVal(self, val):
        self.ui.progressBar.setValue(val)
    
    def Update_image(self):  # update the canvas
        pixmap =  QtGui.QPixmap(self.imagefilepath)
        self.ui.label.setPixmap(pixmap)
        self.update()
    
# clear to blank
    def clear(self):
        pixmap =  QtGui.QPixmap("multiview/blank.png")
        self.imagefilepath = "multiview/blank.png"
        pixmap2 = pixmap.scaled(256,256)
        self.ui.label.setPixmap(pixmap2)
     
# confirm to change window
    def confirm(self):
        if self.windows2 == None:
            self.windows2 = PaintWindow(self.imagefilepath)
            self.windows2.show()
        else:
            #self.saveFile()
            #print("Inside file mask")
            self.saveFile_Mask()
            #pass camera pos
            camerapos = list(self.pose[0:3])  # for recording in the Painting View
            self.windows2.Add_ViewPose(camerapos)   # use this function to pass current camera position to the painting window (windows2)
            self.windows2.update_grid(self.grid)   #update the grid to the painting window (windows2)
            self.windows2.activateWindow();  # activate the second window (the paiting window)

    
#combing rotation here 
    def getPosPress(self,event):
        if event.buttons()== QtCore.Qt.LeftButton: 
            x = event.pos().x()
            y = event.pos().y()
            self.p1 = self.pointtoVec3d((x,y),256,256)
            #print(MyForm.p1)
            self.isIn =True

    def getPosRelease(self,event):
            self.isIn=False

    def getPosMove(self,event):
        x = event.pos().x()
        y = event.pos().y()
    
        if  self.isIn and (abs(self.last2d[0]-x) >=1 or abs(self.last2d[1]-y>=1)) and self.grid!=None: #avoid zero move
            self.p2 =self.pointtoVec3d((x,y),256,256)
            caxis = np.cross(self.p1-1e-4,self.p2+1e-4)
            dot_product = np.dot(self.p1,self.p2)
            theta= np.arccos(dot_product)*180.0/math.pi
            delta= Quaternion(axis=caxis,degrees=-theta) #flip theta?
            q = self.originquaternion
            q = q*delta
            self.originquaternion = q
            self.pose = q.rotate(self.perturbvec)
            rd.update_image(list(self.pose[0:3]),self.grid,"opened")
            #global globalGrid
            #globalGrid = self.grid
            self.Update_image()
            self.p1 = self.p2
        self.last2d[0] = x
        self.last2d[1] = y

    def RotationMatrix(self,axis,angle):
        x = axis[0]
        y = axis[1]
        z = axis[2]
        c = math.cos(angle)
        s = math.sin(angle)
        rot = np.zeros((4,4))
        rot[3,3]=1.0
        rot[0,0]=x*x+c*(1-x*x)
        rot[0,1]=x*y*(1-c)-z*s
        rot[0,2]=x*z*(1-c)+y*s
        rot[1,0]=x*y*(1-c)+z*s
        rot[1,1]=y*y+c*(1-y*y)
        rot[1,2]=x*z*(1-c)-x*s
        rot[2,0]=x*z*(1-c)-y*s
        rot[2,1]=y*z*(1-c)+x*s
        rot[2,2]= z*z+c*(1-z*z)
        #return the inverse matrix of rot
        rot_inv = np.linalg.inv(rot) 
        return rot_inv
    
    def normalize_axis(self,axis):
        newaxis = axis/np.linalg.norm(axis)
        return newaxis
    
    def pointtoVec3d(self,point,width,height):
        scalefactor = min(width,height)
        x = point[0] / (scalefactor/2.0)
        y = point[1] / (scalefactor/2.0)
        x = x-1.0
        y = 1.0-y
        ztmp = 1.0-x*x-y*y
        z = 0.0
        if ztmp>0:
            z = math.sqrt(ztmp)       
        projectedpoint = np.array([x,y,z])
        projectedpoint /= LA.norm(projectedpoint,2)
        return projectedpoint


if __name__=="__main__":
    app = QApplication(sys.argv)
    grid_new = None  #Fistly as None for initialization
    qw = AppWindow(grid_new) 
    w = PaintWindow(None, qw)
    w.move(600,300)
    qw.move(1200,300)  #adjust positions of two windows
    w.show()
    qw.addWindow(w)
    w.addWindow(qw)  # add each other
    qw.show()
    sys.exit(app.exec_())
