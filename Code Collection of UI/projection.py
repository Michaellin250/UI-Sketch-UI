import torchvision
import numpy as np
import torch
import math
import sys, os
import sketch as sk # for generate sketch from the shape
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint,QRect
from PyQt5.QtGui import  QIcon, QImage, QPainter, QPen, QBrush, QPixmap, QColor
from PyQt5.QtWidgets import QMainWindow, QApplication, QMenu, QMenuBar, QAction, QFileDialog, QLabel,QButtonGroup, QWidget, QListWidgetItem
from NavigationView3D import *
import cv2
import time
import qimage2ndarray
from numpy import linalg as LA
import rendererpyw as rd
from pyquaternion import Quaternion

class MyThread(QThread):
    # Create a counter thread
    change_value = pyqtSignal(int)
    def run(self):
        cnt = 0
        while cnt <100:
            cnt+=5
            time.sleep(0.3)
            self.change_value.emit(cnt)

#Duotun's implementations
def normalized(v):
    norm = LA.norm(v)
    if norm >0:
        return v/norm;
    else: 
        return v;

class ModelTransform:
    def __init__(self):    #underscore means protected data
        self._scale = np.full(3,2);
        self._position = np.zeros(3)
        self._rotation = Quaternion(1,0,0,0)
        self._matrix = None
        self._translation_matrix = None
        self._rotation_matrix = None
        self._matrix = None

    @property
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self, value):
        self._scale = np.asarray(value)
        self._matrix, self_translation_matrix = None, None
    
    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, value):
        self._position = np.asarray(value);
        self._matrix, self._translation_matrix = None, None
    
    @property
    def rotation(self):
        return self._rotation
    
    @rotation.setter
    def rotation(self, value):
        self._rotation = Quaternion(value);
        self.matrix, self._rotation_matrix = None, None
    
    @property
    def matrix(self):
        if self._matrix is None:
            if self._translation_matrix is None:
                sx, sy, sz = self._scale
                x, y, z = self._position
                self._translation_matrix = np.array((
                    (sx,  0,  0, x),
                    ( 0, sy,  0, y),
                    ( 0,  0, sz, z),
                    ( 0,  0,  0, 1)))

                if self._rotation_matrix is None:
                    self._rotation_matrix = np.identity(4)
                    self._rotation_matrix = self.rotation.transformation_matrix
                
                R = self._rotation_matrix
                T = self._translation_matrix
                self._matrix = R@T
        return self._matrix


class ViewTransform:
    def __init__(self):
        self._eye = np.ones(3)    #camera position
        self._translationmatrix = np.identity(4);
        neye = np.negative(self.eye)
        self._translationmatrix[:3,3] = neye;   #define the translationmatrix accordingly
        self.look_at(np.array((0,0,0)),np.array((0,1,0)))  #look at zero and the up vector is (0, 1, 0)
    
    @property
    def eye(self):
        if self._eye is None:
            self._eye = LA.inv(np.negative(self.translationmatrix))[:3,3];
        return self._eye

    @eye.setter
    def eye(self, position):
        self._eye = np.asarray(position);
        _ = self.rotation
        _ = self.translationmatrix  #make sure the rotation is saved
        self._matrix = None  

    @property
    def rotation(self):
        if self._rotation is None:
            self._rotation = Quaternion(matrix = self.matrix[:3,:3])
        return self._rotation
    @rotation.setter
    def rotation(self,value):
        self._rotation = Quaternion(value)   # from numpy array
        _ = self.eye
        self._matrix = None

    @property
    def translationmatrix(self):
        if self._translationmatrix is None:
            self._translationmatrix = np.identity(4);
            self._translationmatrix[:3,3] = self.matrix[:3,3]
        return self._translationmatrix
    
    @translationmatrix.setter 
    def translationmatrix(self,position):
        self._translationmatrix = np.identity(4);
        self._eye = np.asarray(position);
        self._translationmatrix[:3,3] = np.negative(self.eye);

    @property
    def matrix(self):
        if self._matrix is None:
            R = self.rotation.transformation_matrix
            T = self.translationmatrix
            self._matrix = R@T
        return self._matrix
    
    @matrix.setter 
    def matrix(self,value):
        self._matrix = value
        self._eye = None
        self._rotation = None
    
    def look_at(self,target,up):
        zax = normalized(self.eye - np.asarray(target))
        xax = normalized(np.cross(np.asarray(up),zax))
        yax = np.cross(zax,xax)
        self.rotation = Quaternion(matrix = np.stack((xax,yax,zax),axis=0))   #3*3 here

class PerspectiveTransform:
    def __init__(self):
        self._aspect = 2
        self._fov = 40*np.pi /180
        self._near = 0.1
        self._far = 1.0
        self._matrix = None
    
    @property
    def aspect(self):
        return self._aspect

    @aspect.setter
    def aspect(self, value):
        self._aspect = value
        self._matrix = None

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        self._fov = value
        self._matrix = None

    @property
    def near(self):
        return self._near

    @near.setter
    def near(self, value):
        self._near = value
        self._matrix = None

    @property
    def far(self):
        return self._far

    @far.setter
    def far(self, value):
        self._far = value
        self._matrix = None

    @property
    def matrix(self):
        if self._matrix is None:
            t = np.tan(self._fov / 2) * self._near
            b = - t
            assert abs(t - b) > 0
            l, r = b * self._aspect, t * self._aspect
            assert abs(r - l) > 0
            n, f = self._near, self._far
            assert abs(n - f) > 0
            self._matrix = np.array((
                (  2*n/(r-l),           0,            (r+l)/(r-l),  0),
                (          0,   2*n/(t-b),             (t+b)/(t-b),  0),
                (0, 0,   (f+n)/(n-f), 2*(f*n)/(n-f)),
                (          0,           0, -1,  0)))
        return self._matrix

class MVPTransform:
    def __init__(self):
        self.model = ModelTransform()
        self.view = ViewTransform()
        self.projection = PerspectiveTransform()
    
    def matrix(self):
        print("Printing projection Matrix")
        print(self.projection.matrix)
        print("Printing View Matrix")
        print(self.view.matrix)
        print("Printing Model Matrix")
        print(self.model.matrix)
        m = self.projection.matrix @ self.view.matrix @ self.model.matrix
        return m.astype(np.float32)

class MVPxy(MVPTransform):
    def __init__(self):
        super().__init__()
        self.name = 'MVP'
        self.projection.aspect = 2  # 256x256
        self.projection.far = 1.0
        self.view.eye = np.array((5,0,0))

def projectXY(grid, imgWidth, imgHeight, cameraPos):
    binaryRep = convertToBinaryGrid(grid)
    #print(binaryRep[32,32,32]);
    img1 = torch.zeros([imgWidth, imgHeight])
    Mvp = MVPxy()
    mvpmatrix = Mvp.matrix();
    print("Mvp Matrix: ")
    print(mvpmatrix)
    allSides = [[-1, 0], [1, 0], [0,0], [0, -1], [0,1]]
    radius = 10
    #print(M)
    #M = np.dot(Mvp, Morth)
    for i in range(64):
        for j in range(64):
            for k in range(64):
                if binaryRep[i,j,k]:
                    #print("i, j, k: ",i,j,k); 
                    coords = np.reshape([i, j, k, 1], (4, 1))
                    #print(M)
                    #print(coords)
                    p = np.multiply(mvpmatrix, coords)
                    #print(p)
                    for side in allSides:
                        for eachRadius in range(1, radius + 1):
                            xPixel = int(p[0,0]) + side[0] * eachRadius        # x is y and y is x
                            yPixel = int(p[1, 0]) + side[1] * eachRadius
                            #print(xPixel, yPixel);   #ignore z and w -> depth   
                            img1[(math.ceil(yPixel + math.ceil((imgHeight / 1.5))) % imgHeight), (math.floor(xPixel + math.floor((imgWidth / 4))) % imgWidth)] = 1   
    torchvision.utils.save_image(img1, "./" + "projection.png")

def convertToBinaryGrid(grid):
    binaryGrid = grid.clone()
    binaryGrid[grid > 0] = 0
    binaryGrid[grid <= 0] = 1
    #print("Binary Grid is")
    #print(binaryGrid)
    return binaryGrid


def printGrid(grid):
    print("Printing every voxel")
    for i in range(64):
        for j in range(64):
            for k in range(64):
                print(grid[i,j,k])

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
        if not filePath:
            return 
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
            #self.saveFile()
            #print("Inside file mask")
            #self.saveFile_Mask()
            #pass camera pos
            camerapos = list(self.pose[0:3])  # for recording in the Painting View
            #call the projection code
            projectXY(self.grid, 256, 256, list(self.pose[0:3]))
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
    qw.show()
    sys.exit(app.exec_())
