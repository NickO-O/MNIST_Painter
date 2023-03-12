import sys
import numpy as np
from math import dist
import tensorflow as tf
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi
from keras.optimizers import Adam




class Pain(QDialog):
    def __init__(self):
        super(Pain, self).__init__()
        loadUi('er.ui', self)
        self.field = np.zeros((28, 28))
        self.move = False
        self.rast = 0
        #QComboBox.currentTextChanged.connect()
        self.yes = True
        self.model = tf.keras.models.load_model(f'{self.comboBox.currentText()}.h5')
        self.comboBox.currentTextChanged.connect(self.changeModel)
        self.last_coords = None
        #print(self.label.size().width())
        self.field_image = QImage(self.label.size().width(), self.label.size().height(), QImage.Format_ARGB32_Premultiplied)
        #self.initUI()

        self.pushButton.clicked.connect(self.Neural)
        self.repaintField()

    def repaintField(self):
        painter = QPainter()
        painter.begin(self.field_image)
        #painter.setBrush(QColor(255, 255, 255))
        for i in range(self.field.shape[0]):
            for j in range(self.field.shape[1]):
                er = QColor(*[round(255 * (1 - self.field[i, j])) for _ in range(3)])
                painter.setBrush(er)
                painter.drawRect(QRect(i * 20, j * 20, i * 20 + 20, j * 20 + 20))
        painter.end()
        self.label.setPixmap(QPixmap(self.field_image))
    
    def mouseMoveEvent(self, event):
        #funRast = lambda q, w: q * 20 + 10, w * 20 + 10
        if event.buttons() == Qt.LeftButton:
            ind = 1
        else:
            ind = - 1
        rast = self.horizontalSlider.value()
        x, y = event.x() / 20, event.y() / 20
        if self.last_coords is None:
            self.last_coords = (x, y)
        else:
            if self.move and self.rast < 3:
                return
            else:
                if self.rast > 3:
                    self.rast = 0
            self.rast += dist(self.last_coords, (x, y))
        #print(self.rast)
        if x > 28:
            return
        for i in range(self.field.shape[0]):
            for j in range(self.field.shape[1]):
                q, w = (i * 20 + 10) / 20, (j * 20 + 10) / 20
                sqt = ((q - x) ** 2 + (w - y) ** 2) ** 0.5
                #print(sqt)
                if sqt < rast + 1:
                    if sqt < rast - 1:
                        self.field[i][j] += ind
                    else:
                        #print(rast - sqt + 1)
                        self.field[i][j] += (max(1 - (sqt - rast + 1), 0)) * ind
                self.field[i][j] = max(min(self.field[i][j], 1), 0)
        self.repaintField()
        #self.Neural()

    def mousePressEvent(self, event):
        self.move = False
        self.mouseMoveEvent(event)

    def Neural(self):
        #self.field = np.random.random((28, 28))
        #self.repaintField()
        pred = self.model.predict(np.array([self.field.transpose()]), callbacks=None)
        #print(pred, y_test_one_hot[0:1])
        e = (((pred - pred.min()) / (pred - pred.min()).sum()))[0]
        s = [self.p0, self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8, self.p9]
        #print(e)
        for i in range(10):
            s[i].setValue(int(e[i] * 100))
        if e.max() > 0.4:
            self.yes = False

    def changeModel(self):
        self.model = tf.keras.models.load_model(f'{self.comboBox.currentText()}.h5')



def main():
    app = QApplication(sys.argv)
    ex = Pain()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()