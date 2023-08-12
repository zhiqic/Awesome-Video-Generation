
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from dataclasses import dataclass
from matplotlib.widgets import TextBox
from pathlib import Path
import pickle
import os


def pload(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def pdump(path: str, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_pred(path: str, name: str = ''):

    r = []

    with open(path) as f:

        for line in f.readlines():

            line = line.strip().split(' ')

            assert len(line) >= 2
            assert len(line) % 2 == 0
            if (len(line) - 2) / 2 != int(line[1]):
                print(line[0])
                print(int(line[1]))
                print((len(line) - 2) / 2)
            assert (len(line) - 2) / 2 == int(line[1])

            filename = line[0]
            count = int(line[1])
            points = [int(i) for i in line[2:]]

            if count > 0:
                points = np.array(points).reshape(-1, 2)
            r.append({'filename': filename, 'count': count, 'points': points})
    
    if name == '':
        return r
    
    for i in r:
        if i['filename'] == name:
            return i

    raise KeyError(f'Entry with filename {filename} not found!')


@dataclass
class Marker():
    label: int
    x: int
    y: int
    mpl_refs: list


class GetMarkers:

    def __init__(self, img: Image.Image, result_path: str = None):
        # Desaturate image to allow markers to stand out
        img = np.array(ImageEnhance.Color(img).enhance(0.3))

        self.dpi = 100
        self.result_path: str = result_path
        self.img: np.ndarray = img
        self.cur_label: int = 1
        self.markers: list[Marker] = []

        self.fig = plt.figure(result_path.split('/')[-1])
        self.ax = self.fig.add_subplot(111)
        #self.fig, self.ax = plt.subplots(1)


    def update_gui(self):
        self.ax.set_title(f'Current Label: {self.cur_label}')
        self.fig.canvas.draw()


    def place_marker(self, label, x, y):
        refs = [
            self.ax.plot(x, y, marker='+',
                color='red', markersize=5)[0],
            self.ax.annotate(label, (x+10, y-5),
                color='red', fontsize=8),
        ]
        m = Marker(label, int(x), int(y), refs)
        self.markers.append(m)


    def del_marker(self, x, y):
        xl, xh = self.ax.get_xlim()
        radius = 0.02 * abs(xl - xh)

        for m in self.markers.copy():
            if np.linalg.norm((x - m.x, y - m.y)) < radius:
                for i in m.mpl_refs:
                    i.remove()
                self.cur_label = m.label
                self.markers.remove(m)
                break


    def run(self) -> dict[str, tuple[int,int]]:
        self.ax.imshow(self.img)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_up)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_up)

        if self.result_path is not None:
            if os.path.isfile(self.result_path + '.mks'):
                for (l, (x,y)) in pload(self.result_path + '.mks').items():
                    self.place_marker(l, x, y)

        #self.box = TextBox(self.fig.add_axes([0.1, 0.90, 0.2, 0.085]), "Label: ")
        #self.box.on_submit(lambda e : (
        #    setattr(self, 'cur_label', int(e)) if e.isdecimal() else None,
        #    self.box.set_val(''),
        #    self.update_gui(),
        #))

        self.update_gui()
        plt.show()

        self.fig.set_size_inches(self.img.shape[1] / 100, self.img.shape[0] / 100)
        self.fig.set_dpi(100)
        plt.figure(
            self.fig,
            figsize=(self.img.shape[1] / 100, self.img.shape[0] / 100),
            dpi=100,
        )
        self.ax.set_title('')
        self.ax.set_xlim((0, self.img.shape[1]))
        self.ax.set_ylim((self.img.shape[0], 0))
        self.ax.axis('off')
        self.fig.canvas.draw()

        plt.savefig(
            self.result_path + '-markers.png',
            bbox_inches='tight',
            pad_inches=0,
            dpi=100,
        )

        r = {}
        for m in self.markers:
            r[m.label] = (m.x, m.y)

        if self.result_path is not None:
            pdump(self.result_path + '.mks', r)

        return r

    def on_mouse_up(self, e):
        # About `fig.canvas.manager.toolbar.mode`
        # https://stackoverflow.com/a/63447351/5702494

        if self.fig.canvas.manager.toolbar.mode != '':
            return

        x, y = e.xdata, e.ydata

        # Place marker
        if e.inaxes is self.ax and e.button == 1:
            self.place_marker(self.cur_label, x, y)
            self.cur_label += 1

        # Delete marker upon right click
        elif e.button == 3:
            self.del_marker(x, y)

        self.update_gui()


    def on_key_up(self, e):
        if e.key in 'iI':
            self.cur_label += 1 if e.key == 'i' else 10
        elif e.key in 'dD':
            self.cur_label -= 1 if e.key == 'd' else 10
            self.cur_label = max(1, self.cur_label)

        self.update_gui()

