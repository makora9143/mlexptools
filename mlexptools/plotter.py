from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams['image.cmap'] = 'jet'


color_list = [
        '#005AFF',
        '#FF4B00',
        '#03AF7A',
        '#4DC4FF',
        '#F6AA00',
        '#FFF100',
        '#990099',
    ]

cmap = ListedColormap(
    color = color_list,
    name='universal',
)


# class Plotter:
#     def __init__(self, dpi=300):
#         self.ratio = 1.414
#         self.dpi = 300
#         self.figsize = (self.ratio * 1, 1)

#     def plot(self, *data, *, dpi=300,  **kwargs):
#         logger.info(f"Plotting with {self.dpi * self.figsize[0]} x {self.dpi * self.figsize[1]}")
#         fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
#         ax = fig.add_subplot(111)
