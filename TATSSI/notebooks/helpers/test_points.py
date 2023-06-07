import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9.0, 9.0))

# Left plot
left_p = plt.subplot2grid((2, 2), (0, 0), colspan=1)
left_p.set_xlim([0, 10])

# Right plot
right_p = plt.subplot2grid((2, 2), (0, 1), colspan=1,
        sharex=left_p, sharey=left_p)
right_p.set_ylim([0, 10])

# Time series plot
ts_p = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ts_p.plot(np.arange(10))

def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))

    if len(left_p.lines) > 0:
        del left_p.lines[0]
        del right_p.lines[0]

    left_p.plot(event.xdata, event.ydata, 'ro')
    right_p.plot(event.xdata, event.ydata, 'ro')

    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
