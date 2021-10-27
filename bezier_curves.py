import numpy as np
import matplotlib.pyplot as plt

P0, P1, P2 = np.array([
	[1, 0],
	[2.5, 1],
	[5, 0]
])

def plot(int x0, int y0, int x1, int y1, int x2, int y2):
    sx = x2 -x1
    sy = y2-y1
    xx =x0-x1
    yy = y0-y1
    #long xy double dx dy err  
    cur =xx*sy-yy*sx
    #assert(xx*sx <= 0 && yy*sy <= 0);
    if (sx*sx+sy*sy > xx*xx+yy*yy)
        x2 = x0
        x0 = sx+x1
        y2 = y0
        y0 = sy+y1
        cur = -cur

    if (cur != 0)
        xx += sx 
        #xx *= sx = x0 < x2 ? 1 : -1;         /* x step direction */
        xx *= sx = 1 if x0 < x2 else -1
        yy += sy
        #yy *= sy = y0 < y2 ? 1 : -1;           /* y step direction */
        yy *= sy = 1 if y0 < y2 else -1
        xy = 2*xx*yy
        xx *= xx
        yy *= yy
        if (cur*sx*sy < 0)
            xx = -xx
            yy = -yy
            xy = -xy
            cur = -cur
        dx = 4.0*sy*cur*(x1-x0)+xx-xy
        dy = 4.0*sx*cur*(y0-y1)+yy-xy
        xx += xx
        yy += yy 
        err = dx+dy+xy
        

# define bezier curve
P = lambda t: (1 - t)**2 * P0 + 2 * t * (1 - t) * P1 + t**2 * P2

# evaluate the curve on [0, 1] sliced in 50 points
points = np.array([P(t) for t in np.linspace(0, 5, 20)])

print(points)
# get x and y coordinates of points separately
x, y = points[:,0], points[:,1]

# plot
'''
plt.plot(x, y, 'b-')
plt.plot(*P0, 'r.')
plt.plot(*P1, 'r.')
plt.plot(*P2, 'r.')
plt.text(*P0, "P0")
plt.text(*P1, "P1")
plt.text(*P2, "P2")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()
'''