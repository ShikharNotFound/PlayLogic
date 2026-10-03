import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def draw_field(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7))
    ax.set_facecolor('#2e8b57')
    ax.set_xlim(0,105); ax.set_ylim(0,68)
    ax.set_aspect('equal'); ax.axis('off')
    ax.plot([0,105,105,0,0],[0,0,68,68,0],'white',lw=2)
    ax.plot([52.5,52.5],[0,68],'white',lw=2)
    ax.add_patch(Circle((52.5,34),9.15,ec='white',fc='none',lw=2))
    for x in [0,105]:
        ax.plot([x,x],[30.34,37.66],'white',lw=5)
    return ax