import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import time
from astropy.modeling import models


beta = 4.765
slit_corr = []
seeings = np.arange(0.5, 3, 0.01)
slit_width = 0.9 # arcsec




# Generate image parameters
img_size = 100


arcsec_to_pix = 100/5 # Assumes a 5 arcsec image
slit_width_pix = arcsec_to_pix * 0.9
seeing_pix = arcsec_to_pix * seeings

x, y = np.mgrid[:img_size, :img_size]
source_pos = [int(img_size/2), int(img_size/2)]

fig = pl.figure()
ims = []
for pp in seeing_pix:



    # Simulate source Moffat
    gamma = pp / (2 * np.sqrt(2**(1/beta) - 1))
    source = models.Moffat2D.evaluate(x, y, 1, source_pos[0], source_pos[1], gamma, beta)





    # mask = np.tile(mask, (len(y2), 1))
    # print(np.shape(source))
    mask = slice(int(source_pos[1] - slit_width_pix/2),int(source_pos[1] + slit_width_pix / 2))
    # im = pl.imshow(source[:, mask], cmap="viridis", interpolation='none')
    sl = np.trapz(np.trapz(source)) / np.trapz(np.trapz(source[:, mask]))
    # pl.show()
    # exit()
    # ims.append([im])
    slit_corr.append(sl)
exit()
import matplotlib.animation as animation
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
pl.show()
# Saving to .dat file
dt = [("seeings", np.float64), ("slit_corr", np.float64)]
data = np.array(list(zip(seeings, slit_corr)), dtype=dt)
np.savetxt("Slit_loss_calculation_%s.dat"%slit_width, data, header="seeing/[FWHM] slitloss_correction", fmt = ['%1.2f', '%1.3f'])