import numpy as np

def anisodiff(img, niter=1, kappa=50, gamma=0.1, sigma=0):
    
    # Anisotropic diffusion denoising
    # Based on the python implementation by Alistair Muldal <alistair.muldal@pharm.ox.ac.uk>
    # Kappa controls conduction gradient, Gamma controls diffusion speed

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize deltas
    deltaS = np.zeros(imgout.shape)
    deltaE = np.zeros(imgout.shape)
    NS = np.zeros(imgout.shape)
    EW = np.zeros(imgout.shape)

    # initialize gradients
    gS = np.ones(imgout.shape)
    gE = np.ones(imgout.shape)

    for ii in np.arange(1,niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)
        deltaSf=deltaS
        deltaEf=deltaE
        
        # update matrices with the conduction gradient and the delta
        E = np.exp(-(deltaEf/kappa)**2.)*deltaE
        S = np.exp(-(deltaSf/kappa)**2.)*deltaS
        
        # subtract a copy that has been shifted 'North/West' by
        # 1 pixel (this needs to be here???)
        # according to Alistair, "Just don't ask"
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]
        # update the image
        imgout += gamma*(NS+EW)

    return imgout

def filter_image(im, filter):
    im = np.pad(im, 1, mode='reflect') # reflect better preserves derivatives
    s = filter.shape + tuple(np.subtract(im.shape, filter.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(im, shape = s, strides = im.strides * 2)
    return np.einsum('ij,ijkl->kl', filter, subM)

def denoise_img(img):
    img_out = np.zeros(img.shape)
    for x in range(img.shape[2]):
        img_out[:,:,x] = anisodiff(img[:,:,x],niter=50,kappa=80,gamma=0.025)
        # sharpen output
        kernel = np.array([[-0.5,-1,-0.5], [-1,7,-1], [-0.5,-1,-0.5]])
        img_out[:,:,x] = filter_image(img_out[:,:,x], kernel)
    
    return img_out
