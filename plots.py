import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gp_functions as gpf


def plot_loss_evolution(figx, figy,lls):
    '''
    2D plot of log-likelihood in each iteration
    '''
    plt.figure(figsize=(figx, figy))
    for j in range(lls.shape[1]):
        plt.plot(lls[:, j])
    # plt.plot(lls[:, 1])
    plt.xlabel("Training iteration")
    plt.ylabel("Log marginal likelihood")
    plt.show()


def plot_sin3D_rand_points(figx, figy, coord_range, obs_index_pts, obs, number_of_points):
    fig = plt.figure(figsize=(figx, figy))
    ax = fig.gca(projection='3d')
    stepx = (coord_range[0][1] - coord_range[0][0])/number_of_points # coord_range[0] = 'xrange', coord_range[1] = 'yrange'
    x = np.arange(coord_range[0][0], coord_range[0][1]+stepx, stepx)
    stepy = (coord_range[1][1] - coord_range[1][0])/number_of_points
    y = np.arange(coord_range[1][0], coord_range[1][1]+stepy, stepy)
    X, Y = np.meshgrid(x, y, sparse=False)
    # Z = np.sin((X)) * np.sin((Y))
    Z = np.zeros(X.shape)
    for j in range(Z.shape[1]):
        for i in range(Z.shape[0]):
            Z[i,j] = gpf.sinusoid(np.array([[X[0,j],Y[i,0]]])) # draw above x,y, 1,2 shape
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False, alpha=0.35)
    # ax.scatter(xyz_pts[:,0],xyz_pts[:,1],xyz_pts[:,2])
    ax.scatter(obs_index_pts[0, :, 0],
               obs_index_pts[0, :, 1],
               obs, s=40 )
    plt.show()

def plot_marginal_likelihood3D(xedges, H, figx=12, figy=12):
    '''
    3d plot of log marginal likelihood
    '''
    fig2 = plt.figure(figsize=(figx, figy))
    ax2 = fig2.gca(projection='3d')

    # start2, stop2, step2 = np.min(r2), np.max(r2), (np.max(r2)-np.min(r2))/20
    start2, stop2, step2 = 1/xedges, 1, 1/xedges
    xedges2 = np.arange(start2, stop2+step2, step2)
    yedges2 = np.arange(start2, stop2+step2, step2)

    # Make data.
    X2 = np.linspace(start2, stop2, len(xedges2))
    Y2 = np.linspace(start2, stop2, len(yedges2))
    X2, Y2 = np.meshgrid(X2, Y2, sparse=False)    # 20x20

    # Plot the surface.
    H2 = (H-np.min(H))/(np.max(H)-np.min(H))
    # print(np.max(H2), np.min(H2))
    # print(X2.shape, Y2.shape, H2.shape)
    ax2.plot_surface(X2, Y2, H2, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False, alpha=0.7)
    # Add a color bar whic(H2-np.min(H2))/(np.max(H2)-np.min(H2))h2.colorbar(surf2, shrink=0.5, aspect=5)
    ax2.set_xlabel('length_scale_assign')
    ax2.set_ylabel('amplitude_assign')
    ax2.set_zlabel('Z Label')
    plt.show()


def plot_samples2D(fx, fy, axis_k, pred_idx_pts, obs_idx_pts, obs, samples_, num_samples):
    plt.figure(figsize=(fx, fy))
    assert len(obs_idx_pts.shape) == len(pred_idx_pts.shape)

    fn_val = gpf.sinusoid(pred_idx_pts)

    if(1 < len(pred_idx_pts.shape)):
        plt.plot(pred_idx_pts[:, axis_k], fn_val, label='True fn =sinusoid(pred_idx_pts)')
        plt.scatter(obs_idx_pts[0, :, axis_k], obs[0, :], label='Observations', s=40)
        for i in range(num_samples):
            plt.plot(pred_idx_pts[0, :, axis_k], samples_[i, 0, :].T, c='r', alpha=.1,
                     label='Posterior Sample' if i == 0 else None)
    else:
        assert 1 == len(pred_idx_pts.shape)
        plt.plot(pred_idx_pts[:], fn_val, label='True fn =sinusoid(pred_idx_pts)')
        plt.scatter(obs_idx_pts[:], obs, label='Observations', s=40)
        for i in range(num_samples):
            plt.plot(pred_idx_pts[:], samples_[i, 0, :].T, c='r', alpha=.1,
                     label='Posterior Sample' if i == 0 else None)

    leg = plt.legend(loc='upper right')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.xlabel(r"Index points ($\mathbb{R}^1$)")
    plt.ylabel("Observation space")
    plt.show()


def plot_samples3D(figx,figy, axis_j ,pred_index, obs_index_pts, obs, pred_samples, PRED_FRACTION, u, v):
    fig_ = plt.figure(figsize=(figx, figy))
    assert len(obs_index_pts.shape) >= 2 and "need surface plot here"
    ax = fig_.gca(projection='3d')

    max_x = np.max(pred_index[0, :, 0]) #define range
    max_y = np.max(pred_index[0, :, 1])
    min_x = np.min(pred_index[0, :, 0])
    min_y = np.min(pred_index[0, :, 1])
    step_x = (max_x - min_x) / PRED_FRACTION
    step_y = (max_y - min_y) / PRED_FRACTION

    # stepx = (np.max(samples[:,0,:][1] - coord_range[0][0]) / number_of_points  # coord_range[0] = 'xrange', coord_range[1] = 'yrange'
    x = np.arange(min_x, max_x, step_x) # from to is excluding max_x to value -> we want to include max_x so +step_x
    # stepy = (coord_range[1][1] - coord_range[1][0]) / number_of_points
    y = np.arange(min_y, max_y, step_y)
    X, Y = np.meshgrid(x, y, sparse=False)
    # Z = np.zeros(X.shape)
    Z = pred_samples[0, 0, :]
    # Z = np.swapaxes(Z, 0,1)
    # assert Z.shape[2] == 2
    # Z = Z[:,0]
    Z = Z.reshape(X.shape[0],-1)
    # Z = np.reshape(0,1)
    # print("max_x: ", max_x.shape)
    # print("y: ", y.shape)
    # print("x: ", x.shape)
    # print("y: ", y.shape)
    # print("X: ",X.shape)
    # print("Y: ",Y.shape)
    # print("Z: ",Z.shape)
    plt.plot([u[0],v[0]], [u[1],v[1]],ls="--", c=".4") # section line

    ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False, alpha=0.4)
    ax.scatter(obs_index_pts[0, :, 0],
               obs_index_pts[0, :, 1],
               obs, s=40)    # ax.plot(predictive_index_points_[:, axis_j], sinusoid(predictive_index_points_),label='True fn')
    # ax.scatter(observation_index_points_[:, axis_j], observations_,label='Observations')
    # for i in range(num_samples):
    #     ax.plot(predictive_index_points_[:, axis_j], samples_[i, axis_j, :].T, c='r', alpha=.1,
    #              label='Posterior Sample' if i == 0 else None)
    # leg = ax.legend(loc='upper right')
    # for lh in leg.legendHandles:
    #     lh.set_alpha(1)
    # ax.xlabel(r"Index points ($\mathbb{R}^1$)")
    # ax.ylabel("Observation space")
    plt.show()


def plot_2d_observations(figx, figy, pts, sel_pts, u, v):
    '''
    nd input, plot 0th and 1th col
    epsilon pts colour # sin true curve, GP and confidence intervals
    line on XY plane and all points in space
    '''
    plt.figure(figsize=(figx,figy))
    pts=np.array(pts)
    sel_pts=np.array(sel_pts)
    plt.scatter(pts[:,0], pts[:,1], c='purple')
    plt.scatter(sel_pts[:,0], sel_pts[:,1], c='red')
    plt.scatter(u[0], u[1])
    plt.scatter(v[0], v[1])
    plt.plot([u[0],v[0]], [u[1],v[1]], ls="--", c=".4")
    plt.show()


def plot_capture_line(figx, figy, sel_pts, u, v):
    '''
    plot section line, old coordinates and their projection
    '''
    plt.figure(figsize=(figx, figy))
    sel_pts=np.array(sel_pts)
    projsel_pts = gpf.project_2d_to_line_coordinates(sel_pts,u,v) # project to line 2D output
    plt.scatter(sel_pts[:,0], sel_pts[:,1], color='blue') # original coordinates
    plt.scatter(projsel_pts[:,0], projsel_pts[:,1], color='orange') # observations
    plt.plot([u[0],v[0]], [u[1],v[1]], ls="--", c=".4") # section line
    plt.show()


def plot_capture_xy_line(figx, figy, sel_pts, u, v): # 2D
    '''
    plot section line, old coordinates and their projection
    '''
    plt.figure(figsize=(figx, figy))
    a = u[0:2]
    b = v[0:2]
    projsel_pts = gpf.project_to_line_coordinates(sel_pts,a,b) # project to line
    plt.scatter(sel_pts[:,0], sel_pts[:,1], color='blue') # original coordinates
    plt.scatter(projsel_pts[:,0], projsel_pts[:,1], color='orange') # observations
    plt.plot([a[0],b[0]], [a[1],b[1]], ls="--", c=".4") # section line
    plt.show()



def plot_section_observations(figx, figy, eppts, u, v):
    '''
    eppts : projected coordinates, extended with a 4th column
        to represent their distance from the section start point
    XY linspace along section line
    Z corresponding sinusoid value
    L corresponding distance along the line
    '''
    plt.figure(figsize=(figx,figy))
    line_XYZ = np.linspace(u, v, num=10000) #(num, 3) 3rd dim const.
    line_XY = gpf.del_last_dimension(line_XYZ) # same as gpf.project_to_line_coordinates here
    line_L = gpf.create_1d_w_line_coords(line_XY, u, v) # 2D input, 1D output
    Z = gpf.sinusoid(line_XY).reshape(-1, 1) # 2D input 1D output
    line_XYZ_ = gpf.add_dimension(line_XY, Z) # 2D input, 3D output
    line_XYZL = gpf.add_dimension(line_XYZ_, line_L) # 3D input, 4D output

    plt.scatter(eppts[:, 3], eppts[:, 2], color='orange')  # observations
    plt.scatter(line_XYZL[:,3], line_XYZL[:,2], color='red', s=0.1)  # sin cut
    plt.show()


def plot_sin_and_sel_pts_at_section(x_new, pts, x_original):
    plt.figure(figsize=(12, 4))
    plt.scatter(x_new[:,0], x_new[:,1], color='orange')
    y = np.array(gpf.sinusoid(x_new))
    plt.scatter(pts[:,0], y,label='True fn')
    x_ = np.linspace(0, 60, 500)
    sin_y= np.sin(x_)
    plt.plot(x_, sin_y)
    # plt.scatter(observation_index_points_[:, axis_k], observations_,label='Observations')
    # for i in range(num_samples):
    #     plt.plot(predictive_index_points_[:, axis_k], samples_[i, axis_k, :].T, c='r', alpha=.1,
    #              label='Posterior Sample' if i == 0 else None)
    plt.show()


def plot_gp_2D_samples(figx, figy, pred_1D_pts, sel_obs, line_idx, line_obs, num_samples, samples_xy, samples_d , samples_z): # Plot the true function, observations, and posterior samples.
    '''

    '''
    plt.figure(figsize=(figx, figy))
    plt.plot(line_idx, line_obs)
    plt.scatter(pred_1D_pts, sel_obs, label='Observations')

    print("samples_z : ",samples_z)
    print("samples_d : ",samples_d)
    print("---------")

    plt.plot(samples_d, samples_z, c='r', alpha=.1,label='Posterior Sample') # line_idx : (500,1), samples_ : (20,2,81) 20 samples, 2D, 81
    leg = plt.legend(loc='upper right')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.xlabel(r"Index points ($\mathbb{R}^1$)")
    plt.ylabel("Observation space")
    plt.show()


def plot2d_sinusoid_samples_section(figx, figy, eppts, line_idx, obs_proj_idx_pts, line_obs, samples_section, num_samples, u, v): # Plot the true function, observations, and posterior samples.

    plt.figure(figsize=(figx, figy))
    line_XYZ = np.linspace(u, v, num=10000) #(num, 3) 3rd dim const.
    line_XY = line_XYZ[:,0:2] # same as gpf.project_to_line_coordinates here
    line_L = gpf.create_1d_w_line_coords(line_XY, u, v) # 2D input, 1D output
    Z = gpf.sinusoid(line_XY).reshape(-1, 1) # 2D input 1D output
    line_XYZ_ = gpf.add_dimension(line_XY, Z) # 2D input, 3D output
    line_XYZL = gpf.add_dimension(line_XYZ_, line_L) # 3D input, 4D output
    predictive_index_points_ = np.linspace(-2,2,200, dtype=np.float64)
    predictive_index_points_ = predictive_index_points_[..., np.newaxis]

    plt.scatter(line_XYZL[:,3], line_XYZL[:,2], color='blue', s=0.2, alpha=0.3)  # sin cut

    # print("samples_section[i,:].T",samples_section[1, :].T )
    # # print("eppts[i , 0:2]",eppts[: , 0:2])
    # print("predictive_index_points_.shape : ", predictive_index_points_.shape)
    # print("samples_section.shape : ", samples_section.shape) # 50,1 200 instead 50,2,3
    # print("--------------")
    # plt.scatter(pred_idx_pts[:, 0], obs,
    #             label='Observations')
    for i in range(num_samples):

        plt.plot(line_idx, samples_section[i, :].T,c='r',alpha=.08,label='Posterior Sample' if i == 0 else None)
    leg = plt.legend(loc='upper right')
    plt.scatter(eppts[:, 3], eppts[:, 2], color='purple', s=40)  # observations
    # for lh in leg.legendHandles:
    #     lh.set_alpha(1)
    plt.xlabel(r"Index points ($\mathbb{R}^1$)")
    plt.ylabel("Observation space")
    plt.show()


def plot2d_samples_section(figx, figy, eppts, line_idx, obs_proj_idx_pts, line_obs, samples_section, num_samples, u, v): # Plot the true function, observations, and posterior samples.

    plt.figure(figsize=(figx, figy))
    line_XYZ = np.linspace(u, v, num=10000) #(num, 3) 3rd dim const.
    line_XY = line_XYZ[:,0:2] # same as gpf.project_to_line_coordinates here
    line_L = gpf.create_1d_w_line_coords(line_XY, u, v) # 2D input, 1D output
    Z = gpf.sinusoid(line_XY).reshape(-1, 1) # 2D input 1D output
    line_XYZ_ = gpf.add_dimension(line_XY, Z) # 2D input, 3D output
    line_XYZL = gpf.add_dimension(line_XYZ_, line_L) # 3D input, 4D output
    predictive_index_points_ = np.linspace(-2,2,200, dtype=np.float64)
    predictive_index_points_ = predictive_index_points_[..., np.newaxis]

    # plt.scatter(line_XYZL[:,3], line_XYZL[:,2], color='blue', s=0.2, alpha=0.3)  # sin cut
    # print("samples_section[i,:].T",samples_section[1, :].T )
    # # print("eppts[i , 0:2]",eppts[: , 0:2])
    # print("predictive_index_points_.shape : ", predictive_index_points_.shape)
    # print("samples_section.shape : ", samples_section.shape) # 50,1 200 instead 50,2,3
    # print("--------------")
    # plt.scatter(pred_idx_pts[:, 0], obs,
    #             label='Observations')
    for i in range(num_samples):
        mu = samples_section[i,0,:]
        s = samples_section[i,1,:]
        lidx = line_idx.flatten()
        plt.gca().fill_between(lidx, mu - s, mu+s, color="#dddddd")

    for i in range(num_samples):
        plt.plot(line_idx, samples_section[i,0,:].T,c='r',alpha=.08,label='Posterior Sample' if i == 0 else None)
        # plt.plt(line_idx, samples_section[i, 0, :].T,c='b',alpha=.04)
        # plt.plt(line_idx, samples_section[i, 1, :].T,c='b',alpha=.04)
    leg = plt.legend(loc='upper right')
    plt.scatter(eppts[:, 3], eppts[:, 2], color='purple', s=40)  # observations
    # plt.ylim(0,0.0075)
    # plt.ylim(-50,50)
    # for lh in leg.legendHandles:
    #     lh.set_alpha(1)
    plt.xlabel(r"Index points ($\mathbb{R}^1$)")
    plt.ylabel("Observation space")
    plt.show()


def plot_kernel(figx, figy, k, u, v):
    plt.figure(figsize=(figx, figy))

    RANGE = [-4, 4]
    LIN_NUM = 250

    X = np.linspace(RANGE[0], RANGE[1], LIN_NUM)[:, None]
    # Y = np.tanh(X[:,0]).reshape(-1,1)
    Y = np.array([[0.]])
    vs_ = [0., 1., 10.]
    vs = np.linspace(0.1, 1, 1000)

    # k = GPy.kern.RBF(1)

    plt.subplot(121)
    K = k.K(X, Y)
    plt.plot(X, K)
    plt.title("x"), plt.ylabel("$\kappa$")
    plt.title("$\kappa_{rbf}(x,0)$")

    plt.subplot(122)
    K = k.K(X,X)
    plt.pcolor(X.T, X, K)
    plt.gca().invert_yaxis(), plt.gca().axis("image")
    plt.xlabel("x"), plt.ylabel("x'"), plt.colorbar()
    plt.title("$\kappa_{rbf}(x,x')$")
    plt.show()


def plot_val_histogram(fw, fh, feature):
    plt.figure(figsize=(fw, fh))
    plt.hist(feature, bins=100)
    plt.title("Histogram with 50 bins, values, ~ 16000")
    plt.show()


def plot_range_of_values(fw,fh,X,Y):
    plt.figure(figsize=(fw, fh))
    plt.hist(X, bins=100)
    plt.hist(Y, bins=100)
    plt.title("dimension values")
    plt.show()


def plot_range_of_values_and_sel(figx,figy, pts, sel_pts):
    plt.figure(figsize=(figx, figy))
    pts = np.array(pts)
    sel_pts = np.array(sel_pts)
    #
    plt.subplot(121)
    plt.hist(pts[:, 0], bins=100, color='orange')
    plt.hist(sel_pts[:,0], bins=100, color='red')

    plt.title("x"), plt.ylabel("$\kappa$")
    plt.title("$\kappa_{rbf}(x,0)$")
    #
    plt.subplot(122)
    plt.hist(pts[:,1], bins=100, color='blue')
    plt.hist(sel_pts[:,1], bins=100, color='purple')

    plt.title("$\kappa_{rbf}(x,x')$")
    plt.show()


def plot_algorithm_scale(figx, figy, n_array, time_array):
    plt.figure(figsize=(figx, figy))
    length = time_array.shape[0]
    for i in range(length):
        plt.scatter(n_array[i], time_array[i], color='grey', s=40)  # observations
    plt.show()


def plot_history(figx, figy ,history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure(figsize=(figx, figy))
    plt.autoscale(True, axis='both')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error')
    # plt.plot(hist['epoch'], hist['loss'], label = 'Val Error')
    # plt.ylim([0,5])
    plt.legend()
    plt.show()


def plot_encoder_output_distribution(figx, figy, values):
    plt.figure(figsize=(figx, figy))

    sns.distplot(values, hist=True, kde=True,
                 bins=int(180 / 5), color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})

    plt.show()


def plot_decoder_output_distribution(figx, figy, values):
    plt.figure(figsize=(figx, figy))

    for s in range(values.shape[1]):
        sns.distplot(values[:,s], hist=True, kde=True,
                     bins=int(180 / 5), color='darkblue',
                     hist_kws={'edgecolor': 'black', 'alpha': 0.1},
                     kde_kws={'linewidth': 4, 'alpha': 0.1})

    plt.show()



def pairplots(trainA):
    sns.set(style="ticks", color_codes=True)
    sns.pairplot(trainA, diag_kind='kde')
    plt.show()


def plot_gp_linesamples(figx, figy, et_r, t_idx_, w_pred_idx_linsp, samples_section, num_samples): # Plot the true function, observations, and posterior samples

    plt.figure(figsize=(figx, figy))
    input_dim  = et_r.shape[1]
    ''' section creation, TODO '''
    # if input_dim > 1:
        # line_XYZ = np.linspace(u, v, num=10000) #(num, 3) 3rd dim const.

        # line_XY = line_XYZ[:,0:2] # same as gpf.project_to_line_coordinates here
        # line_L = gpf.create_1d_w_line_coords(line_XY, u, v) # 2D input, 1D output
        # Z = gpf.sinusoid(line_XY).reshape(-1, 1) # 2D input 1D output
        # line_XYZ_ = gpf.add_dimension(line_XY, Z) # 2D input, 3D output
        # line_XYZL = gpf.add_dimension(line_XYZ_, line_L) # 3D input, 4D output
        # predictive_index_points_ = np.linspace(-2,2,200, dtype=np.float64)
        # predictive_index_points_ = predictive_index_points_[..., np.newaxis]

    ''' confidence interval '''
    for i in range(num_samples):
        if samples_section.ndim == 3:
            mu = samples_section[i,0,:] # 2D_emb:(200,)
            s = samples_section[i,1,:] # 2D_emb: (200,)
        # if samples_section.ndim == 2:
        #     mu = samples_section[i, :]  # 2D_emb:(50,500)
        #     s = samples_section[i, :]  # 2D_emb: (50,500)

        l_idx = w_pred_idx_linsp.flatten() # 2D_emb: (400, )

        # l_idx = np.linspace(np.min(w_pred_idx_linsp), np.min(w_pred_idx_linsp), samples_section.shape[2],dtype=np.float64)

        idxshape = l_idx.shape[0]/input_dim
        idxshape = int(idxshape)
        l_idx = l_idx[:idxshape]
        plt.gca().fill_between(l_idx, mu - s, mu+s, color="#dddddd")

        # GP samples #
        plt.plot(w_pred_idx_linsp, samples_section[i, 0, :].T, c='r', alpha=.08,
             label='Posterior Sample' if i == 0 else None)

        # plt.plt(line_idx, samples_section[i, 0, :].T,c='b',alpha=.04)
        # plt.plt(line_idx, samples_section[i, 1, :].T,c='b',alpha=.04)

    ''' observations '''
    plt.scatter(et_r[:,0], t_idx_, color='purple', s=40)  # observations
    plt.show()


def plot1d_data_samples_section(figx, figy, et_r, t_idx_r, w_pred_idx_linsp, samples_section, num_samples): # Plot the true function, observations, and posterior samples.
    ''' Instead of taking section, the input indices are already 1d
        Instead of gpf.sinusoid we can take
    '''

    plt.figure(figsize=(figx, figy))

    for i in range(num_samples):
        plt.plot(w_pred_idx_linsp, samples_section[i, :].T,c='r',alpha=.08,label='Posterior Sample' if i == 0 else None)
    leg = plt.legend(loc='upper right')
    plt.scatter( et_r, t_idx_r, color='purple', s=40)  # observations

    # line_XYZ = np.linspace(u, v, num=10000) #(num, 3) 3rd dim const.
    # line_XY = line_XYZ[:,0:2] # same as gpf.project_to_line_coordinates here
    # line_L = gpf.create_1d_w_line_coords(line_XY, u, v) # 2D input, 1D output

    # Z = gpf.sinusoid(line_XY).reshape(-1, 1) # 2D input 1D output

    # line_XYZ_ = gpf.add_dimension(line_XY, Z) # 2D input, 3D output
    # line_XYZL = gpf.add_dimension(line_XYZ_, line_L) # 3D input, 4D output
    # predictive_index_points_ = np.linspace(-2,2,200, dtype=np.float64)
    # predictive_index_points_ = predictive_index_points_[..., np.newaxis]

    # plt.scatter(line_L, Z, color='blue', s=0.2, alpha=0.3)  # sin cut

    # print("samples_section[i,:].T",samples_section[1, :].T )
    # # print("eppts[i , 0:2]",eppts[: , 0:2])
    # print("predictive_index_points_.shape : ", predictive_index_points_.shape)
    # print("samples_section.shape : ", samples_section.shape) # 50,1 200 instead 50,2,3
    # print("--------------")
    # plt.scatter(pred_idx_pts[:, 0], obs,
    #             label='Observations')

    # for lh in leg.legendHandles:
    #     lh.set_alpha(1)
    plt.xlabel(r"Index points ($\mathbb{R}^1$)")
    plt.ylabel("Observation space")
    plt.show()
