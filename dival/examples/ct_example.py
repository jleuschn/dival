import numpy as np
import odl
from dival.data import DataPairs
from dival.evaluation import TaskTable
from dival.measure import PSNR, SSIM
from dival.reconstructors.odl_reconstructors import (FBPReconstructor,
                                                     CGReconstructor,
                                                     GaussNewtonReconstructor,
                                                     LandweberReconstructor,
                                                     MLEMReconstructor,
                                                     ISTAReconstructor,
                                                     PDHGReconstructor,
                                                     DouglasRachfordReconstructor,
                                                     ForwardBackwardReconstructor,
                                                     ADMMReconstructor, 
                                                     BFGSReconstructor)

np.random.seed(0)

# %% data
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300],
    dtype='float32')
phantom = odl.phantom.shepp_logan(reco_space, modified=True)
ground_truth = phantom

geometry = odl.tomo.cone_beam_geometry(reco_space, 40, 40, 360)
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cpu')
proj_data = ray_trafo(phantom)
observation = (proj_data + np.random.poisson(0.3, proj_data.shape)).asarray()

test_data = DataPairs(observation, ground_truth, name='shepp-logan + pois')

# %% task table and reconstructors
eval_tt = TaskTable()


fbp_reconstructor = FBPReconstructor(ray_trafo, hyper_params={
    'filter_type': 'Hann',
    'frequency_scaling': 0.8})
cg_reconstructor = CGReconstructor(ray_trafo, reco_space.zero(), 4)
gn_reconstructor = GaussNewtonReconstructor(ray_trafo, reco_space.zero(), 2)
lw_reconstructor = LandweberReconstructor(ray_trafo, reco_space.zero(), 8)
mlem_reconstructor = MLEMReconstructor(ray_trafo, 0.5*reco_space.one(), 1)
ista_reconstructor = ISTAReconstructor(ray_trafo,reco_space.zero(), 10)
pdhg_reconstructor = PDHGReconstructor(ray_trafo, reco_space.zero(), 10)
dougrach_reconstructor = DouglasRachfordReconstructor(ray_trafo, 
                                                      reco_space.zero(), 10)
forwardbackward_reconstructor = ForwardBackwardReconstructor(ray_trafo, 
                                                      reco_space.zero(), 10)
admm_reconstructor = ADMMReconstructor(ray_trafo, reco_space.zero(), 10)
bfgs_reconstructor = BFGSReconstructor(ray_trafo, reco_space.zero(), 10)

reconstructors = [fbp_reconstructor, cg_reconstructor, gn_reconstructor,
                  lw_reconstructor, mlem_reconstructor, ista_reconstructor,
                  pdhg_reconstructor, dougrach_reconstructor, 
                  forwardbackward_reconstructor, admm_reconstructor, 
                  bfgs_reconstructor]

options = {'save_iterates': True}

eval_tt.append_all_combinations(reconstructors=reconstructors,
                                test_data=[test_data], options=options)

# %% run task table
results = eval_tt.run()
results.apply_measures([PSNR, SSIM])
print(results)

# %% plot reconstructions
fig = results.plot_all_reconstructions(fig_size=(9, 4), vrange='individual')

# %% plot convergence of CG
results.plot_convergence(1, fig_size=(9, 6), gridspec_kw={'hspace': 0.5})

# %% plot performance
results.plot_performance(PSNR, figsize=(10, 4))
