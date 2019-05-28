import numpy as np
import odl
from dival.data import TestData
from dival.evaluation import TaskTable
from dival.measure import L2
from dival.reconstructors.odl_reconstructors import FBPReconstructor

np.random.seed(0)

# %% data
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300],
    dtype='float32')
phantom = odl.phantom.shepp_logan(reco_space, modified=True)
ground_truth = phantom

geometry = odl.tomo.parallel_beam_geometry(reco_space, 30, 182)
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
proj_data = ray_trafo(phantom)
observation = proj_data + np.random.poisson(0.3, proj_data.shape)

test_data = TestData(observation, ground_truth, name='shepp_logan + gaussian')

# %% task table and reconstructors
eval_tt = TaskTable()

rs = np.random.RandomState(0)
reconstructor = FBPReconstructor(ray_trafo)
options = {'save_iterates': True,
           'hyper_param_search': {'measure': L2,
                                  'hyperopt_rstate': rs}}

eval_tt.append(reconstructor=reconstructor, test_data=test_data,
               options=options)

# %% run task table
results = eval_tt.run()
print(results.to_string(formatters={'reconstructor': lambda r: r.name}))

# %% plot reconstructions
fig = results.plot_all_reconstructions(fig_size=(9, 4), vrange='individual')
