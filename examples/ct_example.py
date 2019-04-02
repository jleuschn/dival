import numpy as np
import odl
from evaluation import EvaluationTaskTable, TestData, L2Measure
from reconstruction import OperatorReconstructor, CGReconstructor

reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300],
    dtype='float32')
phantom = odl.phantom.shepp_logan(reco_space, modified=True)
ground_truth = phantom

detector_partition = odl.uniform_partition(-60, 60, 512)
geometry = odl.tomo.cone_beam_geometry(reco_space, 40, 40, 360)
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cpu')
proj_data = ray_trafo(phantom)
observation = proj_data + np.random.standard_normal(proj_data.shape)

test_data = TestData(observation, ground_truth)

eval_tt = EvaluationTaskTable()

fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)

fbp_reconstructor = OperatorReconstructor(fbp)
cg_reconstructor = CGReconstructor(ray_trafo, reco_space.zero(), 7)
eval_tt.append(test_data, fbp_reconstructor, [L2Measure()])
eval_tt.append(test_data, cg_reconstructor, [L2Measure()])
results = eval_tt.run()
results.plot_reconstruction(0)
results.plot_reconstruction(1)
