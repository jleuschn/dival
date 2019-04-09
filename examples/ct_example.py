import numpy as np
import odl
from evaluation import EvaluationTaskTable, TestData, L2Measure, PSNRMeasure
from reconstruction import (FunctionReconstructor, CGReconstructor,
                            GaussNewtonReconstructor, LandweberReconstructor)

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

test_data = TestData(observation, ground_truth, name='shepp_logan + gaussian')

eval_tt = EvaluationTaskTable()

fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)

fbp_reconstructor = FunctionReconstructor(fbp)
cg_reconstructor = CGReconstructor(ray_trafo, reco_space.zero(), 7)
gn_reconstructor = GaussNewtonReconstructor(ray_trafo, reco_space.zero(), 4)
lw_reconstructor = LandweberReconstructor(ray_trafo, reco_space.zero(), 100)
reconstructors = [fbp_reconstructor, cg_reconstructor, gn_reconstructor,
                  lw_reconstructor]
eval_tt.append_all_combinations([test_data], reconstructors)
results = eval_tt.run()
results.plot_all_reconstructions()
results.apply_measures([L2Measure(), PSNRMeasure()])
print(results)
