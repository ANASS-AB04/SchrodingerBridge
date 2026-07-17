import numpy as np

from phdtruel.data.nascar_dataloader import ParameterAsDict
from phdtruel.experiments.experiment_compagnon import ExperimentCompagnon
from phdtruel.fields import DynamicOfInterest
from phdtruel.fields.parameters import ParameterSet
from phdtruel.interpolations.avg_descriptors_cdi import AveragedDescriptorsCDI
from phdtruel.sensors.sensors import localize_field_fit_gaussian


def interpolate_along_time(
    exp: ExperimentCompagnon,
    got: AveragedDescriptorsCDI,
    test_timesteps: list[int | float] | np.ndarray,
    target_parameter: ParameterAsDict | ParameterSet,
    _override_gaussian_with_dynamic: DynamicOfInterest | None = None,
):
    if isinstance(target_parameter, ParameterSet):
        target_parameter = target_parameter.as_dict()

    fields = []
    parameters = []
    gaussians = []
    overrided_gaussian = None
    for k, t in enumerate(test_timesteps):
        test_parameter = ParameterSet(**{**target_parameter, "t": t})
        parameters.append(test_parameter)
        if _override_gaussian_with_dynamic is not None:
            target_field = _override_gaussian_with_dynamic.fields[k]
            overrided_gaussian = localize_field_fit_gaussian(target_field, p=2)

        field = got(
            test_parameter,
            exp.test_points,
            foi_shape=exp.test_points_shape,
            return_foi=True,
            _override_mean_gaussian=overrided_gaussian,
        )
        gaussians.append(got._last_mean_gaussian)
        fields.append(field)
    dynamic = DynamicOfInterest(
        ParameterSet(**target_parameter), fields, np.asarray(test_timesteps)
    )
    return dynamic, gaussians
