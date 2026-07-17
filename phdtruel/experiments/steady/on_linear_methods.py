import logging
import math

from phdtruel.data.data import construct_random_gmm_fields_of_interest
from phdtruel.descriptors.descriptors import FieldDescriptorFactory, FOIasDescriptor
from phdtruel.experiments.experiment_compagnon import ExperimentCompagnon
from phdtruel.interpolations.align_stripes import AlignStripesInterpolator
from phdtruel.interpolations.preconfigured_methods import linear_interpolation
from phdtruel.pod import PODInterpolator
from phdtruel.sensors import sensors
from phdtruel.visualisations.experiment import plot_result
from phdtruel.visualisations.interpolations import (
    plot_linear_interpolation_and_snapshots,
    plot_many_fields,
)
from phdtruel.visualisations.pod import plot_pod_modes, plot_singular_values

logger = logging.getLogger(__name__)


def linear_baselines_experiment(exp: ExperimentCompagnon):
    field = linear_interpolation(exp.fields, exp.test_mesh)

    exp.add_result("Linear", field, hook=plot_result)
    plot_linear_interpolation_and_snapshots(exp, field)

    pod_projection_experiment(exp.fields, exp, f"{len(exp.fields)} original snapshots")


def pod_projection_experiment(enriched_fields, exp: ExperimentCompagnon, name: str):
    pod = PODInterpolator(exp.test_points, enriched_fields)
    projected_field = pod.projection(exp.target_field, as_foi=True)

    exp.add_result(
        f"Projection in POD({name})",
        projected_field,
        hook=plot_result,
    )
    fig, _ = plot_pod_modes(pod.pod_basis, exp=exp)
    if exp.savefig is not None:
        exp.savefig(fig, f"pod_modes_on_{name}")
    fig, _ = plot_singular_values(pod.singular_values)
    if exp.savefig is not None:
        exp.savefig(fig, f"singular_values_on_{name}")


def align_wake_stripes_experiment(exp: ExperimentCompagnon):
    logger.info("Non linear baseline")

    for source_field in exp.fields:
        target_field = exp.target_field
        stripe_interpolator = AlignStripesInterpolator(
            [source_field, target_field],
            sensors.identity_sensor,
            FieldDescriptorFactory(FOIasDescriptor),
            mapping_factory=None,
        )
        interp_field = stripe_interpolator.interpolate(
            points=exp.test_points,
            s=1.0,
            strategy="cdi_ot1d",
        )

        exp.add_result(
            f"Aligned wake stripes {source_field.parameter}",
            interp_field,
            hook=plot_result,
        )


def gmm_enrich_experiment(exp: ExperimentCompagnon, map_shape: tuple):
    generated_gmm_fields = construct_random_gmm_fields_of_interest(
        exp.test_mesh,
        num_fields=math.prod(map_shape) - 4,
        num_gaussians=8,
    )
    fig, _ = plot_many_fields(
        {str(i): field for i, field in enumerate(generated_gmm_fields)},
        truncation=exp.truncation,
    )
    if exp.savefig is not None:
        exp.savefig(fig, "generated_fields.png")
    pod_projection_experiment(
        exp.fields + generated_gmm_fields,
        exp,
        name=f"{len(generated_gmm_fields)} rnd + {len(exp.fields)} originals",
    )
