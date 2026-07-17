from phdtruel import visualisations
from phdtruel.fields import FieldOfInterest
from phdtruel.mappings import GaussianOTMapping
from phdtruel.visualisations.fields import plot_f0ref1


def plot_for_got_method(
    fields: list[FieldOfInterest],
    got_map: GaussianOTMapping,
    interpolation: FieldOfInterest | None = None,
    axsize: tuple[float, float] | None = None,
):
    # TODO : Add the gaussians fitted on the target field of initerst using the sensor
    # TODO : Add the transported gaussians using the mapping
    data = got_map.plot_data
    if data is None:
        raise ValueError("GaussianOTMapping must be fitted before plotting")

    fig, axes = visualisations.subplots(2, 3, axsize=axsize)

    fig.suptitle(f"Gaussians for GOT method for {data.sensor_name}")
    norm = plot_f0ref1(fig, axes[0, :], fields)

    u0_hat = data.u0_hat
    u1_hat = data.u1_hat

    g0_field = FieldOfInterest(
        u0_hat.parameter,
        u0_hat.mesh,
        data.g0(u0_hat.mesh.points),
        name="g0 values",
    )
    g1_field = FieldOfInterest(
        u1_hat.parameter,
        u1_hat.mesh,
        data.g1(u1_hat.mesh.points),
        name="g1 values",
    )

    visualisations.pcolormesh(fig, axes[1, 0], u0_hat, cmap="Blues", norm="minmax")
    for ax in [axes[0, 0], axes[1, 0]]:
        visualisations.contour(
            fig,
            ax,
            g0_field,
            colors="k",
            levels=10,
        )
    axes[1, 0].set_title(r"$\hat{u_0}$")
    visualisations.pcolormesh(fig, axes[1, 2], u1_hat, cmap="Blues", norm="minmax")
    for ax in [axes[0, 2], axes[1, 2]]:
        visualisations.contour(
            fig,
            ax,
            g1_field,
            colors="k",
            levels=10,
        )
    axes[1, 2].set_title(r"$\hat{u_1}$")

    if interpolation is None:
        axes[1, 1].axis("off")
    else:
        visualisations.pcolormesh(fig, axes[1, 1], interpolation, norm=norm)
        axes[1, 1].set_title("GOT interpolation")

    fig.tight_layout()
    visualisations.savefig(fig, "got_gaussians.png")
