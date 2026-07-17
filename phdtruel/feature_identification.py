import numpy as np
from contourpy import contour_generator


DEFAULT_SEED = 96


def generate_contour(x, y, image, levels):
    cont_gen = contour_generator(
        x,
        y,
        image,
    )

    contour_segments: list[np.ndarray] = []
    for level in levels:
        lines = cont_gen.lines(level)
        if len(lines) > 0:
            for line in lines:
                contour_segments.append(np.asarray(line))

    if contour_segments:
        return np.vstack(contour_segments)
    return np.zeros((0, 2))


# def random_points_in_feature_only():
#     cpd_points_indices_0 = feature_pts_indices_0[
#         np.random.randint(0, feature_pts_indices_0.shape[0], NB_CPD_POINTS)
#     ].ravel()
#     cpd_points_indices_1 = feature_pts_indices_1[
#         np.random.randint(0, feature_pts_indices_1.shape[0], NB_CPD_POINTS)
#     ].ravel()
#     target = points[cpd_points_indices_1, :]
#     source = points[cpd_points_indices_0, :]

#     return source, target


# def random_gaussian(points, field, nb_points=1000):
#     from scipy.stats import multivariate_normal
#
#     from phdtruel.fields.gaussians import Gaussian
#
#     g = Gaussian.from_field(points, field)
#
#     feature_points = multivariate_normal(mean=g.mu.ravel(), cov=g.sigma).rvs(nb_points)
#     return feature_points


# def random_with_distribution_based_on_field(
#     points, field, nb_points=1000, replace=False, seed=None
# ):
#     # TODO add seed
#     if seed is None:
#         seed = DEFAULT_SEED
#
#     mean = np.mean(field)
#     field_abs_centered = np.abs(field.ravel() - mean)
#
#     # distribution = (field_abs_centered - field_abs_centered.min()) / (field_abs_centered.max() - field_abs_centered.min())
#     distribution = field_abs_centered - field_abs_centered.min()
#     distribution = distribution**2
#     distribution = distribution / np.sum(distribution)
#
#     feature_points_indices = np.random.choice(
#         range(points.shape[0]), nb_points, p=distribution, replace=replace
#     )
#
#     feature_points = points[feature_points_indices, :]
#     return feature_points


# def random_uniform_sampled_on_field(points, nb_points=1000, seed=DEFAULT_SEED):
#
#     field_points_indices = np.random.randint(0, points.shape[0], nb_points)
#
#     sampled_points = points[field_points_indices, :]
#     return sampled_points


def grid_on_field(x, y, nb_x=11, nb_y=11):
    nx = x.shape[0]
    ny = x.shape[1]
    ix, iy = np.meshgrid(
        np.linspace(0, nx - 1, nb_x, endpoint=True, dtype=int),
        np.linspace(0, ny - 1, nb_y, endpoint=True, dtype=int),
    )
    grid_points_indices = ix.ravel() + nx * iy.ravel()

    points = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
    grid_points = points[grid_points_indices, :]
    return grid_points


# def sample_field(u: FieldOfInterestLike, n: int, concentration=1.0, seed=DEFAULT_SEED):
#     # TODO add seed
#
#     feature_points = np.zeros((0, 2))
#     while len(feature_points) < n:
#         random_points = (np.random.rand(n * 100, 2) - 0.5) * 20.0
#         random_weights = np.random.rand(n * 100, 1)
#         values = u(random_points)
#         values = np.abs(values)
#         values = (values - values.min()) / values.max()
#         mask = values > random_weights ** (1.0 / concentration)
#         new_feature_points = random_points[mask.ravel(), :]
#         feature_points = np.vstack([feature_points, new_feature_points])
#         print(len(feature_points))
#     return feature_points[:n, :]


# def borders_of_domain(x,y):
#     points = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])


#     x[0,:] =
# def sample_on_distrib(
#     n_points: int,
#     normalized_field_values,
#     field_points_coordinates,
#     p=1,
#     seed=DEFAULT_SEED,
# ):
#     """Sample points according to any normalized field.
#
#     The sampling is continuous on the interpolated normalized field.
#     Uses the rejection sampling method.
#     """
#
#     dim = field_points_coordinates.shape[1]
#
#     from scipy.interpolate import griddata
#
#     x_min, x_max = np.min(field_points_coordinates[:, 0]), np.max(
#         field_points_coordinates[:, 0]
#     )
#     y_min, y_max = np.min(field_points_coordinates[:, 1]), np.max(
#         field_points_coordinates[:, 1]
#     )
#
#     samples = np.zeros((0, dim))
#     while len(samples) < n_points:
#         # Sample candidates on uniform distrib
#         oversample_ratio = 100  # To sample faster ?
#         sample_candidates = np.random.rand(n_points * oversample_ratio, dim)
#         sample_candidates[:, 0] = sample_candidates[:, 0] * (x_max - x_min) + x_min
#         sample_candidates[:, 1] = sample_candidates[:, 1] * (y_max - y_min) + y_min
#
#         candidates_values = griddata(
#             field_points_coordinates, normalized_field_values, sample_candidates
#         )
#         # Retain candidates that are at big field values using a uniform distribution.
#         to_retain = candidates_values.ravel() >= np.random.rand(
#             n_points * oversample_ratio
#         ) ** (1.0 / p)
#
#         samples = np.vstack((samples, sample_candidates[to_retain]))
#     samples = samples[:n_points]
#     return samples
