from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import ufl

import dolfinx
from dolfinx import default_scalar_type

def mark_exterior_boundary_entities(mesh, boundary_function):
    facet_dim = mesh.topology.dim-1

    boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, facet_dim, boundary_function)
    mesh.topology.create_connectivity(facet_dim, mesh.topology.dim)
    mesh.topology.create_connectivity(mesh.topology.dim, facet_dim)

    # dictionary of facet->cell
    f_to_c = mesh.topology.connectivity(facet_dim, mesh.topology.dim)
    # dictionary of cell->array[facets]
    c_to_f = mesh.topology.connectivity(mesh.topology.dim, facet_dim)

    boundary_entities = []
    # Loop over all edges that belong to our exterior
    for f in boundary_facets:
        # returns the cells linked to this edge
        cells = f_to_c.links(f)
        # Exterior facets only have 1 attached cell, 
        # hence get the first one 
        c = cells[0] 
        
        # Find the local index (e.g., 0, 1, 2, or 3 for quads) of facet f within cell c
        local_f = np.where(c_to_f.links(c) == f)[0][0]
        #print(f"f={f}, cell={c}, local_f = {local_f}")
        
        boundary_entities.extend([c, local_f])

    # FEniCSx custom form arrays must be typed as int32
    return np.array(boundary_entities, dtype=np.int32)

class IntegralDomain(Enum):
    """The geometric entities over which a custom kernel is integrated."""

    CELL = dolfinx.fem.IntegralType.cell
    EXTERIOR_FACET = dolfinx.fem.IntegralType.exterior_facet
    INTERIOR_FACET = dolfinx.fem.IntegralType.interior_facet

@dataclass(frozen=True)
class Domain:
    """
    Description of the entities over which a form is integrated.

    Parameters
    ----------
    integral_type:
        Type of integral (cell, exterior facet, interior facet).
    entities:
        Mesh entities used for integration.
        For cells this is a 1-D array of cell indices.
        For exterior/interior facets this is the entity array expected
        by DOLFINx for that integral type.
    subdomain_id:
        Integral/subdomain identifier. Usually 0 for an unmarked domain.
    """

    integral_type: IntegralDomain
    entities: np.ndarray
    subdomain_id: int = 0

def mark_cells(mesh) -> Domain:
    """Return all locally owned cells of `mesh`."""

    tdim = mesh.topology.dim
    n_local = mesh.topology.index_map(tdim).size_local

    entity_indices = np.arange(n_local,dtype=np.int32)

    return Domain(
        integral_type=IntegralDomain.CELL,
        entities=entity_indices,
    )

def exterior_facets(entities: np.ndarray, subdomain_id: int = 0) -> Domain:
    """
    Return an exterior-facet integration domain.

    `entities` should contain the facet integration entities produced by
    DOLFINx, e.g. the `(cell, local_facet)` pairs returned by
    `compute_integration_domains` / `locate_entities_boundary` workflows.
    """

    return Domain(
        integral_type=IntegralDomain.EXTERIOR_FACET,
        entities=np.asarray(entities, dtype=np.int32),
        subdomain_id=subdomain_id,
    )

def interior_facets(entities: np.ndarray, subdomain_id: int = 0) -> Domain:
    """Return an interior-facet integration domain."""

    return Domain(
        integral_type=IntegralDomain.INTERIOR_FACET,
        entities=np.asarray(entities, dtype=np.int32),
        subdomain_id=subdomain_id,
    )

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_constants(ufl_form: ufl.Form):
    """
    Extract UFL constants in exactly the order expected by FFCx.
    """
    return [
        c._cpp_object
        for c in ufl.algorithms.analysis.extract_constants(ufl_form)
    ]

def _make_integrals(integrals: Sequence[tuple[Domain, object]]) -> dict:
    """
    Convert user-facing (Domain, kernel) pairs to the low-level
    DOLFINx integral dictionary.
    """

    result = {}

    for domain, kernel in integrals:
        integral_type = domain.integral_type.value

        result.setdefault(integral_type, []).append(
            (
                domain.subdomain_id,
                kernel.address,
                domain.entities,
                np.array([0], dtype=np.int8),
            )
        )

    return result

def _make_form(*,mesh, spaces: Sequence,ufl_form: ufl.Form,
    coefficients: Sequence, integrals,
    dtype=default_scalar_type):
    """
    Internal bridge between a custom kernel and `dolfinx.fem.Form`.
    """
    if not isinstance(ufl_form, ufl.Form):
        raise TypeError(
            f"ufl_form must be a UFL form, got {type(ufl_form)!r}"
        )
    
    formtype = dolfinx.fem.form_cpp_class(dtype)

    cpp_form = formtype(
        spaces=[
            space._cpp_object
            for space in spaces
        ],
        integrals=_make_integrals(integrals),
        coefficients=[
            coefficient._cpp_object
            for coefficient in coefficients
        ],
        constants=_extract_constants(ufl_form),
        need_permutation_data=False,
        entity_maps=[],
        mesh=mesh._cpp_object,
    )

    return dolfinx.fem.Form(cpp_form)

def make_bilinear_form(*,mesh,ufl_form: ufl.Form,
    trial_space,test_space,coefficients=(),
    integrals=(),dtype=default_scalar_type,):
    """
    Build a bilinear DOLFINx form from one or more custom kernels.
    """

    if not isinstance(coefficients, (tuple, list)):
        coefficients = (coefficients,)

    return _make_form(
        mesh=mesh,
        spaces=(trial_space, test_space),
        ufl_form=ufl_form,
        coefficients=coefficients,
        integrals=integrals,
        dtype=dtype,
    )


def make_linear_form(*,mesh, ufl_form: ufl.Form,
    test_space,coefficients=(),
    integrals=(),dtype=default_scalar_type):
    """
    Build a linear DOLFINx form from one or more custom kernels.

    Parameters
    ----------
    integrals:
        Sequence of ``(domain, kernel)`` pairs. Multiple pairs may use
        different integral types, e.g. cell + exterior facet.
    """

    if not isinstance(coefficients, (tuple, list)):
        coefficients = (coefficients,)

    return _make_form(
        mesh=mesh,
        spaces=(test_space,),
        ufl_form=ufl_form,
        coefficients=coefficients,
        integrals=integrals,
        dtype=dtype,
    )