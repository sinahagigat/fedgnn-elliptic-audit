// graph_ops.rs — reference Rust (PyO3) implementation of compute_boundary_fast
// ==============================================================================
// This file is kept only as a future extension path (outside the scope of the
// paper); compiling it is not needed to run the project on Elliptic, because
// fallback.py is fully sufficient.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet};
use std::collections::{HashMap, HashSet};

#[pyfunction]
fn compute_boundary(
    src: Vec<i64>,
    dst: Vec<i64>,
    assignment: Vec<i64>,
) -> PyResult<(HashMap<i64, i64>, HashMap<i64, HashSet<i64>>)> {
    let mut boundary_degree: HashMap<i64, i64> = HashMap::new();
    let mut neighbor_map: HashMap<i64, HashSet<i64>> = HashMap::new();

    for i in 0..src.len() {
        let s = src[i];
        let d = dst[i];
        let cs = assignment[s as usize];
        let cd = assignment[d as usize];
        if cs != cd {
            *boundary_degree.entry(s).or_insert(0) += 1;
            neighbor_map.entry(s).or_insert_with(HashSet::new).insert(cd);
        }
    }

    Ok((boundary_degree, neighbor_map))
}

#[pymodule]
fn fedgnn_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_boundary, m)?)?;
    Ok(())
}
