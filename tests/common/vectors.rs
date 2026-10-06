//! Reading and writing the shared scenario files in `tests/vectors/`.
//!
//! Each scenario lives in `tests/vectors/<ID>/`: a `model.json` describing the problem, plus raw
//! little-endian arrays of shape `[runs, steps, dim]` for the true states and measurements. The
//! format is deliberately plain so the C, C++ and Python implementations can read the same
//! files. See `tests/vectors/README.md`.

use std::fs;
use std::path::PathBuf;

use nalgebra::{SMatrix, SVector};
use serde::{Deserialize, Serialize};

use super::scenarios::Scenario;

/// The contents of `model.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub description: String,
    pub state_dim: usize,
    pub measurement_dim: usize,
    /// `"linear"` (z = H x) or `"range_bearing"` (z = [‖(x, y)‖, atan2(y, x)]).
    pub measurement_model: String,
    pub dt: f64,
    pub runs: usize,
    pub steps: usize,
    /// The generator seed of each run.
    pub seeds: Vec<u64>,
    /// Matrices are row-major: `f[i][j]` is row `i`, column `j`.
    pub f: Vec<Vec<f64>>,
    pub h: Vec<Vec<f64>>,
    pub q: Vec<Vec<f64>>,
    pub r: Vec<Vec<f64>>,
    pub x0: Vec<f64>,
    pub p0: Vec<Vec<f64>>,
    /// True states, if stored. Absent when only stability is measured.
    pub truth: Option<DataFile>,
    pub measurements: DataFile,
}

/// A raw little-endian array file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataFile {
    pub file: String,
    /// `"float64"` or `"float32"`.
    pub dtype: String,
    /// `[runs, steps, dim]`, row-major.
    pub shape: [usize; 3],
}

/// Returns `tests/vectors/<id>`.
pub fn dir(id: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/vectors")
        .join(id)
}

fn rows<const R: usize, const C: usize>(m: &SMatrix<f64, R, C>) -> Vec<Vec<f64>> {
    (0..R)
        .map(|i| (0..C).map(|j| m[(i, j)]).collect())
        .collect()
}

fn matrix<const R: usize, const C: usize>(rows: &[Vec<f64>]) -> SMatrix<f64, R, C> {
    assert_eq!(rows.len(), R, "wrong row count");
    SMatrix::from_fn(|i, j| rows[i][j])
}

fn encode(values: impl Iterator<Item = f64>, dtype: &str) -> Vec<u8> {
    match dtype {
        "float64" => values.flat_map(f64::to_le_bytes).collect(),
        "float32" => values.flat_map(|v| (v as f32).to_le_bytes()).collect(),
        other => panic!("unknown dtype {other}"),
    }
}

fn decode(bytes: &[u8], dtype: &str) -> Vec<f64> {
    match dtype {
        "float64" => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect(),
        "float32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f64::from(f32::from_le_bytes(*b)))
            .collect(),
        other => panic!("unknown dtype {other}"),
    }
}

/// What to write for one scenario.
pub struct Spec<'a> {
    pub id: &'a str,
    pub description: &'a str,
    pub measurement_model: &'a str,
    pub dt: f64,
    pub seeds: Vec<u64>,
    pub measurement_dtype: &'a str,
    pub with_truth: bool,
}

/// Writes `runs` (all sharing one model) to `tests/vectors/<spec.id>/`.
pub fn write<const N: usize, const M: usize>(spec: &Spec, runs: &[Scenario<N, M>]) {
    let first = &runs[0];
    let steps = first.zs.len();
    assert!(
        runs.iter().all(|r| r.zs.len() == steps),
        "runs differ in length"
    );
    let dir = dir(spec.id);
    fs::create_dir_all(&dir).unwrap();

    let truth = spec.with_truth.then(|| {
        let file = DataFile {
            file: "truth.bin".into(),
            dtype: "float64".into(),
            shape: [runs.len(), steps, N],
        };
        let values = runs
            .iter()
            .flat_map(|r| r.truth.iter().flat_map(|x| x.iter().copied()));
        fs::write(dir.join(&file.file), encode(values, &file.dtype)).unwrap();
        file
    });
    let measurements = DataFile {
        file: "measurements.bin".into(),
        dtype: spec.measurement_dtype.into(),
        shape: [runs.len(), steps, M],
    };
    let values = runs
        .iter()
        .flat_map(|r| r.zs.iter().flat_map(|z| z.iter().copied()));
    fs::write(
        dir.join(&measurements.file),
        encode(values, &measurements.dtype),
    )
    .unwrap();

    let model = Model {
        id: spec.id.into(),
        description: spec.description.into(),
        state_dim: N,
        measurement_dim: M,
        measurement_model: spec.measurement_model.into(),
        dt: spec.dt,
        runs: runs.len(),
        steps,
        seeds: spec.seeds.clone(),
        f: rows(&first.f),
        h: rows(&first.h),
        q: rows(&first.q),
        r: rows(&first.r),
        x0: first.x0.iter().copied().collect(),
        p0: rows(&first.p0),
        truth,
        measurements,
    };
    let json = serde_json::to_string_pretty(&model).unwrap();
    fs::write(dir.join("model.json"), json + "\n").unwrap();
}

/// Loads scenario `id`, returning its model description and one `Scenario` per run.
///
/// Runs without stored truth have an empty `truth`. `float32` measurements are widened to
/// `f64` exactly, so casting them back to `f32` recovers the stored values.
pub fn load<const N: usize, const M: usize>(id: &str) -> (Model, Vec<Scenario<N, M>>) {
    let dir = dir(id);
    let json = fs::read_to_string(dir.join("model.json"))
        .unwrap_or_else(|e| panic!("can't read {id}/model.json: {e}"));
    let model: Model = serde_json::from_str(&json).unwrap();
    assert_eq!(
        (model.state_dim, model.measurement_dim),
        (N, M),
        "{id}: wrong dimensions"
    );

    let read = |file: &DataFile, dim: usize| {
        let values = decode(&fs::read(dir.join(&file.file)).unwrap(), &file.dtype);
        assert_eq!(
            file.shape,
            [model.runs, model.steps, dim],
            "{id}: bad shape"
        );
        assert_eq!(
            values.len(),
            model.runs * model.steps * dim,
            "{id}: bad file length"
        );
        values
    };
    let zs = read(&model.measurements, M);
    let truth = model.truth.as_ref().map(|file| read(file, N));

    let runs = (0..model.runs)
        .map(|run| Scenario {
            f: matrix(&model.f),
            h: matrix(&model.h),
            q: matrix(&model.q),
            r: matrix(&model.r),
            x0: SVector::from_iterator(model.x0.iter().copied()),
            p0: matrix(&model.p0),
            truth: truth.as_ref().map_or_else(Vec::new, |t| {
                let run_values = &t[run * model.steps * N..(run + 1) * model.steps * N];
                run_values
                    .as_chunks::<N>()
                    .0
                    .iter()
                    .map(|c| SVector::from(*c))
                    .collect()
            }),
            zs: zs[run * model.steps * M..(run + 1) * model.steps * M]
                .as_chunks::<M>()
                .0
                .iter()
                .map(|c| SVector::from(*c))
                .collect(),
        })
        .collect();
    (model, runs)
}
