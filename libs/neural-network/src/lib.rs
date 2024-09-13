use rand::{Rng, RngCore};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("got {got} inputs, but {expected} inputs were expected")]
    MismatchedInputSize { got: usize, expected: usize },
}

#[derive(Debug)]
pub struct LayerTopology {
    pub neurons: usize,
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);

        let layers = layers
            .windows(2)
            .map(|layers| Layer::random(rng, layers[0].neurons, layers[1].neurons))
            .collect();

        Self { layers }
    }

    pub fn propogate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |acc, layer| layer.propagate(acc))
    }
}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn random(rng: &mut dyn RngCore, input_size: usize, output_size: usize) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::random(rng, input_size))
            .collect();

        Self { neurons }
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .filter_map(|neuron| neuron.propagate(&inputs).ok())
            .collect()
    }
}

#[derive(Debug)]
struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    fn random(rng: &mut dyn RngCore, size: usize) -> Self {
        let bias = rng.gen_range(-1.0..=1.0);

        let weights = (0..size).map(|_| rng.gen_range(-1.0..=1.0)).collect();

        Self { bias, weights }
    }

    fn propagate(&self, inputs: &[f32]) -> Result<f32> {
        if inputs.len() != self.weights.len() {
            return Err(Error::MismatchedInputSize {
                got: inputs.len(),
                expected: self.weights.len(),
            });
        };

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(i, weight)| i * weight)
            .sum::<f32>();

        Ok((self.bias + output).max(0.0))
    }
}
