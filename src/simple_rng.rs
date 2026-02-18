// -----------------------------
// Simple RNG (XorShift64 + Marsaglia Polar Method)
// -----------------------------

pub struct Rng {
    state: u64,
    next_gauss: Option<f64>, // Cache for the second Gaussian value
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Ensure the seed is never zero for the Xorshift algorithm
        let s = if seed == 0 { 0xACE1_u64 } else { seed };
        Self { state: s, next_gauss: None }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 16) as u32
    }

    /// Returns a value in the open interval (0, 1)
    pub fn uniform(&mut self) -> f64 {
        // This mapping ensures the result is in the open interval (0, 1)
        (self.next_u32() as f64 + 0.5) / (u32::MAX as f64 + 1.0)
    }

    /// Returns a value in the range (-1.0, 1.0)
    fn uniform_signed(&mut self) -> f64 {
        (self.next_u32() as f64 / u32::MAX as f64) * 2.0 - 1.0
    }

    pub fn gauss(&mut self, mean: f64, std: f64) -> f64 {
        // Check if we have a cached value from the last run
        if let Some(second_val) = self.next_gauss.take() {
            return mean + std * second_val;
        }

        // Marsaglia Polar Method
        let mut x: f64;
        let mut y: f64;
        let mut s: f64;

        loop {
            x = self.uniform_signed();
            y = self.uniform_signed();
            s = x * x + y * y;
            // Reject if outside the unit circle or at the center (to avoid ln(0))
            if s < 1.0 && s > 0.0 { break; }
        }

        let multiplier = (-2.0 * s.ln() / s).sqrt();

        // Store the second value for the next call
        self.next_gauss = Some(y * multiplier);

        // Return the first value
        mean + std * x * multiplier
    }
}
