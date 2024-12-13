use bevy::prelude::*;
use cores::{EvenCore, EvenCoreError, UnevenCore, UnevenCoreError};
use serde::{Deserialize, Serialize};

/// This is just a convenience wrapper over some serializable built-in Bevy curve types
/// These curves are all intended to have unit domain ([0, 1]).
#[derive(Serialize, Deserialize, Clone, Reflect, Debug)]
pub enum FireworkCurve<T> {
    SampleAuto(SampleAutoCurve<T>),
    UnevenSampleAuto(UnevenSampleAutoCurve<T>),
    Constant(ConstantCurve<T>),
}

impl<T> Curve<T> for FireworkCurve<T>
where
    T: StableInterpolate,
{
    fn domain(&self) -> Interval {
        match self {
            FireworkCurve::SampleAuto(c) => c.domain(),
            FireworkCurve::UnevenSampleAuto(c) => c.domain(),
            FireworkCurve::Constant(c) => c.domain(),
        }
    }

    fn sample_unchecked(&self, t: f32) -> T {
        match self {
            FireworkCurve::SampleAuto(c) => c.sample_unchecked(t),
            FireworkCurve::UnevenSampleAuto(c) => c.sample_unchecked(t),
            FireworkCurve::Constant(c) => c.sample_unchecked(t),
        }
    }
}

impl<T: Clone> FireworkCurve<T> {
    /// Creates an appropriate curve type based on the number of samples (e.g. if 1 sample, then
    /// constant. If 2 samples, then UnevenSampleAutoCurve)
    ///
    /// If constant, the domain will be [0,1].
    pub fn uneven_samples(samples: impl IntoIterator<Item = (f32, T)>) -> Self {
        // PERF: We only really need to get the first two items out to figure out the curve type.
        //       It shouldn't really affect performance much though, this function is not in hot path.
        let samples = samples.into_iter().collect::<Vec<_>>();
        match samples.len() {
            0 => panic!("Cannot create curve from 0 samples"),
            1 => FireworkCurve::Constant(ConstantCurve::new(
                interval(0., 1.).unwrap(),
                samples[0].1.clone(),
            )),
            _ => FireworkCurve::UnevenSampleAuto(UnevenSampleAutoCurve::new(samples).unwrap()),
        }
    }

    /// Creates an appropriate curve type based on the number of samples (e.g. if 1 sample, then
    /// constant. If 2 samples, then SampleAutoCurve)
    pub fn even_samples(samples: impl IntoIterator<Item = T>) -> Self {
        // PERF: We only really need to get the first two items out to figure out the curve type.
        //       It shouldn't really affect performance much though, this function is not in hot path.
        let samples = samples.into_iter().collect::<Vec<_>>();
        match samples.len() {
            0 => panic!("Cannot create curve from 0 samples"),
            1 => FireworkCurve::Constant(ConstantCurve::new(
                interval(0., 1.).unwrap(),
                samples[0].clone(),
            )),
            _ => FireworkCurve::SampleAuto(
                SampleAutoCurve::new(interval(0., 1.).unwrap(), samples).unwrap(),
            ),
        }
    }

    pub fn constant(sample: T) -> Self {
        FireworkCurve::Constant(ConstantCurve::new(interval(0., 1.).unwrap(), sample))
    }
}

/// A curve whose samples are defined by a collection of colors, with 0..1 domain
#[derive(Clone, Debug, Reflect, Serialize, Deserialize)]
pub struct ColorSampleAutoCurve<T> {
    core: EvenCore<T>,
}

impl<T> ColorSampleAutoCurve<T>
where
    T: Mix + Clone,
{
    pub fn new(colors: impl IntoIterator<Item = T>) -> Result<Self, EvenCoreError> {
        let colors = colors.into_iter().collect::<Vec<_>>();
        if colors.len() < 2 {
            Err(EvenCoreError::NotEnoughSamples {
                samples: colors.len(),
            })
        } else {
            Ok(Self {
                core: EvenCore::new(Interval::new(0., 1.).unwrap(), colors)?,
            })
        }
    }
}

impl<T> Curve<T> for ColorSampleAutoCurve<T>
where
    T: Mix + Clone,
{
    #[inline]
    fn domain(&self) -> Interval {
        interval(0., 1.).unwrap()
    }

    #[inline]
    fn sample_clamped(&self, t: f32) -> T {
        // `EvenCore::sample_with` clamps the input implicitly.
        self.core.sample_with(t, T::mix)
    }

    #[inline]
    fn sample_unchecked(&self, t: f32) -> T {
        self.sample_clamped(t)
    }
}

/// A curve whose samples are defined by a collection of colors, with 0..1 domain
#[derive(Clone, Debug, Reflect, Serialize, Deserialize)]
pub struct ColorSampleUnevenAutoCurve<T> {
    core: UnevenCore<T>,
}

impl<T> ColorSampleUnevenAutoCurve<T>
where
    T: Mix + Clone,
{
    pub fn new(colors: impl IntoIterator<Item = (f32, T)>) -> Result<Self, UnevenCoreError> {
        let colors = colors.into_iter().collect::<Vec<_>>();
        if colors.len() < 2 {
            Err(UnevenCoreError::NotEnoughSamples {
                samples: colors.len(),
            })
        } else {
            Ok(Self {
                core: UnevenCore::new(colors)?,
            })
        }
    }
}

impl<T> Curve<T> for ColorSampleUnevenAutoCurve<T>
where
    T: Mix + Clone,
{
    #[inline]
    fn domain(&self) -> Interval {
        interval(0., 1.).unwrap()
    }

    #[inline]
    fn sample_clamped(&self, t: f32) -> T {
        self.core.sample_with(t, T::mix)
    }

    #[inline]
    fn sample_unchecked(&self, t: f32) -> T {
        self.sample_clamped(t)
    }
}

/// We currently cannot reuse [`FireworkCurve`] for colors because colors
/// don't implement [`StableInterpolate`]. Instead, they implement [`Mix`] so we
/// must define dedicated curve types to achieve the same goal.
/// These curves are all intended to have unit domain ([0, 1]).
#[derive(Serialize, Deserialize, Clone, Reflect, Debug)]
pub enum FireworkGradient<T> {
    ColorSampleAuto(ColorSampleAutoCurve<T>),
    ColorSampleUnevenAuto(ColorSampleUnevenAutoCurve<T>),
    Constant(ConstantCurve<T>),
}

impl<T> Curve<T> for FireworkGradient<T>
where
    T: Mix + Clone,
{
    fn domain(&self) -> Interval {
        match self {
            FireworkGradient::ColorSampleAuto(c) => c.domain(),
            FireworkGradient::ColorSampleUnevenAuto(c) => c.domain(),
            FireworkGradient::Constant(c) => c.domain(),
        }
    }

    fn sample_unchecked(&self, t: f32) -> T {
        match self {
            FireworkGradient::ColorSampleAuto(c) => c.sample_unchecked(t),
            FireworkGradient::ColorSampleUnevenAuto(c) => c.sample_unchecked(t),
            FireworkGradient::Constant(c) => c.sample_unchecked(t),
        }
    }
}

impl<T> FireworkGradient<T>
where
    T: Mix + Clone,
{
    /// Creates an appropriate curve type based on the number of samples (e.g. if 1 sample, then
    /// constant. If 2 samples, then UnevenSampleAutoCurve)
    ///
    /// If constant, the domain will be [0,1].
    pub fn uneven_samples(samples: impl IntoIterator<Item = (f32, T)>) -> Self {
        // PERF: We only really need to get the first two items out to figure out the curve type.
        //       It shouldn't really affect performance much though, this function is not in hot path.
        let samples = samples.into_iter().collect::<Vec<_>>();
        match samples.len() {
            0 => panic!("Cannot create curve from 0 samples"),
            1 => FireworkGradient::Constant(ConstantCurve::new(
                interval(0., 1.).unwrap(),
                samples[0].1.clone(),
            )),
            _ => FireworkGradient::ColorSampleUnevenAuto(
                ColorSampleUnevenAutoCurve::new(samples).unwrap(),
            ),
        }
    }

    /// Creates an appropriate curve type based on the number of samples (e.g. if 1 sample, then
    /// constant. If 2 samples, then SampleAutoCurve)
    pub fn even_samples(samples: impl IntoIterator<Item = T>) -> Self {
        let samples = samples.into_iter().collect::<Vec<_>>();
        match samples.len() {
            0 => panic!("Cannot create curve from 0 samples"),
            1 => FireworkGradient::Constant(ConstantCurve::new(
                interval(0., 1.).unwrap(),
                samples[0].clone(),
            )),
            _ => FireworkGradient::ColorSampleAuto(ColorSampleAutoCurve::new(samples).unwrap()),
        }
    }

    pub fn constant(sample: T) -> Self {
        FireworkGradient::Constant(ConstantCurve::new(interval(0., 1.).unwrap(), sample))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_curve_linear_rgba() {
        let curve = FireworkGradient::ColorSampleAuto(
            ColorSampleAutoCurve::new(vec![
                Srgba::new(1.0, 0.0, 0.0, 1.0),
                Srgba::new(0.0, 1.0, 0.0, 1.0),
                Srgba::new(0.0, 0.0, 1.0, 1.0),
            ])
            .unwrap(),
        );
        assert_eq!(curve.sample_unchecked(0.0), Srgba::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(curve.sample_unchecked(0.5), Srgba::new(0.0, 1.0, 0.0, 1.0));
        assert_eq!(curve.sample_unchecked(1.0), Srgba::new(0.0, 0.0, 1.0, 1.0));
    }
}
