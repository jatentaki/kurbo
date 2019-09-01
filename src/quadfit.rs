use crate::{
    Point, QuadBez,
    fitting::{DMatrix, fit, FromPointIter, Constraint, ParamCurveFit}
};
use lazy_static::lazy_static;

lazy_static! {
    static ref M_6: DMatrix = {
        // This matrix is actually block-diagonal and could be simplified as
        //
        // M_6 = [[M, 0],
        //        [0, M]]
        //
        // where M is the matrix defined in
        // https://pomax.github.io/bezierinfo/#curvefitting

        DMatrix::from_row_slice(6, 6, &[
             1.,  0.,  0.,   0.,  0.,  0.,
            -2.,  2.,  0.,   0.,  0.,  0.,
             1., -2.,  1.,   0.,  0.,  0.,

             0.,  0.,  0.,   1.,  0.,  0.,
             0.,  0.,  0.,  -2.,  2.,  0.,
             0.,  0.,  0.,   1., -2.,  1.,
         ])
    };
}

impl FromPointIter for QuadBez {
    fn from_point_iter(mut iter: impl Iterator<Item = Point>) -> Self {
        let p0 = iter.next().unwrap();
        let p1 = iter.next().unwrap();
        let p2 = iter.next().unwrap();
        assert!(iter.next().is_none(), "iterator not exhausted");

        QuadBez { p0, p1, p2 }
    }
}

impl ParamCurveFit for QuadBez {
    type Constraints = [Constraint; 3];

    /// Fit a quadratic Bezier curve which tries to best approximate the points
    /// provided in the `points` argument. Additionally, it accepts an array
    /// of 3 `Constraint`s, one per each control point (`p0`, `p1`, `p2`).
    /// 
    /// Example use below:
    /// ```rust
    /// use kurbo::{Point, QuadBez, fitting::{ParamCurveFit, Constraint::Free}};

    /// let points: &[Point] = &[
    ///     (0., 2.).into(),
    ///     (1., 3.).into(),
    ///     (2., 2.).into(),
    ///     (1., 0.).into(),
    ///     (3., 3.).into(),
    /// ];

    /// let cubic_fit = QuadBez::fit(points, &[Free, Free, Free]);
    /// ```
    fn fit(points: &[Point], constraints: &Self::Constraints) -> (f64, Self) {
        fit::<QuadBez>(points, constraints, 3, &M_6)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        Point, QuadBez, assert_abs_diff_eq,
        fitting::{Constraint, ParamCurveFit},
    };
    use Constraint::*;

    #[test]
    fn test_fitting_fixed() {
        let points = [
            Point::new(0., 0.),
            Point::new(90., 60.),
            Point::new(30., 10.),
            Point::new(50., 30.),
            Point::new(60., 20.),
            Point::new(80., 15.),
            Point::new(65., 40.)
        ];

        let constraints: [Constraint; 3] = [
            Fixed((5., 5.).into()),
            Free,
            Free,
        ];
        
        let (_error, curve) = QuadBez::fit(&points, &constraints);
        assert_eq!(curve.p0, Point::new(5., 5.));
    }

    #[test]
    fn test_fitting_3() {
        let points = [
            Point::new(5., 7.),
            Point::new(3., 2.),
            Point::new(90., 60.),
        ];

        let constraints = [Free, Free, Free];

        let (error, _curve) = QuadBez::fit(&points, &constraints);
        assert!(error < 5.);
    }

    #[test]
    fn test_fitting_2() {
        let points = [
            Point::new(5., 7.),
            Point::new(3., 2.),
        ];

        let constraints = [Free, Free, Free];

        let (error, _curve) = QuadBez::fit(&points, &constraints);
        assert!(error < 5.);
    }

    #[test]
    fn test_fitting_no_dof() {
        let points = [
            Point::new(0., 0.),
            Point::new(90., 60.),
            Point::new(30., 10.),
            Point::new(50., 30.),
            Point::new(60., 20.),
            Point::new(80., 15.),
            Point::new(65., 40.)
        ];

        let ctrl_points = [
            Point::new(0., 0.),
            Point::new(1., 0.),
            Point::new(3., 3.),

        ];

        let constraints: [Constraint; 3] = [
            ctrl_points[0].into(),
            ctrl_points[1].into(),
            ctrl_points[2].into(),
        ];

        let (_error, curve) = QuadBez::fit(&points, &constraints);
        let expected = QuadBez::new(
            ctrl_points[0],
            ctrl_points[1],
            ctrl_points[2],
        );

        assert_abs_diff_eq!(curve, expected);
    }

    #[test]
    fn test_fitting_line() {
        let points = [
            Point::new(0., 0.),
            Point::new(90., 60.),
            Point::new(30., 10.),
            Point::new(50., 30.),
            Point::new(60., 20.),
            Point::new(80., 15.),
            Point::new(65., 40.)
        ];

        let constraints: [Constraint; 3] = [
            Line(Point::new(0., 0.), Point::new(1., 0.)),
            Free,
            Line(Point::new(0., 0.), Point::new(1., 1.)),
        ];
        
        let (_error, curve) = QuadBez::fit(&points, &constraints);

        assert_abs_diff_eq!(curve.p0.y, 0.);

        let p2 = curve.p2;
        assert_abs_diff_eq!(p2.x - p2.y, 0.);
    }
}
