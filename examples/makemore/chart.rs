//! A tiny hand-rolled terminal scatter chart: labeled points on a
//! bordered character grid.
//!
//! Labels are the marks — an embedding map is legible only when each
//! point shows which token it is, which is exactly what dot-plotting
//! libraries cannot draw. A label may carry ANSI styling; escape
//! sequences occupy no cells, so styled labels pass through untouched.

/// Renders `points` — a position plus a one-cell label — into a
/// `columns` by `rows` bordered character grid, scaling the points'
/// bounding box to the grid. A flat axis widens to a unit span so a
/// degenerate cloud still renders.
///
/// A point whose cell is taken moves to the nearest free cell within a
/// small ring, so near-identical points stay individually visible at
/// the price of a slight positional lie; a point with no free cell in
/// the ring is dropped.
///
/// # Panics
/// Panics if `points` is empty or the grid has no cells.
pub fn scatter(points: &[(f64, f64, String)], columns: usize, rows: usize) -> String {
    assert!(!points.is_empty(), "the chart needs at least one point");
    assert!(columns > 0 && rows > 0, "the chart needs a non-empty grid");

    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &(x, y, _) in points {
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    let x_span = if x_max > x_min { x_max - x_min } else { 1.0 };
    let y_span = if y_max > y_min { y_max - y_min } else { 1.0 };

    let mut grid: Vec<Vec<Option<&str>>> = vec![vec![None; columns]; rows];
    for (x, y, label) in points {
        let column = ((x - x_min) / x_span * (columns - 1) as f64).round() as isize;
        // The vertical axis flips: larger `y` renders higher on screen.
        let row = ((y_max - y) / y_span * (rows - 1) as f64).round() as isize;
        'placed: for radius in 0..=3_isize {
            for row_offset in -radius..=radius {
                for column_offset in -radius..=radius {
                    if row_offset.abs().max(column_offset.abs()) != radius {
                        continue;
                    }
                    let target_row = row + row_offset;
                    let target_column = column + column_offset;
                    if target_row < 0
                        || target_row >= rows as isize
                        || target_column < 0
                        || target_column >= columns as isize
                    {
                        continue;
                    }
                    let cell = &mut grid[target_row as usize][target_column as usize];
                    if cell.is_none() {
                        *cell = Some(label);
                        break 'placed;
                    }
                }
            }
        }
    }

    let mut output = String::new();
    output.push('+');
    output.push_str(&"-".repeat(columns));
    output.push_str("+\n");
    for row in &grid {
        output.push('|');
        for cell in row {
            match cell {
                Some(label) => output.push_str(label),
                None => output.push(' '),
            }
        }
        output.push_str("|\n");
    }
    output.push('+');
    output.push_str(&"-".repeat(columns));
    output.push('+');
    output
}
