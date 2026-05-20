// ═══════════════════════════════════════════════════════════════
// MHT-CET 2026 ML Prediction Engine
// Techniques: Cubic Spline Interpolation, Multi-Feature Ridge
// Regression, Isotonic Regression (PAVA), Residual-Based CI
// ═══════════════════════════════════════════════════════════════

const MLEngine = (() => {

  // ── Raw 2025 anchor data (marks, percentile) ──
  const rawAnchors = [
    [0,0],[10,2],[20,8],[30,15],[40,22],[50,30],[55,42],[57,60],
    [60,52],[62,43],[63,69],[65,72],[66,72],[69,72],[70,77],
    [74,83],[75,86],[76,81],[77,74],[78,87],[80,87],[83,87],
    [83,91],[85,90],[88,93],[88,99],[89,88],[91,92],[92,93],
    [93,95],[94,93],[94,99.4],[96,93],[97,97],[98,94],[98,99],
    [99,95],[99,99],[102,95],[104,95],[109,96],[112,97],[113,97],
    [114,97.5],[115,98],[116,97],[117,97],[118,97.9],[120,98.5],
    [122,98.4],[124,98.5],[125,98.3],[126,98.8],[127,98.4],
    [128,99],[130,98.3],[130,98.6],[131,98.8],[132,99],[133,99.1],
    [134,99],[135,99.3],[136,99],[139,99.2],[140,99.5],[141,99.5],
    [142,99.5],[144,99.6],[146,99.6],[147,99.6],[149,99.5],
    [150,99.6],[151,99.6],[153,99.7],[154,99.6],[155,99.7],
    [160,99.86],[167,99.9],[172,99.94],[178,99.97],[200,99.99]
  ];

  // ── Step 1: Aggregate duplicates by averaging ──
  function aggregateAnchors(data) {
    const map = new Map();
    data.forEach(([m, p]) => {
      if (!map.has(m)) map.set(m, []);
      map.get(m).push(p);
    });
    const out = [];
    for (const [m, ps] of map) {
      out.push([m, ps.reduce((a, b) => a + b, 0) / ps.length]);
    }
    out.sort((a, b) => a[0] - b[0]);
    return out;
  }

  // ── Step 2: Isotonic Regression (PAVA) ──
  // Ensures monotonically non-decreasing percentiles
  function isotonicRegression(points) {
    const n = points.length;
    const y = points.map(p => p[1]);
    const w = new Array(n).fill(1);
    const blocks = y.map((v, i) => ({ sum: v, weight: 1, start: i, end: i }));
    const merged = [blocks[0]];

    for (let i = 1; i < n; i++) {
      merged.push(blocks[i]);
      while (merged.length > 1) {
        const last = merged[merged.length - 1];
        const prev = merged[merged.length - 2];
        if (prev.sum / prev.weight > last.sum / last.weight) {
          prev.sum += last.sum;
          prev.weight += last.weight;
          prev.end = last.end;
          merged.pop();
        } else break;
      }
    }

    const result = points.map(p => [p[0], 0]);
    merged.forEach(block => {
      const val = block.sum / block.weight;
      for (let i = block.start; i <= block.end; i++) {
        result[i][1] = val;
      }
    });
    return result;
  }

  // ── Step 3: Cubic Spline Interpolation ──
  // Natural cubic spline through the isotonic-regressed anchor points
  function buildCubicSpline(points) {
    const n = points.length;
    const x = points.map(p => p[0]);
    const y = points.map(p => p[1]);

    if (n < 2) return (_m) => 0;
    if (n === 2) {
      const slope = (y[1] - y[0]) / (x[1] - x[0]);
      return (m) => y[0] + slope * (m - x[0]);
    }

    const h = [];
    for (let i = 0; i < n - 1; i++) h.push(x[i + 1] - x[i]);

    // Tridiagonal system for natural spline
    const alpha = [0];
    for (let i = 1; i < n - 1; i++) {
      alpha.push((3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1]));
    }

    const l = [1], mu = [0], z = [0];
    for (let i = 1; i < n - 1; i++) {
      l.push(2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]);
      mu.push(h[i] / l[i]);
      z.push((alpha[i] - h[i - 1] * z[i - 1]) / l[i]);
    }

    const c = new Array(n).fill(0);
    const b = new Array(n - 1).fill(0);
    const d = new Array(n - 1).fill(0);

    for (let j = n - 2; j >= 0; j--) {
      c[j] = z[j] - mu[j] * c[j + 1];
      b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
      d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
    }

    return function splineEval(m) {
      if (m <= x[0]) return y[0];
      if (m >= x[n - 1]) return y[n - 1];

      // Binary search for interval
      let lo = 0, hi = n - 2;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (x[mid + 1] < m) lo = mid + 1;
        else hi = mid;
      }

      const dx = m - x[lo];
      return y[lo] + b[lo] * dx + c[lo] * dx * dx + d[lo] * dx * dx * dx;
    };
  }

  // ── Step 4: Multi-Feature Ridge Regression for Shift Difficulty ──
  // Features: [avg, median, maths_avg, physics_avg, chem_avg]
  // Target: difficulty delta (how much to shift the curve)
  // We use ridge regression with a small lambda for regularization

  function computeFeatureStats(shifts) {
    const featureNames = ['avg', 'med', 'mat', 'phy', 'che'];
    const means = {};
    const stds = {};

    featureNames.forEach(f => {
      const vals = shifts.map(s => s[f]);
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const std = Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length) || 1;
      means[f] = mean;
      stds[f] = std;
    });

    return { means, stds, featureNames };
  }

  function normalizeFeatures(shift, stats) {
    return stats.featureNames.map(f =>
      (shift[f] - stats.means[f]) / stats.stds[f]
    );
  }

  // Ridge regression: w = (X^T X + λI)^(-1) X^T y
  function ridgeRegression(X, y, lambda = 0.5) {
    const n = X.length;
    const p = X[0].length;

    // X^T X
    const XtX = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        for (let k = 0; k < n; k++) {
          XtX[i][j] += X[k][i] * X[k][j];
        }
      }
    }

    // Add regularization
    for (let i = 0; i < p; i++) XtX[i][i] += lambda;

    // X^T y
    const Xty = new Array(p).fill(0);
    for (let i = 0; i < p; i++) {
      for (let k = 0; k < n; k++) {
        Xty[i] += X[k][i] * y[k];
      }
    }

    // Solve via Cholesky or simple Gaussian elimination for small p
    return solveLinearSystem(XtX, Xty);
  }

  function solveLinearSystem(A, b) {
    const n = A.length;
    const aug = A.map((row, i) => [...row, b[i]]);

    for (let col = 0; col < n; col++) {
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
      }
      [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

      if (Math.abs(aug[col][col]) < 1e-12) continue;

      for (let row = col + 1; row < n; row++) {
        const factor = aug[row][col] / aug[col][col];
        for (let j = col; j <= n; j++) aug[row][j] -= factor * aug[col][j];
      }
    }

    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      x[i] = aug[i][n];
      for (let j = i + 1; j < n; j++) x[i] -= aug[i][j] * x[j];
      x[i] /= aug[i][i] || 1;
    }
    return x;
  }

  // ── Step 5: Residual-based Confidence Intervals ──
  // Compute residuals from spline vs raw data, then estimate CI as a function of marks

  function buildResidualModel(splineFn, rawData) {
    // Group residuals into buckets
    const buckets = new Map();
    const bucketSize = 20;

    rawData.forEach(([m, p]) => {
      const predicted = splineFn(m);
      const residual = Math.abs(p - predicted);
      const bucket = Math.floor(m / bucketSize) * bucketSize;
      if (!buckets.has(bucket)) buckets.set(bucket, []);
      buckets.get(bucket).push(residual);
    });

    // Compute 95th percentile of residuals per bucket
    const bucketCI = [];
    for (const [bucket, residuals] of buckets) {
      residuals.sort((a, b) => a - b);
      const idx95 = Math.min(residuals.length - 1, Math.floor(residuals.length * 0.95));
      const ci95 = residuals[idx95];
      // Also compute std
      const mean = residuals.reduce((a, b) => a + b, 0) / residuals.length;
      const std = Math.sqrt(residuals.reduce((a, v) => a + (v - mean) ** 2, 0) / residuals.length);
      bucketCI.push([bucket + bucketSize / 2, 1.96 * std, ci95]);
    }

    // Return a function that interpolates CI for any marks value
    bucketCI.sort((a, b) => a[0] - b[0]);

    return function getCI(marks) {
      if (bucketCI.length === 0) return 1.5;
      if (marks <= bucketCI[0][0]) return Math.max(bucketCI[0][1], bucketCI[0][2]);
      if (marks >= bucketCI[bucketCI.length - 1][0]) {
        const last = bucketCI[bucketCI.length - 1];
        return Math.max(last[1], last[2]);
      }

      for (let i = 0; i < bucketCI.length - 1; i++) {
        if (marks >= bucketCI[i][0] && marks < bucketCI[i + 1][0]) {
          const t = (marks - bucketCI[i][0]) / (bucketCI[i + 1][0] - bucketCI[i][0]);
          const ci1 = Math.max(bucketCI[i][1], bucketCI[i][2]);
          const ci2 = Math.max(bucketCI[i + 1][1], bucketCI[i + 1][2]);
          return ci1 + t * (ci2 - ci1);
        }
      }
      return 1.5;
    };
  }

  // ── Step 6: Kernel Density weighting for local smoothing ──
  function gaussianKernel(x, bandwidth) {
    return Math.exp(-0.5 * (x / bandwidth) ** 2) / (bandwidth * Math.sqrt(2 * Math.PI));
  }

  function localWeightedAverage(targetX, data, bandwidth = 8) {
    let wSum = 0, wySum = 0;
    data.forEach(([x, y]) => {
      const w = gaussianKernel(targetX - x, bandwidth);
      wSum += w;
      wySum += w * y;
    });
    return wSum > 0 ? wySum / wSum : 0;
  }

  // ── Build the full model ──
  function buildModel(shifts) {
    // 1. Aggregate and make isotonic
    const aggregated = aggregateAnchors(rawAnchors);
    const isotonic = isotonicRegression(aggregated);

    // 2. Build cubic spline on isotonic data
    const baseSpline = buildCubicSpline(isotonic);

    // 3. Build residual-based CI model
    const ciModel = buildResidualModel(baseSpline, rawAnchors);

    // 4. Multi-feature ridge regression for shift difficulty
    const fStats = computeFeatureStats(shifts);
    const overallAvg = shifts.reduce((s, x) => s + x.avg, 0) / shifts.length;
    const overallMed = shifts.reduce((s, x) => s + x.med, 0) / shifts.length;

    // Create training targets: difficulty delta based on how far each shift
    // deviates from global average (proxy for true difficulty)
    const X = shifts.map(s => normalizeFeatures(s, fStats));
    // Target: combined difficulty signal from multiple indicators
    const y = shifts.map(s => {
      const avgDelta = (overallAvg - s.avg);
      const medDelta = (overallMed - s.med);
      const matDelta = (fStats.means.mat - s.mat);
      // Composite difficulty signal (weighted combination)
      return avgDelta * 0.5 + medDelta * 0.35 + matDelta * 0.15;
    });

    const weights = ridgeRegression(X, y, 0.5);

    // 5. Difficulty function
    function getShiftDelta(shift) {
      const features = normalizeFeatures(shift, fStats);
      let delta = 0;
      for (let i = 0; i < features.length; i++) delta += weights[i] * features[i];
      // Scale factor: how much the delta translates to marks adjustment
      return delta * 0.22;
    }

    // 6. Difficulty index (0-100 scale, higher = harder)
    function getDifficultyIndex(shift) {
      const delta = getShiftDelta(shift);
      // Normalize: delta ranges roughly from -3 to +3, map to 0-100
      return Math.min(100, Math.max(0, 50 + delta * 12));
    }

    // 7. Final prediction with ensemble: spline + local weighted average
    function predict(marks, shift) {
      const delta = getShiftDelta(shift);
      const adjMarks = marks + delta;

      // Ensemble: 75% spline, 25% kernel-smoothed local estimate
      const splineP = baseSpline(adjMarks);
      const localP = localWeightedAverage(adjMarks, isotonic, 10);
      const ensemble = splineP * 0.75 + localP * 0.25;

      return Math.max(0, Math.min(99.99, ensemble));
    }

    // 8. Confidence interval
    function getConfidenceInterval(marks, shift) {
      const delta = getShiftDelta(shift);
      const adjMarks = marks + delta;
      // Base CI from residual model
      let ci = ciModel(adjMarks);
      // Add uncertainty from shift adjustment magnitude
      ci += Math.abs(delta) * 0.08;
      // Clamp to reasonable range
      return Math.max(0.3, Math.min(3.5, ci));
    }

    return { predict, getShiftDelta, getDifficultyIndex, getConfidenceInterval, weights, fStats };
  }

  // ── Band classification ──
  function bandInfo(pct) {
    if (pct >= 99.5) return ['99.5+ Elite', 'p-top'];
    if (pct >= 99)   return ['99+ Tier', 'p-top'];
    if (pct >= 97)   return ['Very Strong', 'p-vstrong'];
    if (pct >= 93)   return ['Competitive', 'p-comp'];
    if (pct >= 85)   return ['Average', 'p-avg'];
    return ['Below Avg', 'p-low'];
  }

  function diffColor(idx) {
    if (idx > 65) return '#f05252';
    if (idx > 50) return '#f6a623';
    if (idx > 35) return '#4f8ef7';
    return '#43d985';
  }

  return { buildModel, bandInfo, diffColor };

})();
