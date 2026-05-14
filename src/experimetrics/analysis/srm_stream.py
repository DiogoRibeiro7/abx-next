"""Incremental SRM (Sample Ratio Mismatch) detection for streaming data."""

from __future__ import annotations
from scipy.stats import chisquare

from ..core.validate import ensure_probability

class SrmWatch:
	"""Incremental SRM detector that can be updated with new count deltas.
	This class maintains running counts and performs SRM tests as new data arrives,
	allowing for early detection of sample ratio mismatches during experiment execution.

	Args:
		p_expected: Expected proportion of control group (default 0.5 for 1:1 split)
		warn_p: P-value threshold for SRM warning (default 1e-3)
	Example:
		>>> watch = SrmWatch(p_expected=0.5, warn_p=1e-3)
		>>> result = watch.update(nc_delta=50, nt_delta=45)
		>>> if result["srm_detected"]:
		...     print(f"SRM detected! p-value: {result['pvalue']:.6f}")
	"""
	def __init__(self, p_expected: float = 0.5, warn_p: float = 1e-3) -> None:
		"""Initialize SRM watch with expected proportion and warning threshold.

		Args:
			p_expected: Expected proportion of control group (between 0 and 1)
			warn_p: P-value threshold below which SRM is flagged
		Raises:
			ValueError: If parameters are invalid
		"""
		ensure_probability(p_expected, "p_expected")
		if not (0 < warn_p < 1):
			raise ValueError("warn_p must be between 0 and 1")

		self.p_expected = p_expected
		self.warn_p = warn_p

		# Running counts
		self.n_control = 0
		self.n_treatment = 0
		# State tracking
		self.srm_detected = False
		self.detection_update = None  # Update number when SRM was first detected
		self.update_count = 0

	def update(self, nc_delta: int, nt_delta: int) -> dict[str, float | int | bool]:
		"""Update counts and perform SRM test.

		Args:
			nc_delta: New control group count increment (non-negative)
			nt_delta: New treatment group count increment (non-negative)

		Returns:
			Dictionary containing:
				- n_control: Total control count
				- n_treatment: Total treatment count
				- n_total: Total sample size
				- expected_control: Expected control count given current total
				- expected_treatment: Expected treatment count given current total
				- chi2: Chi-square test statistic
				- pvalue: P-value from chi-square test
				- srm_detected: Whether SRM is detected (p < warn_p)
				- detection_update: Update number when SRM was first detected (if any)
				- update_count: Number of updates performed

		Raises:
			ValueError: If delta counts are negative
		"""
		if nc_delta < 0:
			raise ValueError("nc_delta must be non-negative")
		if nt_delta < 0:
			raise ValueError("nt_delta must be non-negative")

		# Update counts
		self.n_control += nc_delta
		self.n_treatment += nt_delta
		self.update_count += 1

		# Calculate current totals and expected values
		n_total = self.n_control + self.n_treatment

		# Skip test if no samples yet
		if n_total == 0:
			return {
				"n_control": self.n_control,
				"n_treatment": self.n_treatment,
				"n_total": n_total,
				"expected_control": 0.0,
				"expected_treatment": 0.0,
				"chi2": 0.0,
				"pvalue": 1.0,
				"srm_detected": False,
				"detection_update": self.detection_update,
				"update_count": self.update_count,
			}

		expected_control = n_total * self.p_expected
		expected_treatment = n_total * (1.0 - self.p_expected)

		# Perform chi-square test
		chi2, pvalue = chisquare(
			[self.n_control, self.n_treatment],
			f_exp=[expected_control, expected_treatment]
		)

		# Check for SRM detection
		current_srm_detected = pvalue < self.warn_p

		# Update SRM detection state
		if current_srm_detected and not self.srm_detected:
			self.srm_detected = True
			self.detection_update = self.update_count
		elif not current_srm_detected and self.srm_detected:
			# SRM previously detected but p-value is now above threshold
			# Keep the detection flag but note this in the result
			pass

		return {
			"n_control": self.n_control,
			"n_treatment": self.n_treatment,
			"n_total": n_total,
			"expected_control": float(expected_control),
			"expected_treatment": float(expected_treatment),
			"chi2": float(chi2),
			"pvalue": float(pvalue),
			"srm_detected": current_srm_detected,
			"detection_update": self.detection_update,
			"update_count": self.update_count,
		}

	def reset(self) -> None:
		"""Reset the SRM watch to initial state."""
		self.n_control = 0
		self.n_treatment = 0
		self.srm_detected = False
		self.detection_update = None
		self.update_count = 0

	def get_current_status(self) -> dict[str, float | int | bool]:
		"""Get current status without updating counts.

		Returns:
			Current SRM test results based on accumulated counts
		"""
		# Calculate current totals and expected values
		n_total = self.n_control + self.n_treatment

		# Skip test if no samples yet
		if n_total == 0:
			return {
				"n_control": self.n_control,
				"n_treatment": self.n_treatment,
				"n_total": n_total,
				"expected_control": 0.0,
				"expected_treatment": 0.0,
				"chi2": 0.0,
				"pvalue": 1.0,
				"srm_detected": False,
				"detection_update": self.detection_update,
				"update_count": self.update_count,
			}

		expected_control = n_total * self.p_expected
		expected_treatment = n_total * (1.0 - self.p_expected)

		# Perform chi-square test
		chi2, pvalue = chisquare(
			[self.n_control, self.n_treatment],
			f_exp=[expected_control, expected_treatment]
		)

		# Check current SRM status
		current_srm_detected = pvalue < self.warn_p

		return {
			"n_control": self.n_control,
			"n_treatment": self.n_treatment,
			"n_total": n_total,
			"expected_control": float(expected_control),
			"expected_treatment": float(expected_treatment),
			"chi2": float(chi2),
			"pvalue": float(pvalue),
			"srm_detected": current_srm_detected,
			"detection_update": self.detection_update,
			"update_count": self.update_count,
		}

	def get_ratio_deviation(self) -> float | None:
		"""Calculate current deviation from expected ratio.

		Returns:
			Absolute difference between observed and expected control proportion,
			or None if no samples yet
		"""
		n_total = self.n_control + self.n_treatment
		if n_total == 0:
			return None

		observed_proportion = self.n_control / n_total
		return abs(observed_proportion - self.p_expected)

	def __repr__(self) -> str:
		"""String representation of SrmWatch state."""
		return (
			f"SrmWatch(p_expected={self.p_expected}, warn_p={self.warn_p}, "
			f"n_control={self.n_control}, n_treatment={self.n_treatment}, "
			f"srm_detected={self.srm_detected})"
		)
