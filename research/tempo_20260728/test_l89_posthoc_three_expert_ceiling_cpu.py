import unittest

import numpy as np

from research.tempo_20260728.audit_l89_posthoc_three_expert_ceiling import (
    canonical_event_cluster_bootstrap,
    fixed_fusions,
)
from research.tempo_20260728.tempo_l89_global import metric_bundle, safe_logit


class PosthocThreeExpertCeilingContracts(unittest.TestCase):
    def test_fixed_fusions_use_exact_declared_logit_weights(self):
        p5 = np.asarray([0.2, 0.8], dtype=np.float64)
        d1 = [
            np.asarray([0.3, 0.7], dtype=np.float64),
            np.asarray([0.4, 0.6], dtype=np.float64),
            np.asarray([0.5, 0.5], dtype=np.float64),
        ]
        d7 = np.asarray([0.6, 0.4], dtype=np.float64)
        observed = fixed_fusions(p5, d1, d7)
        d1_logit = np.mean(
            np.stack([safe_logit(value) for value in d1], axis=0), axis=0
        )
        expected_reference_logit = 0.5 * safe_logit(p5) + 0.5 * d1_logit
        expected_candidate_logit = (
            safe_logit(p5) + d1_logit + safe_logit(d7)
        ) / 3.0
        observed_reference_logit = safe_logit(
            observed["p5_d1_equal_logit_reference"]
        )
        observed_candidate_logit = safe_logit(
            observed["p5_d1_d7_equal_thirds_posthoc_ceiling"]
        )
        np.testing.assert_allclose(
            observed_reference_logit, expected_reference_logit, atol=1e-12
        )
        np.testing.assert_allclose(
            observed_candidate_logit, expected_candidate_logit, atol=1e-12
        )

    def test_identical_systems_have_zero_cluster_bootstrap_deltas(self):
        labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
        events = ["a", "a", "b", "b", "c", "d"]
        probability = np.asarray(
            [0.1, 0.2, 0.8, 0.9, 0.3, 0.7], dtype=np.float64
        )
        probabilities = {"reference": probability, "candidate": probability}
        metrics = {
            name: metric_bundle(labels, value, events)
            for name, value in probabilities.items()
        }
        audit = canonical_event_cluster_bootstrap(
            labels,
            events,
            probabilities,
            metrics,
            candidate_name="candidate",
            reference_name="reference",
            replicates=100,
            seed=7,
        )
        for record in audit["candidate_minus_reference"].values():
            self.assertEqual(record["point_delta"], 0.0)
            self.assertEqual(record["ci_95_low"], 0.0)
            self.assertEqual(record["ci_95_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
